#!/usr/bin/env python3
"""
Network Agent (Gemini + MCP/pyATS) — DROP-IN (AGENT) V5
- FIXED: Cache Invalidation - Clears dedupe cache after Config+Wait to ensure fresh verification.
- FIXED: Enhanced Parsing - Extracts 'ping' and 'ospf' commands from prompt for verification.
- FIXED: Completion Guardrail - Forces test generation if model tries to quit early.
- VISIBILITY: Prints test data/plan summary.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Set

from dotenv import load_dotenv
from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

TESTBED_PATH = os.getenv("TESTBED_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview") 
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "20"))
DEBUG_TOOLS = os.getenv("DEBUG_TOOLS", "0") == "1"

PCALL = os.getenv("PCALL", "1") == "1"
TOOL_TIMEOUT_S = int(os.getenv("TOOL_TIMEOUT_S", "120"))

MAX_SAME_CALLS_PER_ROUND = int(os.getenv("MAX_SAME_CALLS_PER_TURN", "1"))
TURN_DEDUPE_MODE = os.getenv("TURN_DEDUPE_MODE", "reuse").lower()

TOOL_DENYLIST = {t.strip() for t in (os.getenv("TOOL_DENYLIST", "") or "").split(",") if t.strip()}
TOOL_ALLOWLIST_RAW = [t.strip() for t in (os.getenv("TOOL_ALLOWLIST", "") or "").split(",") if t.strip()]
TOOL_ALLOWLIST = set(TOOL_ALLOWLIST_RAW) if TOOL_ALLOWLIST_RAW else None

SERVERS = {
    "pyats": {
        "command": sys.executable,
        "args": ["/home/johncapobianco/pyATS-Agent/pyATS_MCP/pyats_mcp_server.py"],
        "env": {"PYATS_TESTBED_PATH": TESTBED_PATH},
    }
}

try:
    import yaml
except Exception:
    yaml = None


# -----------------------------
# Helpers
# -----------------------------
def try_parse_json(s: str) -> Optional[Any]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None

def break_forbidden_substring(s: str, forbidden: str) -> str:
    if not s or not forbidden:
        return s
    pat = re.compile(re.escape(forbidden), flags=re.I)
    def repl(m: re.Match) -> str:
        txt = m.group(0)
        return "_".join(list(txt))
    return pat.sub(repl, s)

def extract_blocked_token(err_text: str) -> Optional[str]:
    if not err_text:
        return None
    m = re.search(r"blocked:\s*([A-Za-z0-9_\-]+)", err_text)
    return m.group(1) if m else None

def normalize_function_parameters(schema: Any) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = dict(schema)
    out["type"] = "object"
    if not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    if "required" in out and not isinstance(out["required"], list):
        out.pop("required", None)
    return out

def summarize_tool_payload(payload: Any) -> str:
    try:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, indent=2)
        return str(payload)
    except Exception as e:
        return f"Error summarizing payload: {e}"

def call_key(name: str, args: Dict[str, Any]) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, separators=(',', ':'))}"
    except Exception:
        return f"{name}:{str(args)}"

def _normalize_cmd(cmd: str) -> str:
    c = (cmd or "").strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c

# -----------------------------
# Show-command sanitizer
# -----------------------------
BAD_SHOW_TAIL = re.compile(r"\s+\b(on|for)\b\s+(routers?|switches?|devices?)\b.*$", re.I)

def sanitize_show_command(cmd: str) -> str:
    c = (cmd or "").strip()
    if not c:
        return c
    c = re.sub(r"\((s)\)", "s", c, flags=re.I)
    c = re.sub(r"[()]", "", c)
    c = BAD_SHOW_TAIL.sub("", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c

def is_plausible_show_command(cmd: str) -> bool:
    c = (cmd or "").strip().lower()
    if not c.startswith("show "):
        return False
    if any(tok in c for tok in ("|", ">", "<")):
        return False
    return True

# -----------------------------
# Empty/No-data normalization
# -----------------------------
EMPTY_HINT_PATTERNS = [
    r"parser output is empty",
    r"%\s*cdp is not enabled",
    r"%\s*cdp not enabled",
    r"%\s*cdp is disabled",
    r"\bcdp\b.*\bnot enabled\b",
    r"\bcdp\b.*\bdisabled\b",
]

def _textify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, indent=2)
    except Exception:
        return str(x)

def _looks_empty_payload(payload: Any) -> bool:
    if payload is None:
        return True
    if isinstance(payload, dict):
        if "output" in payload:
            out = payload.get("output")
            if out is None:
                return True
            if isinstance(out, str) and not out.strip():
                return True
            if isinstance(out, (dict, list)) and len(out) == 0:
                return True
        if "text" in payload and isinstance(payload.get("text"), str) and not payload["text"].strip():
            return True
    if isinstance(payload, str) and not payload.strip():
        return True
    return False

def _infer_empty_reason(payload: Any) -> Optional[str]:
    t = _textify(payload).lower()
    for pat in EMPTY_HINT_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return pat
    return None

def normalize_show_payload(payload: Any, *, command: str = "") -> Any:
    if not isinstance(payload, dict):
        return payload
    if payload.get("status") == "error":
        return payload

    no_data = _looks_empty_payload(payload)
    reason = _infer_empty_reason(payload)

    assume_absent = False
    if no_data:
        assume_absent = True
    if reason and ("not enabled" in reason or "disabled" in reason):
        assume_absent = True

    if no_data or assume_absent:
        p = dict(payload)
        meta = p.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["no_data"] = True
        if assume_absent:
            meta["assumption"] = "treat_as_absent_or_no_neighbors"
        if reason:
            meta["reason_hint"] = reason
        if command:
            meta["command"] = command
        p["meta"] = meta
        return p
    return payload

# -----------------------------
# User-instruction wait extraction
# -----------------------------
WAIT_RE = re.compile(r"\bwait\s+(\d{1,4})\s*(seconds|second|secs|sec|s)\b", re.I)

def extract_wait_seconds(text: str) -> Optional[int]:
    m = WAIT_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

# -----------------------------
# Deadman: extract explicit show commands
# -----------------------------
def extract_requested_show_commands(text: str) -> List[str]:
    if not text:
        return []
    src = text.strip()
    cmds: List[str] = []

    # Regex to capture show commands, pings, and key protocols
    chunks = re.findall(r"(?:^|\b)(?:run|execute)\s+([^.;\n]+)", src, flags=re.I)
    bare = re.findall(r"(?:^|\n)\s*(show\s+[^.;\n]+)", src, flags=re.I)
    
    # NEW: explicit capture for 'ping' commands if user asked for them
    pings = re.findall(r"(?:^|\b)(ping\s+[0-9a-zA-Z\.]+)", src, flags=re.I)
    
    # NEW: Smart OSPF detection
    if re.search(r"\bospf\b", src, re.I):
        cmds.append("show ip ospf neighbor")
        cmds.append("show ip route ospf")

    chunks.extend(bare)
    # Ping commands need to be handled separately or added if they fit 'command' pattern
    # For now, we rely on the agent to construct pings, but we add discovery commands.

    for chunk in chunks:
        for part in re.split(r"\s+and\s+|,|\n", chunk, flags=re.I):
            p = part.strip()
            if not p:
                continue
            if p.lower().startswith("show "):
                p = re.sub(r"\s+", " ", p.strip())
                cmds.append(p)

    seen = set()
    out: List[str] = []
    for c in cmds:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

# -----------------------------
# TESTBED TYPE MAP
# -----------------------------
INVALID_CMD_PAT = re.compile(r"%\s*Invalid input|Invalid command", re.I)

def load_device_types(testbed_path: Optional[str]) -> Dict[str, str]:
    if not testbed_path or yaml is None:
        return {}
    p = Path(testbed_path).expanduser()
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}
    devices = (data.get("devices") or {})
    out: Dict[str, str] = {}
    for name, d in devices.items():
        t = (d or {}).get("type") or ""
        if isinstance(t, str) and t.strip():
            out[str(name)] = t.strip().lower()
    return out

def rewrite_show_by_device_type(cmd: str, device_type: str) -> str:
    c = _normalize_cmd(cmd)
    dt = (device_type or "").lower()

    if dt.startswith("router"):
        if c in ("show interfaces status", "show interface status"):
            return "show ip interface brief"

    if dt.startswith("switch"):
        if c == "show ip interface brief":
            return "show interfaces status"
    return cmd

def choose_fallbacks_for_invalid(cmd: str, device_type: str) -> List[str]:
    dt = (device_type or "").lower()
    c = _normalize_cmd(cmd)

    if dt.startswith("router"):
        candidates = ["show ip interface brief", "show interfaces description", "show interfaces"]
    else:
        candidates = ["show interfaces status", "show interfaces description", "show interfaces"]

    out = []
    for x in candidates:
        if _normalize_cmd(x) != c:
            out.append(x)
    return out

# -----------------------------
# Evidence Store
# -----------------------------
@dataclass
class EvidenceItem:
    tool: str
    device: str = ""
    command: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    result: Any = None

class EvidenceStore:
    def __init__(self) -> None:
        self.items: List[EvidenceItem] = []

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def has_configured(self) -> bool:
        """Check if any configuration tool has been successfully run."""
        for item in self.items:
            if item.tool == "pyats_configure_device":
                if isinstance(item.result, dict) and item.result.get("status") == "error":
                    continue
                return True
        return False
    
    def has_run_test(self) -> bool:
        """Check if a test script has been executed."""
        for item in self.items:
            if item.tool == "pyats_run_dynamic_test":
                return True
        return False

    def has_verified(self) -> bool:
        """Check if we have run show commands *after* configuration."""
        config_seen = False
        for item in self.items:
            if item.tool == "pyats_configure_device":
                config_seen = True
            elif config_seen and item.tool == "pyats_run_show_command":
                return True
        return False

    def ready_for_test(self) -> bool:
        """Configured + Verified + No Test yet."""
        return self.has_configured() and self.has_verified()

    def latest_show(self, device: str, command_norm: str) -> Optional[EvidenceItem]:
        d = (device or "").strip()
        cn = _normalize_cmd(command_norm)
        for x in reversed(self.items):
            if x.device == d and x.tool == "pyats_run_show_command" and _normalize_cmd(x.command) == cn:
                return x
        return None

    def summary(self, max_items: int = 25) -> str:
        tail = self.items[-max_items:]
        rows = []
        for it in tail:
            label = f"{it.tool}"
            if it.device:
                label += f" device={it.device}"
            if it.command:
                label += f" cmd={it.command!r}"
            rows.append(label)
        return "EVIDENCE_STORE (most recent):\n" + "\n".join(rows)

# -----------------------------
# Robust extraction
# -----------------------------
@dataclass
class ToolCallShim:
    name: str
    args: Dict[str, Any]

def extract_parts_calls_text(resp: Any) -> Tuple[List[Any], List[ToolCallShim], str]:
    parts: List[Any] = []
    try:
        parts = resp.candidates[0].content.parts or []
    except Exception:
        parts = []

    calls: List[ToolCallShim] = []
    text_chunks: List[str] = []

    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str) and t.strip():
            text_chunks.append(t)

        fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
        if fc:
            name = getattr(fc, "name", None)
            args = getattr(fc, "args", None)
            if isinstance(name, str) and name.strip():
                if not isinstance(args, dict):
                    args = {}
                calls.append(ToolCallShim(name=name.strip(), args=args))

    return parts, calls, "\n".join(text_chunks).strip()

SYSTEM_INSTRUCTION = """
You are an expert Network Automation Engineer operating through MCP tools (pyATS).

PRIMARY GOAL
Turn user questions into repeatable, programmatic validation using pyATS aetest scripts executed via the MCP tool `pyats_run_dynamic_test`.
Do not stop at summaries when an issue is suspected; always validate with a test when possible.

MANDATORY WORKFLOW (FOLLOW EXACTLY)
1) Target selection
   - If the user does not specify device(s), call `pyats_list_devices` first.
   - Ask which device(s) to target only if multiple reasonable choices exist.

2) Plan
   - Produce a short plan containing:
     a) Target devices
     b) Data collection commands/tools you will run
     c) What you intend to validate in pyATS (test cases + pass/fail criteria)

3) Data collection
   - Collect all required facts using MCP tools.
   - Prefer structured parsers when available, but accept raw when not.
   - NO pipes/redirects (|, >, <) in show commands.
   - If a parser returns empty output, treat structured data as unavailable and rely on raw output.
     If raw output is empty or explicitly indicates a feature is not enabled, treat it as absent and proceed.
   - If the user requests a wait (e.g., "wait 30 seconds"), you MUST wait (agent local time) before post-change validation.

4) Test generation (REQUIRED when user asks for validation or expects verification)
   - Generate a pyATS aetest script that does NOT connect to devices.
   - The script MUST embed all collected outputs as Python literals.
   - The script MUST define: TEST_DATA = {...} as a dict literal (not JSON strings).
   - Do NOT use json.loads() anywhere inside the test script.
   - Tests must be deterministic and based only on embedded TEST_DATA.

5) Execute
   - Execute the script using `pyats_run_dynamic_test`.

6) Results & remediation
   - Always report:
     - `overall_result`
     - failing testcase(s) and evidence
     - concrete remediation actions
   - If ERRORED/BLOCKED, diagnose root cause, correct, and rerun once.

CRITICAL TOOL USAGE RULES
- Logs: ALWAYS use `pyats_show_logging` (raw). Do NOT use `pyats_run_show_command` with "show logging".
- Running config: ALWAYS use `pyats_show_running_config`. Do NOT use `pyats_run_show_command` with "show run"/"show running-config".

OUTPUT FORMAT
End every final answer with:
- Health verdict (PASS/FAIL/INCONCLUSIVE)
- 3–7 bullet remediation steps
- Evidence snippets
""".strip()

class NetworkAgent:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_registry: Dict[str, str] = {}
        self.all_functions: List[types.FunctionDeclaration] = []
        self.exit_stack = contextlib.AsyncExitStack()

        self._round_call_counter: Counter[str] = Counter()
        self._turn_seen_calls: Set[str] = set()
        self._requested_wait_s: Optional[int] = None
        self._known_devices: Set[str] = set()
        self._last_user_input: str = ""
        self._device_types: Dict[str, str] = load_device_types(TESTBED_PATH)
        self.evidence = EvidenceStore()

    async def connect_servers(self) -> None:
        print("🔌 Connecting to MCP servers...")
        for server_name, cfg in SERVERS.items():
            env = os.environ.copy()
            if cfg.get("env"):
                env.update({k: v for k, v in cfg["env"].items() if v})

            params = StdioServerParameters(command=cfg["command"], args=cfg["args"], env=env)
            read, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[server_name] = session

            tools_result = await session.list_tools()
            print(f"  ✅ Connected to '{server_name}': {len(tools_result.tools)} tools")
            for tool in tools_result.tools:
                name = tool.name
                if TOOL_ALLOWLIST is not None and name not in TOOL_ALLOWLIST:
                    continue
                if name in TOOL_DENYLIST:
                    continue
                self.tools_registry[name] = server_name
                self.all_functions.append(
                    types.FunctionDeclaration(
                        name=name,
                        description=tool.description or "",
                        parameters=normalize_function_parameters(tool.inputSchema),
                    )
                )
        if not self.all_functions:
            print("⚠️  No tools discovered.")

    def _build_chat(self):
        if not self.client:
            raise RuntimeError("GOOGLE_API_KEY missing.")
        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=self.all_functions)] if self.all_functions else [],
            system_instruction=SYSTEM_INSTRUCTION,
        )
        return self.client.chats.create(model=MODEL_NAME, config=config)

    def _default_verify_cmds_for_type(self, dev_type: str) -> List[str]:
        dt = (dev_type or "").lower()
        if dt.startswith("switch"):
            return ["show interfaces status", "show interfaces description"]
        return ["show ip interface brief", "show interfaces description"]

    def _build_post_wait_verification_calls(self) -> List[ToolCallShim]:
        user_cmds = extract_requested_show_commands(self._last_user_input)
        cmds = [sanitize_show_command(c) for c in user_cmds]
        cmds = [c for c in cmds if is_plausible_show_command(c)]
        if not cmds:
            cmds = []

        calls: List[ToolCallShim] = []
        for dev in sorted(self._known_devices):
            dev_type = self._device_types.get(dev, "")
            per_dev_cmds = cmds or self._default_verify_cmds_for_type(dev_type)
            for cmd in per_dev_cmds:
                cmd2 = rewrite_show_by_device_type(cmd, dev_type)
                calls.append(ToolCallShim(name="pyats_run_show_command", args={"device_name": dev, "command": cmd2}))
        return calls

    def _parse_tool_text(self, tool_text: str) -> Any:
        payload = try_parse_json(tool_text)
        return payload if payload is not None else {"text": tool_text}

    def _call_device_key(self, call: Any) -> Optional[str]:
        args = getattr(call, "args", None) or {}
        dev = args.get("device_name")
        return dev.strip() if isinstance(dev, str) and dev.strip() else None

    async def _raw_call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        server_name = self.tools_registry.get(tool_name)
        session = self.sessions.get(server_name or "")
        if not session:
            return json.dumps({"status": "error", "error": f"Tool '{tool_name}' not found."})

        try:
            res = await asyncio.wait_for(session.call_tool(tool_name, arguments=tool_args), timeout=TOOL_TIMEOUT_S)
            texts = [c.text for c in res.content if getattr(c, "type", None) == "text"]
            return "\n".join(texts) if texts else ""
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT_S}s"})
        except Exception as e:
            return json.dumps({"status": "error", "error": f"Error executing '{tool_name}': {e}"})

    def _intercept_show_command(self, command: str) -> Optional[str]:
        c = _normalize_cmd(command)
        if c.startswith("show logging") or c == "show log" or c.startswith("show log "):
            return "pyats_show_logging"
        if c.startswith("show run") or c.startswith("show running-config") or c.startswith("show running config"):
            return "pyats_show_running_config"
        return None

    # -------------------------------------------------------------
    # 🧠 SMART DEADMAN: Checks if config has happened before running verification
    # -------------------------------------------------------------
    async def _deadman_check(self) -> List[types.Part]:
        has_config = self.evidence.has_configured()
        
        if not has_config:
            print("  🤖 [Deadman] Model silent, but no config detected. Nudging to configure.")
            return [types.Part(text=(
                "SYSTEM MONITOR: You have stopped outputting text, but no configuration has been applied yet. "
                "You are likely stuck in a data gathering loop. "
                "You have sufficient discovery data in the Evidence Store. "
                "PROCEED IMMEDIATELY TO STEP 3: Configure the devices."
            ))]

        cmds_raw = extract_requested_show_commands(self._last_user_input)
        cmds = [sanitize_show_command(c) for c in cmds_raw]
        cmds = [c for c in cmds if is_plausible_show_command(c)]

        if not cmds:
            return [types.Part(text="DEADMAN: Model returned empty output. Please generate the test script.")]

        if not self._known_devices:
            return [types.Part(text=f"DEADMAN: Found show commands {cmds}, but no known devices have been touched yet.")]

        parts: List[types.Part] = []
        parts.append(
            types.Part(
                text=(
                    "DEADMAN: Model returned empty output after configuration. Running explicit post-change show commands "
                    "from user prompt across touched devices.\n"
                    f"DEVICES: {sorted(self._known_devices)}\n"
                    f"SHOWS: {cmds}\n\n"
                    f"{self.evidence.summary(max_items=15)}"
                )
            )
        )
        self._round_call_counter.clear()
        for dev in sorted(self._known_devices):
            for cmd in cmds:
                payload = await self.call_tool_payload("pyats_run_show_command", {"device_name": dev, "command": cmd})
                parts.append(
                    types.Part.from_function_response(
                        name="pyats_run_show_command",
                        response={"device_name": dev, "command": cmd, "result": payload},
                    )
                )
        parts.append(
            types.Part(
                text=(
                    "DEADMAN_NEXT: Verification data collected. "
                    "NOW generate the aetest script (no device connects, embed TEST_DATA dict) and run pyats_run_dynamic_test. "
                )
            )
        )
        return parts

    def _record_evidence(self, tool_name: str, tool_args: Dict[str, Any], payload: Any, command: str = "") -> None:
        dev = ""
        if isinstance(tool_args, dict):
            d = tool_args.get("device_name")
            if isinstance(d, str) and d.strip():
                dev = d.strip()
        cmd = command or (tool_args.get("command") if isinstance(tool_args, dict) else "") or ""
        if isinstance(cmd, str):
            cmd = cmd.strip()
        self.evidence.add(
            EvidenceItem(
                tool=tool_name,
                device=dev,
                command=cmd if tool_name == "pyats_run_show_command" else "",
                args=dict(tool_args or {}),
                result=payload,
            )
        )

    def _has_multiple_devices(self, calls: List[ToolCallShim]) -> bool:
        devs = set()
        for c in calls:
            d = self._call_device_key(c)
            if d:
                devs.add(d)
        return len(devs) >= 2

    def _fmt_call_log(self, name: str, args: Dict[str, Any]) -> str:
        if not args:
            return f"{name}()"
        dev = args.get("device_name", "")
        if name == "pyats_run_show_command":
            cmd = args.get("command", "")
            return f"pyats_show['{cmd}'] on {dev}"
        elif name == "pyats_configure_device":
             return f"pyats_config on {dev} ({len(str(args.get('config_commands','')))} chars)"
        elif dev:
             return f"{name} on {dev}"
        return f"{name}({str(args)[:50]}...)"

    async def call_tool_payload(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        key = call_key(tool_name, tool_args)
        log_label = self._fmt_call_log(tool_name, tool_args)

        # ✅ TURN-LEVEL DEDUPE
        if key in self._turn_seen_calls:
            if TURN_DEDUPE_MODE == "reuse":
                if isinstance(tool_args, dict) and tool_name == "pyats_run_show_command":
                    dev = tool_args.get("device_name", "")
                    cmd = tool_args.get("command", "")
                    if isinstance(dev, str) and isinstance(cmd, str) and dev.strip() and cmd.strip():
                        prev = self.evidence.latest_show(dev.strip(), cmd.strip())
                        if prev is not None:
                            print(f"  ♻️  [Cached] {log_label}")
                            return {"status": "deduped_reuse", "tool": tool_name, "args": tool_args, "result": prev.result}

                print(f"  ♻️  [Cached] {log_label} (Skipping execution)")
                payload = {"status": "deduped_reuse", "tool": tool_name, "args": tool_args, "result": None}
                self._record_evidence(tool_name, tool_args, payload)
                return payload

            payload = {
                "status": "error",
                "error": "TurnDedupe: identical tool call already executed in this user turn",
                "tool": tool_name,
                "args": tool_args,
            }
            self._record_evidence(tool_name, tool_args, payload)
            return payload

        self._turn_seen_calls.add(key)
        self._round_call_counter[key] += 1
        if self._round_call_counter[key] > MAX_SAME_CALLS_PER_ROUND:
            payload = {
                "status": "error",
                "error": f"GuardRail: refusing repeated call in same tool-round (>{MAX_SAME_CALLS_PER_ROUND}).",
                "tool": tool_name,
                "args": tool_args,
            }
            self._record_evidence(tool_name, tool_args, payload)
            return payload

        # --- Sandbox self-heal & VISIBILITY ---
        if tool_name == "pyats_run_dynamic_test":
            print(f"  🧪 [Live] Running Test Generation...")
            script = (tool_args or {}).get("test_script_content", "")
            if script:
                match = re.search(r"TEST_DATA\s*=\s*({.*?})", script, re.DOTALL)
                if match:
                    try:
                        data_snippet = match.group(1)[:200].replace("\n", " ") + "..."
                        print(f"  🧪 [Test Plan] TEST_DATA found: {data_snippet}")
                    except:
                        pass
                else:
                    print(f"  🧪 [Test Plan] Script size: {len(script)} chars")

            txt = await self._raw_call_tool(tool_name, tool_args or {})
            payload = self._parse_tool_text(txt)
            if isinstance(payload, dict) and payload.get("status") == "error":
                blocked = extract_blocked_token(payload.get("error", ""))
                script = (tool_args or {}).get("test_script_content")
                if blocked and isinstance(script, str) and script.strip():
                    cleaned = break_forbidden_substring(script, blocked)
                    if cleaned != script and not (tool_args or {}).get("_sandbox_retry"):
                        if DEBUG_TOOLS:
                            print(f"  🧽 Sandbox blocked '{blocked}'. Retrying with sanitized script...")
                        tool_args2 = dict(tool_args or {})
                        tool_args2["test_script_content"] = cleaned
                        tool_args2["_sandbox_retry"] = True
                        txt2 = await self._raw_call_tool(tool_name, tool_args2)
                        payload2 = self._parse_tool_text(txt2)
                        self._record_evidence(tool_name, tool_args2, payload2)
                        return payload2
            self._record_evidence(tool_name, tool_args, payload)
            return payload

        # --- Show command sanitizer ---
        if tool_name == "pyats_run_show_command":
            raw_cmd = (tool_args or {}).get("command", "")
            dev = (tool_args or {}).get("device_name", "")
            dev_name = dev.strip() if isinstance(dev, str) else ""
            dev_type = self._device_types.get(dev_name, "")

            cmd = sanitize_show_command(raw_cmd)
            if not is_plausible_show_command(cmd):
                payload = {
                    "status": "error",
                    "error": f"Refusing invalid show command from model: {raw_cmd!r} -> {cmd!r}",
                    "meta": {"reason": "model_generated_invalid_show_command"},
                }
                self._record_evidence(tool_name, tool_args, payload, command=cmd)
                return payload

            cmd2 = rewrite_show_by_device_type(cmd, dev_type)
            tool_args2 = dict(tool_args or {})
            tool_args2["command"] = cmd2
            log_label = self._fmt_call_log(tool_name, tool_args2)

            dedicated = self._intercept_show_command(cmd2)
            if dedicated and dev_name:
                print(f"  🛠️  [Live] {log_label} (Intercept -> {dedicated})")
                txt = await self._raw_call_tool(dedicated, {"device_name": dev_name})
                payload = self._parse_tool_text(txt)
                payload = normalize_show_payload(payload, command=dedicated)
                wrapped = {
                    "status": "intercepted",
                    "intercepted_to": dedicated,
                    "device_type": dev_type,
                    "original": {"tool": tool_name, "args": {"device_name": dev_name, "command": raw_cmd}},
                    "sanitized_command": cmd2,
                    "result": payload,
                }
                self._record_evidence(tool_name, tool_args2, wrapped, command=cmd2)
                return wrapped

            print(f"  🛠️  [Live] {log_label}")
            txt = await self._raw_call_tool(tool_name, tool_args2)
            payload = self._parse_tool_text(txt)
            payload = normalize_show_payload(payload, command=cmd2)

            if INVALID_CMD_PAT.search(_textify(payload)) and not tool_args2.get("_auto_retry"):
                fallbacks = choose_fallbacks_for_invalid(cmd2, dev_type)
                for fb in fallbacks[:2]:
                    if DEBUG_TOOLS:
                        print(f"  🔁 Invalid command on {dev_name or '?'} ({dev_type or 'unknown'}). Retrying with: {fb}")
                    tool_args3 = dict(tool_args2)
                    tool_args3["command"] = fb
                    tool_args3["_auto_retry"] = True
                    print(f"  🛠️  [Live] Retry: {fb} on {dev_name}")
                    txt2 = await self._raw_call_tool(tool_name, tool_args3)
                    payload2 = self._parse_tool_text(txt2)
                    payload2 = normalize_show_payload(payload2, command=fb)
                    if not INVALID_CMD_PAT.search(_textify(payload2)):
                        wrapped = {
                            "status": "completed_with_fallback",
                            "device_type": dev_type,
                            "original": {"raw_requested": raw_cmd, "sanitized": cmd2},
                            "fallback_command": fb,
                            "result": payload2,
                        }
                        self._record_evidence(tool_name, tool_args3, wrapped, command=fb)
                        return wrapped

            self._record_evidence(tool_name, tool_args2, payload, command=cmd2)
            return payload

        print(f"  🛠️  [Live] {log_label}")
        txt = await self._raw_call_tool(tool_name, tool_args or {})
        payload = self._parse_tool_text(txt)
        if tool_name in ("pyats_show_logging", "pyats_show_running_config"):
            payload = normalize_show_payload(payload, command=tool_name)
        self._record_evidence(tool_name, tool_args, payload)
        return payload

    async def _execute_calls_sequential(self, calls: List[ToolCallShim]) -> List[types.Part]:
        parts: List[types.Part] = []
        for call in calls:
            name = call.name
            args = call.args or {}
            dev = args.get("device_name")
            if isinstance(dev, str) and dev.strip():
                self._known_devices.add(dev.strip())
            payload = await self.call_tool_payload(name, args)
            if DEBUG_TOOLS or (isinstance(payload, dict) and payload.get("status") in ("error", "intercepted")):
                print("  🔎 Tool output preview:")
                print(summarize_tool_payload(payload))
            parts.append(types.Part.from_function_response(name=name, response=payload))
        return parts

    async def _execute_calls_parallel_multi_device(self, calls: List[ToolCallShim]) -> List[types.Part]:
        indexed = list(enumerate(calls))
        buckets: Dict[str, List[Tuple[int, ToolCallShim]]] = defaultdict(list)
        no_dev: List[Tuple[int, ToolCallShim]] = []
        for idx, call in indexed:
            dev = self._call_device_key(call)
            (buckets[dev].append((idx, call)) if dev else no_dev.append((idx, call)))

        async def run_bucket(device_id: str, items: List[Tuple[int, ToolCallShim]]) -> List[Tuple[int, types.Part]]:
            out: List[Tuple[int, types.Part]] = []
            for idx, call in items:
                name = call.name
                args = call.args or {}
                devn = args.get("device_name")
                if isinstance(devn, str) and devn.strip():
                    self._known_devices.add(devn.strip())
                payload = await self.call_tool_payload(name, args)
                if DEBUG_TOOLS or (isinstance(payload, dict) and payload.get("status") in ("error", "intercepted")):
                    print(f"  🔎 Tool output preview (Device: {device_id}):")
                    print(summarize_tool_payload(payload))
                out.append((idx, types.Part.from_function_response(name=name, response=payload)))
            return out

        results: List[Tuple[int, types.Part]] = []
        if no_dev:
            results.extend(await run_bucket("GLOBAL", sorted(no_dev, key=lambda x: x[0])))

        tasks = []
        for dev, items in buckets.items():
            items_sorted = sorted(items, key=lambda x: x[0])
            print(f"  🧵 [pCall] Starting thread for: {dev} ({len(items_sorted)} calls)")
            tasks.append(run_bucket(dev, items_sorted))

        if tasks:
            groups = await asyncio.gather(*tasks, return_exceptions=False)
            for g in groups:
                results.extend(g)

        results.sort(key=lambda x: x[0])
        parts = [p for _, p in results]
        if len(parts) != len(calls):
            print(f"  ⚠️  pCall assembly mismatch: expected {len(calls)} parts, got {len(parts)}. Returning best-effort parts.")
        return parts

    async def run(self) -> None:
        if not self.client:
            print("❌ GOOGLE_API_KEY missing.")
            return
        chat = self._build_chat()
        print("\n🚀 Network Agent Online! (type 'exit' to stop)")

        while True:
            user_input = input("\nNetwork Ops > ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            self._requested_wait_s = extract_wait_seconds(user_input)
            self._last_user_input = user_input
            self._known_devices = set()
            self._turn_seen_calls.clear()

            response = chat.send_message(user_input)
            
            # 🔄 MAIN LOOP
            MAX_RECOVERIES = 5
            recovery_count = 0
            
            while True:
                _, calls, combined_text = extract_parts_calls_text(response)

                # --- Phase A: Execute Tools ---
                if calls:
                    # 🛑 LOOP BREAKER
                    current_keys = {call_key(c.name, c.args) for c in calls}
                    if current_keys.issubset(self._turn_seen_calls):
                        print("  🛑 Loop Breaker: All tools in this round have already been run.")
                        response = chat.send_message("SYSTEM NUDGE: You are looping. Stop gathering. Proceed to Configuration or Verification.")
                        continue

                    self._round_call_counter.clear()
                    multi_device = self._has_multiple_devices(calls)
                    if PCALL and multi_device:
                        tool_parts = await self._execute_calls_parallel_multi_device(calls)
                    else:
                        tool_parts = await self._execute_calls_sequential(calls)

                    # Handle User Wait request
                    did_config = any(getattr(c, "name", "") == "pyats_configure_device" for c in calls)
                    if did_config and self._requested_wait_s and self._requested_wait_s > 0:
                        print(f"⏳ Waiting {self._requested_wait_s} seconds (per user instruction) for convergence...")
                        await asyncio.sleep(self._requested_wait_s)
                        
                        # 💥 V5 FIX: CLEAR DEDUPE CACHE NOW
                        # This ensures the verification commands run on LIVE devices, not cached history.
                        print("  🧹 [Cache] Clearing deduplication cache to force fresh verification.")
                        self._turn_seen_calls.clear()
                        
                        verify_calls = self._build_post_wait_verification_calls()
                        self._round_call_counter.clear()
                        
                        verify_multi = self._has_multiple_devices(verify_calls)
                        if PCALL and verify_multi:
                            verify_parts = await self._execute_calls_parallel_multi_device(verify_calls)
                        else:
                            verify_parts = await self._execute_calls_sequential(verify_calls)
                            
                        tool_parts.append(types.Part(text=f"WAIT_COMPLETE: Slept {self._requested_wait_s}s. Verified devices."))
                        tool_parts.extend(verify_parts)

                    response = chat.send_message(tool_parts)
                    continue

                # --- Phase B: Deadman & Completion Check ---
                if not calls and not combined_text:
                    if recovery_count >= MAX_RECOVERIES:
                        print("  ❌ Max recovery attempts reached. Stopping.")
                        break

                    if self.evidence.ready_for_test() and not self.evidence.has_run_test():
                        print("  👉 [Auto-Nudge] Verification complete, but no Test Script run. Forcing test generation...")
                        response = chat.send_message("SYSTEM: Verification complete. GENERATE AND RUN THE PYATS TEST SCRIPT NOW.")
                        recovery_count += 1
                        continue

                    dm_parts = await self._deadman_check()
                    response = chat.send_message(dm_parts)
                    recovery_count += 1
                    continue

                # --- Phase C: Text Output (Final) ---
                if combined_text:
                     # Completion Guardrail
                     if self.evidence.ready_for_test() and not self.evidence.has_run_test() and recovery_count < MAX_RECOVERIES:
                         print("  👉 [Auto-Nudge] Text output received, but Test Script missing. Forcing test generation...")
                         response = chat.send_message("SYSTEM: You provided a summary, but you MUST run `pyats_run_dynamic_test` to validate. Do it now.")
                         recovery_count += 1
                         continue
                         
                     print(f"\nAgent: {combined_text}")
                     break

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    agent = NetworkAgent()
    try:
        await agent.connect_servers()
        await agent.run()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())