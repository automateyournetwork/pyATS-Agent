import os
import sys
import asyncio
import contextlib
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# --- CONFIGURATION ---
# We now pull the testbed path from an environment variable as requested
TESTBED_PATH = os.getenv("TESTBED_PATH")

SERVERS = {
    "pyats": {
        "command": sys.executable,
        # Update this path to your specific pyats_mcp_server.py location
        "args": ["/home/johncapobianco/Agent_with_GAIT/pyATS_MCP/pyats_mcp_server.py"],
        "env": {"PYATS_TESTBED_PATH": TESTBED_PATH} 
    }
}

class NetworkAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        
        self.sessions = {}
        self.tools_registry = {}  # tool_name -> server_name
        self.all_functions = []   # List of types.FunctionDeclaration
        self.exit_stack = contextlib.AsyncExitStack()

    async def connect_servers(self):
        if not TESTBED_PATH:
            print("⚠️  Warning: TESTBED_PATH environment variable is not set.")
            
        print("🔌 Connecting to pyATS MCP server...")
        
        for name, config in SERVERS.items():
            try:
                env = os.environ.copy()
                if config.get("env"):
                    env.update({k: v for k, v in config["env"].items() if v})

                server_params = StdioServerParameters(
                    command=config["command"],
                    args=config["args"],
                    env=env
                )
                
                read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                
                await session.initialize()
                self.sessions[name] = session
                
                tools_result = await session.list_tools()
                print(f"  ✅ Connected to '{name}': Found {len(tools_result.tools)} tools")
                
                for tool in tools_result.tools:
                    self.tools_registry[tool.name] = name
                    self.all_functions.append(
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.inputSchema
                        )
                    )
                    
            except Exception as e:
                print(f"  ❌ Failed to connect to '{name}': {e}")

    async def call_tool(self, tool_name, tool_args):
        server_name = self.tools_registry.get(tool_name)
        session = self.sessions.get(server_name)
        if not session:
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            result = await session.call_tool(tool_name, arguments=tool_args)
            output_text = [content.text for content in result.content if content.type == "text"]
            return "\n".join(output_text) if output_text else "Success (no text output)"
        except Exception as e:
            return f"Error executing '{tool_name}': {e}"

    async def run(self):
        if not self.client:
            print("❌ GOOGLE_API_KEY missing.")
            return

        # Prepare Gemini tool config
        tool_config = None
        if self.all_functions:
            tool_config = types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=self.all_functions)],
                system_instruction="""You are an expert Network Automation Engineer. 
                Use pyATS tools to inspect, troubleshoot, and manage network devices. 
                Always provide concise, technical summaries of command outputs."""
            )

        chat = self.client.chats.create(model="gemini-2.0-flash-exp", config=tool_config)

        print("\n🚀 Network Agent Online! (type 'exit' to stop)")
        while True:
            user_input = input("\nNetwork Ops > ")
            if user_input.lower() in ["quit", "exit"]: break
            
            try:
                response = chat.send_message(user_input)
                
                # Tool loop
                while response.function_calls:
                    for call in response.function_calls:
                        print(f"  🛠️  Executing: {call.name}...")
                        result = await self.call_tool(call.name, call.args)
                        
                        response = chat.send_message(
                            types.Part.from_function_response(
                                name=call.name, 
                                response={"result": result}
                            )
                        )
                
                if response.text:
                    print(f"\nAgent: {response.text}")
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    agent = NetworkAgent()
    try:
        await agent.connect_servers()
        await agent.run()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())