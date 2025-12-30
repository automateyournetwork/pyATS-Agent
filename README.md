# 🌐 Gemini pyATS Network Agent

A standalone AI Network Assistant that leverages **Google Gemini 2.0** and the **Model Context Protocol (MCP)** to interact with network infrastructure using **Cisco pyATS**.

This agent allows you to use natural language to inspect, troubleshoot, and manage network devices via a specialized pyATS MCP server. It transforms complex network data into actionable insights through a conversational interface.

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.10+**
* A Google AI Studio **API Key** ([Get it here](https://aistudio.google.com/))
* A pyATS **Testbed file** (e.g., `testbed.yaml`) containing your device credentials and IP addresses.

### 2. Installation

Clone the repository and set up a virtual environment to manage dependencies:

```bash
# Clone the repo
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Configuration

The agent requires two main configuration points to function:

1. **API Key:** Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

2. **Testbed Path:** Export the path to your pyATS testbed file in your current terminal session:
   ```bash
   # Linux / macOS
   export TESTBED_PATH="/absolute/path/to/your/testbed.yaml"

   # Windows (PowerShell)
   $env:TESTBED_PATH="C:\path\to\testbed.yaml"
   ```

### 4. Running the Agent

Start the interactive session:

```bash
python agent.py
```

---

## 🛠️ Architecture

The agent acts as an intelligent bridge between the LLM (Gemini) and your physical or virtual network infrastructure.



### How it works:
1. **User Query:** You ask, *"Is OSPF running on the core-router?"*
2. **Gemini 2.0:** Processes the intent and identifies the correct pyATS tool to use.
3. **MCP Client:** Sends the execution request to the local pyATS MCP Server via STDIO.
4. **pyATS/Genie:** Connects to the device, runs the `show ip ospf` command, parses the raw text into a structured JSON object, and returns it.
5. **Gemini 2.0:** Analyzes the structured data and provides a concise, human-friendly answer.

---

## 💡 Example Queries

You can interact with your network using natural language:

* **Health Checks:** *"Perform a health check on all devices and tell me if any interfaces are down."*
* **Routing:** *"Show me the routing table for leaf-01 and highlight any BGP routes."*
* **Connectivity:** *"Ping 8.8.8.8 from the border-gateway and verify internet reachability."*
* **Details:** *"What version of software is running on the distribution switches?"*

---

## 🔒 Security & Best Practices

* **Environment Variables:** Never commit your `.env` file or your `testbed.yaml` to GitHub. They contain sensitive API keys and network credentials.
* **Access Control:** It is recommended to use a pyATS user with **read-only** privileges for general troubleshooting to ensure the agent cannot accidentally modify configurations.
* **Logs:** Monitor the terminal output to see exactly which commands the agent is choosing to run on your equipment.

---

## 📦 Project Dependencies (requirements.txt)

To ensure the agent runs correctly, the following packages are required in your `requirements.txt` file:

```text
google-genai
mcp
pydantic
python-dotenv
pyats[full]
asyncio
```

---