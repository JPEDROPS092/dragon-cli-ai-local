"""MCP (Model Context Protocol) stdio server connections and tool-calling."""

import json

from ..config import C_DIM, C_GREEN, C_RED, C_RESET, C_YELLOW, EXAMPLE_MCP_CONFIG, MCP_CONFIG_FILE

# Ready-to-use MCP server presets. ``args`` may contain ``{dir}``` placeholders
# filled in at add-time. Each is a commonly used, publicly available connector.
MCP_PRESETS = {
    "filesystem": {
        "description": "Read/write files in a folder you choose",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    },
    "fetch": {
        "description": "Fetch and summarize web pages",
        "command": "uvx",
        "args": ["--with", "mcp<1.0", "mcp-server-fetch"],
    },
    "git": {
        "description": "Inspect and operate on local git repos",
        "command": "uvx",
        "args": ["mcp-server-git", "--repository", "."],
    },
    "memory": {
        "description": "Persistent knowledge graph memory",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
    },
    "sqlite": {
        "description": "Query a local SQLite database file",
        "command": "uvx",
        "args": ["mcp-server-sqlite", "--db-path", "memory.db"],
    },
    "time": {
        "description": "Current time and timezone conversion",
        "command": "uvx",
        "args": ["mcp-server-time"],
    },
}


def load_mcp_config() -> dict:
    """Read mcp_servers.json, creating the example file if absent."""
    if not MCP_CONFIG_FILE.exists():
        MCP_CONFIG_FILE.write_text(json.dumps(EXAMPLE_MCP_CONFIG, indent=2), encoding="utf-8")
    try:
        return json.loads(MCP_CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"mcpServers": {}}


def save_mcp_config(servers: dict) -> None:
    """Write the mcpServers dict back to mcp_servers.json."""
    MCP_CONFIG_FILE.write_text(
        json.dumps({"mcpServers": servers}, indent=2), encoding="utf-8"
    )


class MCPManager:
    """Connects to MCP (stdio) servers defined in mcp_servers.json."""

    def __init__(self):
        self.sessions = {}   # server_name -> ClientSession
        self.tools = {}      # "server.tool" -> {description, schema, server, tool}
        self._stack = None

    async def start(self):
        if not MCP_CONFIG_FILE.exists():
            MCP_CONFIG_FILE.write_text(json.dumps(EXAMPLE_MCP_CONFIG, indent=2), encoding="utf-8")
            print(f"{C_YELLOW}[mcp]{C_RESET} Created an example {MCP_CONFIG_FILE}. "
                  f"Edit it and restart to enable connectors.")
            return
        try:
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            print(f"{C_RED}[mcp]{C_RESET} Package 'mcp' is not installed. Run: pip install mcp")
            return

        config = json.loads(MCP_CONFIG_FILE.read_text(encoding="utf-8"))
        servers = config.get("mcpServers", {})
        if not servers:
            print(f"{C_DIM}[mcp] No servers configured in {MCP_CONFIG_FILE}.{C_RESET}")
            return

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, cfg in servers.items():
            try:
                params = StdioServerParameters(
                    command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"),
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.list_tools()
                self.sessions[name] = session
                for t in result.tools:
                    # mcp SDK >=1.0 renamed inputSchema -> input_schema.
                    schema = getattr(t, "input_schema", None)
                    if schema is None:
                        schema = getattr(t, "inputSchema", {}) or {}
                    self.tools[f"{name}.{t.name}"] = {
                        "server": name, "tool": t.name,
                        "description": t.description or "",
                        "schema": schema,
                    }
                print(f"{C_GREEN}[mcp]{C_RESET} '{name}' connected ({len(result.tools)} tools)")
            except Exception as e:
                print(f"{C_RED}[mcp]{C_RESET} Failed to connect '{name}': {e}")

    async def call(self, tool_name: str, arguments: dict) -> str:
        info = self.tools.get(tool_name)
        if not info:
            return f"Error: tool '{tool_name}' does not exist."
        session = self.sessions[info["server"]]
        try:
            result = await session.call_tool(info["tool"], arguments or {})
            parts = []
            for c in result.content:
                parts.append(getattr(c, "text", str(c)))
            return "\n".join(parts)[:8000]  # keep the context from overflowing
        except Exception as e:
            return f"Error running the tool: {e}"

    async def stop(self):
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass

    def tools_prompt(self) -> str:
        if not self.tools:
            return ""
        lines = ["\nYou have access to the following tools (MCP):"]
        for name, info in self.tools.items():
            schema = json.dumps(info["schema"].get("properties", {}), ensure_ascii=False)
            lines.append(f"- {name}: {info['description']} | parameters: {schema}")
        lines.append(
            "\nTo use a tool, respond ONLY with a block in this format:\n"
            '```tool_call\n{"tool": "server.tool_name", "arguments": {...}}\n```\n'
            "After receiving the result, respond to the user normally. "
            "If you don't need a tool, respond directly."
        )
        return "\n".join(lines)
