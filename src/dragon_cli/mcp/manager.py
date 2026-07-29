"""MCP (Model Context Protocol) stdio server connections and tool-calling."""

import json

from ..config import C_DIM, C_GREEN, C_RED, C_RESET, C_YELLOW, EXAMPLE_MCP_CONFIG, MCP_CONFIG_FILE


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
                    self.tools[f"{name}.{t.name}"] = {
                        "server": name, "tool": t.name,
                        "description": t.description or "",
                        "schema": t.inputSchema,
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
