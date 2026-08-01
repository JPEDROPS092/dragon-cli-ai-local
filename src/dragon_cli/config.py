"""Paths, colors, and other shared constants for Dragon CLI."""

from pathlib import Path

MODELS_DIR = Path("models")
MCP_CONFIG_FILE = Path("mcp_servers.json")
DB_FILE = Path("dragon_cli.db")

# Simple ANSI colors (work in modern Windows Terminal / PowerShell).
C_RESET = "\033[0m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_DIM = "\033[2m"

EXAMPLE_MCP_CONFIG = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        },
        "fetch": {
            "command": "uvx",
            "args": ["--with", "mcp<1.0", "mcp-server-fetch"],
        },
    }
}
