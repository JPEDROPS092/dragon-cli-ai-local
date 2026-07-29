"""Slash-command dispatch for the chat loop: help, model info, and
conversation management (list/new/resume/rename/delete/search/export)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..config import C_DIM, C_GREEN, C_RED, C_RESET, MCP_CONFIG_FILE
from ..models import list_catalog

if TYPE_CHECKING:
    from .session import ChatSession

HELP_TEXT = """Commands:
  /help              show this help
  /models            list the model catalog
  /tools             list connected MCP tools
  /mcp               show MCP server connection status
  /clear             clear the current in-memory history
  /new [title]       start a new conversation (saves and switches)
  /list              list saved conversations
  /resume <id>       resume a saved conversation
  /rename <id> <title>  rename a saved conversation
  /delete <id>       delete a saved conversation
  /search <text>     search saved messages for text
  /export <id> [md|json]  export a conversation to a file
  /exit, /quit       end the program"""


def cmd_help(session: "ChatSession", arg: str) -> None:
    print(HELP_TEXT)


def cmd_models(session: "ChatSession", arg: str) -> None:
    list_catalog()


def cmd_tools(session: "ChatSession", arg: str) -> None:
    if not session.mcp.tools:
        print("No MCP tools connected.")
        return
    for name, info in session.mcp.tools.items():
        print(f"  - {name}: {info['description'][:90]}")


def cmd_mcp(session: "ChatSession", arg: str) -> None:
    servers = list(session.mcp.sessions.keys()) or "none"
    print(f"Connected servers: {servers}")
    print(f"Config: {MCP_CONFIG_FILE.resolve()}")


def cmd_clear(session: "ChatSession", arg: str) -> None:
    session.clear()
    print("History cleared.")


def cmd_new(session: "ChatSession", arg: str) -> None:
    session.start_new(arg.strip() or None)
    print(f"{C_GREEN}[ok]{C_RESET} Started conversation #{session.conv_id}.")


def cmd_list(session: "ChatSession", arg: str) -> None:
    conversations = session.store.list()
    if not conversations:
        print("No saved conversations yet.")
        return
    for c in conversations:
        marker = " *" if c.id == session.conv_id else ""
        print(f"  #{c.id}{marker}  {c.title}  ({c.message_count} msgs, updated {c.updated_at})")


def cmd_resume(session: "ChatSession", arg: str) -> None:
    try:
        conv_id = int(arg.strip())
    except ValueError:
        print("Usage: /resume <id>")
        return
    if session.resume(conv_id):
        print(f"{C_GREEN}[ok]{C_RESET} Resumed conversation #{conv_id}.")
    else:
        print(f"{C_RED}[error]{C_RESET} No conversation with id {conv_id}.")


def cmd_rename(session: "ChatSession", arg: str) -> None:
    parts = arg.strip().split(maxsplit=1)
    if len(parts) != 2 or not parts[0].isdigit():
        print("Usage: /rename <id> <new title>")
        return
    conv_id, title = int(parts[0]), parts[1]
    if session.store.rename(conv_id, title):
        print(f"{C_GREEN}[ok]{C_RESET} Renamed #{conv_id} to '{title}'.")
    else:
        print(f"{C_RED}[error]{C_RESET} No conversation with id {conv_id}.")


def cmd_delete(session: "ChatSession", arg: str) -> None:
    if not arg.strip().isdigit():
        print("Usage: /delete <id>")
        return
    conv_id = int(arg.strip())
    confirm = input(f"Delete conversation #{conv_id}? This cannot be undone. (y/N) ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    if session.store.delete(conv_id):
        print(f"{C_GREEN}[ok]{C_RESET} Deleted #{conv_id}.")
        if conv_id == session.conv_id:
            session.start_new()
    else:
        print(f"{C_RED}[error]{C_RESET} No conversation with id {conv_id}.")


def cmd_search(session: "ChatSession", arg: str) -> None:
    text = arg.strip()
    if not text:
        print("Usage: /search <text>")
        return
    results = session.store.search(text)
    if not results:
        print("No matches.")
        return
    for conv_id, title, role, content in results[:20]:
        snippet = content.replace("\n", " ")[:80]
        print(f"  #{conv_id} {title} [{role}]: {snippet}")


def cmd_export(session: "ChatSession", arg: str) -> None:
    parts = arg.strip().split()
    if not parts or not parts[0].isdigit():
        print("Usage: /export <id> [md|json]")
        return
    conv_id = int(parts[0])
    fmt = parts[1] if len(parts) > 1 else "md"
    if fmt not in ("md", "json"):
        print("Format must be 'md' or 'json'.")
        return
    path = Path(f"conversation_{conv_id}.{fmt}")
    result = session.store.export(conv_id, path, fmt)
    if result is None:
        print(f"{C_RED}[error]{C_RESET} No conversation with id {conv_id}.")
    else:
        print(f"{C_GREEN}[ok]{C_RESET} Exported to {result.resolve()}")


def cmd_exit(session: "ChatSession", arg: str) -> bool:
    return False


COMMANDS: dict[str, Callable[["ChatSession", str], bool | None]] = {
    "help": cmd_help,
    "models": cmd_models,
    "tools": cmd_tools,
    "mcp": cmd_mcp,
    "clear": cmd_clear,
    "new": cmd_new,
    "list": cmd_list,
    "resume": cmd_resume,
    "rename": cmd_rename,
    "delete": cmd_delete,
    "search": cmd_search,
    "export": cmd_export,
    "exit": cmd_exit,
    "quit": cmd_exit,
}


def dispatch(session: "ChatSession", raw_input: str) -> bool | None:
    """Runs a slash command. Returns False if the chat loop should stop."""
    name, _, arg = raw_input[1:].partition(" ")
    handler = COMMANDS.get(name.lower())
    if handler is None:
        print(f"{C_DIM}Unknown command '{raw_input.split()[0]}'. Try /help.{C_RESET}")
        return True
    return handler(session, arg)
