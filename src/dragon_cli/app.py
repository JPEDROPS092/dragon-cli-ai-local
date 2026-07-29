"""Async orchestration: model selection, MCP startup, and the chat session."""

from pathlib import Path

from .chat import ChatSession, chat_loop
from .config import C_RED, C_RESET
from .mcp import MCPManager
from .models import MODEL_CATALOG, available_ram_gb, download_model, list_catalog, load_llm
from .storage import ConversationStore


def choose_model(args) -> dict | None:
    list_catalog()
    if args.model:
        choice = args.model
    else:
        try:
            choice = int(input("\nChoose a model number: ").strip())
        except (ValueError, EOFError, KeyboardInterrupt):
            print("Invalid choice.")
            return None
    if not (1 <= choice <= len(MODEL_CATALOG)):
        print("Number not in the catalog.")
        return None
    model = MODEL_CATALOG[choice - 1]
    ram = available_ram_gb()
    if 0 < ram < model["ram_min_gb"]:
        ok = input(f"{C_RED}[warning]{C_RESET} Model needs ~{model['ram_min_gb']} GB, "
                   f"you have {ram:.1f} GB free. Continue? (y/N) ").strip().lower()
        if ok != "y":
            return None
    return model


def print_conversations(store: ConversationStore) -> None:
    conversations = store.list()
    if not conversations:
        print("No conversations yet.")
        return
    for c in conversations:
        print(f"  #{c.id}  {c.title}  ({c.message_count} msgs, updated {c.updated_at})")


async def run(args) -> None:
    if args.serve:
        model = choose_model(args)
        if not model:
            return
        path = download_model(model)
        llm = load_llm(path, model["ctx"])
        from .server import run_server
        await run_server(llm, model["name"], args.host, args.port)
        return

    store = ConversationStore(Path(args.db))

    if args.list_conversations:
        print_conversations(store)
        store.close()
        return

    model = choose_model(args)
    if not model:
        store.close()
        return

    path = download_model(model)
    llm = load_llm(path, model["ctx"])

    mcp = MCPManager()
    if not args.no_mcp:
        await mcp.start()

    session = ChatSession(llm, mcp, store, model["name"])
    if args.resume is not None:
        if not session.resume(args.resume):
            print(f"{C_RED}[error]{C_RESET} No conversation with id {args.resume}, starting a new one.")
            session.start_new()
    else:
        session.start_new()

    try:
        await chat_loop(session)
    finally:
        await mcp.stop()
        store.close()
