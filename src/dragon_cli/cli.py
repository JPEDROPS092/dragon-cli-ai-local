#!/usr/bin/env python3
"""Dragon CLI — chat with local LLMs (GGUF) + MCP connectors.

Features:
  * ASCII dragon banner at startup
  * Downloads and stores GGUF models locally (./models)
  * Interactive streaming chat, 100% on CPU
  * MCP (Model Context Protocol) connectors via mcp_servers.json
  * Tool-calling: the model can call tools exposed by MCP servers
  * SQLite-backed conversation history: list/new/resume/rename/delete/search/export
  * --serve: OpenAI-compatible HTTP endpoint for the loaded model
  * --profile: Show hardware profile
  * --recommend: Show model recommendations
  * --auto: Automatically select best model for your hardware

Install (in your venv):
    pip install -e .
    dragon-cli-install-llm   (detects your OS/GPU, installs optimal llama-cpp-python wheel)

Usage:
    dragon-cli                       # model menu + chat
    dragon-cli --model 2             # skip the model menu
    dragon-cli --list                # just list models
    dragon-cli --no-mcp              # disable MCP
    dragon-cli --resume 3            # resume saved conversation #3
    dragon-cli --list-conversations  # list saved conversations and exit
    dragon-cli --serve --port 8000   # serve the model over HTTP instead of chatting
    dragon-cli --profile             # show hardware profile
    dragon-cli --recommend           # show model recommendations
    dragon-cli --auto                # auto-select best model

In-chat commands:
    /help    /models   /tools    /mcp     /clear
    /new     /list     /resume   /rename  /delete
    /search  /export   /exit
"""

import argparse
import asyncio
import os
import sys

from . import app
from .banner import BANNER, SUBTITLE
from .config import C_CYAN, C_RESET, C_YELLOW, DB_FILE
from .models import list_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dragon CLI — local LLM + MCP")
    parser.add_argument("--list", action="store_true", help="list the model catalog and exit")
    parser.add_argument("--model", type=int, help="model number from the catalog")
    parser.add_argument("--no-mcp", action="store_true", help="don't connect to MCP servers")
    parser.add_argument("--db", default=str(DB_FILE), help="path to the SQLite conversation database")
    parser.add_argument("--resume", type=int, default=None, help="resume a saved conversation by id")
    parser.add_argument("--list-conversations", action="store_true", help="list saved conversations and exit")
    parser.add_argument("--serve", action="store_true",
                         help="serve the model over an OpenAI-compatible HTTP endpoint instead of chatting")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                         help="bind address for --serve (default: 127.0.0.1, localhost only)")
    parser.add_argument("--profile", action="store_true",
                         help="show hardware profile and exit")
    parser.add_argument("--recommend", action="store_true",
                         help="show model recommendations for your hardware and exit")
    parser.add_argument("--priority", choices=["speed", "quality", "balanced", "memory"],
                         default="balanced", help="optimization priority for --recommend or --auto")
    parser.add_argument("--auto", action="store_true",
                         help="automatically select the best model for your hardware")
    return parser


def main() -> None:
    if os.name == "nt":
        os.system("")  # enable ANSI colors in the Windows console

    args = build_parser().parse_args()

    print(C_CYAN + BANNER + C_RESET)
    print(C_YELLOW + SUBTITLE + C_RESET + "\n")

    # Handle new flags before banner display
    if args.profile:
        from .hardware import print_hardware_profile
        print_hardware_profile()
        return

    if args.recommend:
        from .recommendation import recommend_model_for_hardware
        recommend_model_for_hardware(priority=args.priority, show_all=True)
        return

    if args.list:
        list_catalog()
        return

    required = [("llama_cpp", "llama-cpp-python"), ("huggingface_hub", "huggingface_hub")]
    if args.serve:
        required += [("fastapi", "fastapi"), ("starlette", "starlette"), ("uvicorn", "uvicorn")]
    missing = []
    for module, pip_name in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    if missing and not args.list_conversations:
        if "llama-cpp-python" in missing:
            print("Missing dependency: llama-cpp-python.")
            print("Run: dragon-cli-install-llm   (detects your OS/GPU and installs the optimal wheel)")
            missing.remove("llama-cpp-python")
        if missing:
            print(f"Install the remaining dependencies: pip install {' '.join(missing)}")
        sys.exit(1)

    asyncio.run(app.run(args))


if __name__ == "__main__":
    main()
