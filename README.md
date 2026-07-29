# Dragon CLI

Chat with local GGUF LLMs (100% CPU, via `llama-cpp-python`), with optional
MCP (Model Context Protocol) tool-calling and a SQLite-backed conversation
history.

## Install

```bash
pip install -e .
dragon-cli-install-llm
```

This exposes a `dragon-cli` command. `llama-cpp-python` is intentionally kept
out of the core dependencies, since it ships as C++ source on PyPI for many
platforms — a plain `pip install` would try to compile it locally and fail
without a C/C++ toolchain (Visual Studio Build Tools on Windows, gcc on
Linux, Xcode command line tools on macOS).

`dragon-cli-install-llm` detects your OS/architecture (Windows, Linux, or
macOS) and installs a prebuilt CPU wheel instead, so a compiler usually isn't
needed at all. If no prebuilt wheel is available for your exact
platform/Python combination, it prints the toolchain to install and the
fallback command to run.

If you already have a working C/C++ toolchain and would rather build from
source yourself, skip the installer and use the extra instead:
`pip install -e ".[llm]"`.

You can also run it without installing:

```bash
python dragon_cli.py
# or
python -m dragon_cli
```

## Usage

```bash
dragon-cli                       # model menu + chat
dragon-cli --model 2             # skip the model menu
dragon-cli --list                # just list the model catalog
dragon-cli --no-mcp              # disable MCP connectors
dragon-cli --resume 3            # resume saved conversation #3
dragon-cli --list-conversations  # list saved conversations and exit
dragon-cli --db path/to/other.db # use a different conversation database
dragon-cli --serve               # serve the model over HTTP instead of chatting
dragon-cli --serve --port 8080   # custom port (default: 8000)
dragon-cli --serve --host 0.0.0.0  # expose beyond localhost (e.g. to your LAN) — opt-in only
```

## HTTP server (`--serve`)

`dragon-cli --serve` loads a model and exposes it over a minimal
OpenAI-compatible HTTP API instead of starting the interactive chat. It's
built on FastAPI, so interactive docs come for free:

- `GET /` — landing page with usage examples
- `GET /docs` — Swagger UI
- `GET /redoc` — ReDoc
- `GET /openapi.json` — raw OpenAPI schema
- `GET /v1/models` — lists the currently loaded model
- `POST /v1/chat/completions` — same shape as OpenAI's endpoint, streaming
  (`"stream": true`, server-sent events) and non-streaming both supported,
  request body validated (422 with a clear error on malformed input)

Point any OpenAI-compatible client/SDK at `http://127.0.0.1:8000/v1` (adjust
host/port to match `--host`/`--port`) and it works without extra glue code.
This mode does **not** expose MCP tool-calling — it's the model only,
kept intentionally simple and predictable. It also binds to `127.0.0.1` by
default (only reachable from your own machine); pass `--host 0.0.0.0`
explicitly if you want it reachable from other devices on your network.

## In-chat commands

| Command | Description |
|---|---|
| `/help` | show available commands |
| `/models` | list the model catalog |
| `/tools` | list connected MCP tools |
| `/mcp` | show MCP server connection status |
| `/clear` | clear the current in-memory history |
| `/new [title]` | start a new conversation |
| `/list` | list saved conversations |
| `/resume <id>` | resume a saved conversation |
| `/rename <id> <title>` | rename a saved conversation |
| `/delete <id>` | delete a saved conversation |
| `/search <text>` | search saved messages for text |
| `/export <id> [md\|json]` | export a conversation to a file |
| `/exit`, `/quit` | end the program |

Every message is autosaved to SQLite (`dragon_cli.db` by default) as soon as
it's sent, so nothing is lost if the process crashes.

## MCP connectors

On first run, an example `mcp_servers.json` is created in the working
directory. Edit it to point at your MCP servers and restart to connect.

## Project layout

```
src/dragon_cli/
├── cli.py            # argparse entry point
├── app.py             # model selection + orchestration
├── setup_llm.py        # cross-platform llama-cpp-python installer
├── server.py            # OpenAI-compatible HTTP server (--serve)
├── banner.py            # ASCII banner
├── config.py             # shared paths/colors
├── models/            # model catalog, download, loading
├── mcp/                # MCP server manager + tool-call parsing
├── storage/            # SQLite conversation store
└── chat/                # chat session, streaming, slash commands
```
