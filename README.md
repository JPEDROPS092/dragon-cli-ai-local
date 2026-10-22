# Dragon CLI

![1785346956957](https://raw.githubusercontent.com/JPEDROPS092/dragon-cli-ai-local/main/image/README/1785346956957.png)

[![License: MIT](https://img.shields.io/github/license/JPEDROPS092/dragon-cli-ai-local)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Docs](<https://img.shields.io/badge/docs-GitHub%20Pages-7fe0d1>)](https://jpedrops092.github.io/dragon-cli-ai-local/)
[![GitHub stars](https://img.shields.io/github/stars/JPEDROPS092/dragon-cli-ai-local?style=social)](https://github.com/JPEDROPS092/dragon-cli-ai-local/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/JPEDROPS092/dragon-cli-ai-local)](https://github.com/JPEDROPS092/dragon-cli-ai-local/issues)

[![PyPI](https://img.shields.io/pypi/v/dragon-cli)](https://pypi.org/project/dragon-cli/)

Chat with local GGUF LLMs on CPU or GPU (via `llama-cpp-python`), with
MCP (Model Context Protocol) tool-calling, a SQLite-backed conversation
history, and an OpenAI-compatible HTTP server you can plug into agent
frameworks (OpenAI SDK, OpenAI Agents SDK, LangChain, LlamaIndex, CrewAI,
Continue, Aider, Open WebUI...).

📖 **Full docs, guides and agent-framework examples:**
https://jpedrops092.github.io/dragon-cli-ai-local/

## Install

Requires Python 3.10+ on Windows, Linux or macOS.

```bash
pip install dragon-cli
dragon-cli-install-llm   # installs the right llama-cpp-python build for your CPU/GPU
dragon-cli               # interactive menu
```

Upgrade with `pip install -U dragon-cli`. A virtual environment is
recommended (`python -m venv .venv` and activate it first).

`llama-cpp-python` is intentionally kept out of the core dependencies, since
it ships as C++ source on PyPI for many platforms, so a plain `pip install`
would try to compile it locally and fail without a C/C++ toolchain (Visual
Studio Build Tools on Windows, gcc on Linux, Xcode command line tools on
macOS).

`dragon-cli-install-llm` detects your OS/architecture and GPU (NVIDIA CUDA,
AMD ROCm, Apple Metal) and installs a prebuilt wheel instead, so a compiler
usually isn't needed at all. If no prebuilt wheel is available for your exact
platform/Python combination, it prints the toolchain to install and the
fallback command to run.

If you already have a working C/C++ toolchain and would rather build from
source yourself: `pip install "dragon-cli[llm]"`.

Models (`./models`), `mcp_servers.json` and `dragon_cli.db` are created in
the folder you run `dragon-cli` from.

### Other ways to install

```bash
# latest development version from GitHub
pip install git+https://github.com/JPEDROPS092/dragon-cli-ai-local

# from a local clone (editable)
git clone https://github.com/JPEDROPS092/dragon-cli-ai-local
cd dragon-cli-ai-local
pip install -e .

# or run without installing
python dragon_cli.py
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
dragon-cli --profile             # show detected CPU/RAM/GPU and exit
dragon-cli --recommend           # rank catalog models for your hardware
dragon-cli --auto                # pick the best model for your hardware and chat
dragon-cli --priority speed      # speed | quality | balanced | memory (for --auto/--recommend)
dragon-cli --gpu / --cpu         # force full GPU offload, or CPU only
dragon-cli --ngl 20              # offload exactly 20 layers to the GPU
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

| Command                    | Description                         |
| -------------------------- | ----------------------------------- |
| `/help`                  | show available commands             |
| `/models`                | list the model catalog              |
| `/tools`                 | list connected MCP tools            |
| `/mcp`                   | show MCP server connection status   |
| `/clear`                 | clear the current in-memory history |
| `/new [title]`           | start a new conversation            |
| `/list`                  | list saved conversations            |
| `/resume <id>`           | resume a saved conversation         |
| `/rename <id> <title>`   | rename a saved conversation         |
| `/delete <id>`           | delete a saved conversation         |
| `/search <text>`         | search saved messages for text      |
| `/export <id> [md\|json]` | export a conversation to a file     |
| `/exit`, `/quit`       | end the program                     |

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
