"""Minimal OpenAI-compatible HTTP server exposing the loaded local model.

No MCP tool-calling here by design — this is a small subset of the OpenAI
chat completions API (/v1/models, /v1/chat/completions, streaming included),
enough for any tool or SDK that already speaks that protocol to talk to the
local model, e.g. by pointing an OpenAI client at http://127.0.0.1:8000/v1.
Binds to localhost only unless --host is overridden explicitly.

Built on FastAPI so interactive API docs come for free: Swagger UI at /docs,
ReDoc at /redoc, raw schema at /openapi.json.
"""

import asyncio
import json
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import C_CYAN, C_GREEN, C_RESET, C_YELLOW
from .web import FAVICON_SVG, render_index

# llama.cpp inference is not safe to run concurrently from multiple threads,
# so every request (streaming or not) serializes on this lock.
_llm_lock = asyncio.Lock()


class ChatMessage(BaseModel):
    role: str = Field(examples=["user"])
    content: str = Field(examples=["Hello!"])


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = Field(default=None, description="Ignored — only one model is loaded per server.")
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = Field(default=False, description="If true, respond with server-sent events.")


async def _stream_chat_completion(llm, kwargs):
    async with _llm_lock:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker():
            try:
                for chunk in llm.create_chat_completion(stream=True, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": {"message": str(exc)}})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def create_app(llm, model_name: str) -> FastAPI:
    app = FastAPI(
        title="Dragon CLI — local model server",
        description="Minimal OpenAI-compatible API for the locally loaded GGUF model. "
                     "No MCP tool-calling here — model only.",
        version="1.0.0",
    )
    app.state.llm = llm
    app.state.model_name = model_name

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(render_index(model_name, str(request.base_url)))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(FAVICON_SVG, media_type="image/svg+xml")

    @app.get("/v1/models", tags=["models"], summary="List the loaded model")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "dragon-cli"}],
        }

    @app.post("/v1/chat/completions", tags=["chat"], summary="Create a chat completion")
    async def chat_completions(body: ChatCompletionRequest):
        kwargs = {
            "messages": [m.model_dump() for m in body.messages],
            "temperature": body.temperature,
            "max_tokens": body.max_tokens,
        }

        if not body.stream:
            async with _llm_lock:
                try:
                    return await run_in_threadpool(llm.create_chat_completion, stream=False, **kwargs)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc))

        return StreamingResponse(_stream_chat_completion(llm, kwargs), media_type="text/event-stream")

    return app


async def run_server(llm, model_name: str, host: str, port: int) -> None:
    app = create_app(llm, model_name)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    print(f"{C_GREEN}[serve]{C_RESET} OpenAI-compatible endpoint at "
          f"{C_CYAN}http://{host}:{port}/v1{C_RESET}")
    print(f"{C_GREEN}[serve]{C_RESET} Interactive docs (Swagger UI): "
          f"{C_CYAN}http://{host}:{port}/docs{C_RESET}")
    print(f"{C_YELLOW}[serve]{C_RESET} Model: {model_name}. Press Ctrl+C to stop.")
    await server.serve()
