"""Chat session state: history, current conversation, and the main loop."""

import asyncio

from ..config import C_DIM, C_GREEN, C_YELLOW, C_RESET
from ..mcp import MCPManager, extract_tool_call
from ..storage import ConversationStore
from .commands import dispatch
from .streaming import generate_streaming

SYSTEM_PROMPT = "You are a helpful assistant running locally on the user's machine."


class ChatSession:
    """Bundles the LLM, MCP manager, conversation store, and in-memory history."""

    def __init__(self, llm, mcp: MCPManager, store: ConversationStore, model_name: str):
        self.llm = llm
        self.mcp = mcp
        self.store = store
        self.model_name = model_name
        self.conv_id: int | None = None
        self.history: list[dict] = []

    def _system_message(self) -> str:
        return SYSTEM_PROMPT + self.mcp.tools_prompt()

    def start_new(self, title: str | None = None) -> None:
        title = title or "New conversation"
        self.conv_id = self.store.create(title, self.model_name)
        self.history = [{"role": "system", "content": self._system_message()}]
        self.store.append_message(self.conv_id, "system", self.history[0]["content"])

    def resume(self, conv_id: int) -> bool:
        found = self.store.get(conv_id)
        if found is None:
            return False
        conv, messages = found
        self.conv_id = conv.id
        self.history = [{"role": m.role, "content": m.content} for m in messages]
        if not self.history or self.history[0]["role"] != "system":
            self.history.insert(0, {"role": "system", "content": self._system_message()})
        return True

    def clear(self) -> None:
        self.history = self.history[:1]

    async def turn(self, user_input: str) -> None:
        self.history.append({"role": "user", "content": user_input})
        self.store.append_message(self.conv_id, "user", user_input)

        for _ in range(5):
            response = await asyncio.to_thread(generate_streaming, self.llm, self.history)
            self.history.append({"role": "assistant", "content": response})
            self.store.append_message(self.conv_id, "assistant", response)

            call = extract_tool_call(response)
            if not call or not self.mcp.tools:
                break
            tool_name, arguments = call
            print(f"{C_DIM}[tool] running {tool_name}({arguments}){C_RESET}")
            result = await self.mcp.call(tool_name, arguments)
            print(f"{C_DIM}[tool] result: {result[:200]}...{C_RESET}")
            followup = (
                f"[TOOL RESULT for {tool_name}]:\n{result}\n\n"
                f"Now answer the user's original question based on this."
            )
            self.history.append({"role": "user", "content": followup})
            self.store.append_message(self.conv_id, "user", followup)


async def chat_loop(session: ChatSession) -> None:
    if session.conv_id is None:
        session.start_new()

    print(f"\n{C_GREEN}=== Chat started — /help for commands, /exit to quit ==={C_RESET}\n")
    while True:
        try:
            user_input = (await asyncio.to_thread(input, f"{C_YELLOW}You:{C_RESET} ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            keep_going = dispatch(session, user_input)
            if keep_going is False:
                break
            continue

        await session.turn(user_input)

    print("Shutting down. See you next time!")
