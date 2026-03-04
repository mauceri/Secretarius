import asyncio
import inspect
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from core.ports import InputGatewayInterface


ThoughtSink = Callable[[str], object]
MessageSink = Callable[[str, str], object]


class GuichetUnique(InputGatewayInterface):
    def __init__(self, journal_path: Optional[str] = None):
        self._callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._journal_path = Path(journal_path) if journal_path else None
        self._current_channel: ContextVar[str] = ContextVar("current_channel", default="default")
        self._current_collector: ContextVar[Optional[list[tuple[str, str]]]] = ContextVar(
            "current_collector",
            default=None,
        )
        self._channel_sinks: dict[str, tuple[Optional[ThoughtSink], Optional[MessageSink]]] = {}
        self._submit_lock = asyncio.Lock()

        if self._journal_path:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._append_journal_line("SYSTEM", "JOURNAL", "Guichet unique started", channel="system")

    def _append_journal_line(self, stream: str, role: str, text: str, channel: str) -> None:
        if not self._journal_path:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        try:
            with self._journal_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp}\t{channel}\t{stream}\t{role}\t{normalized}\n")
        except OSError:
            return

    async def _maybe_await(self, value: object) -> None:
        if inspect.isawaitable(value):
            await value

    def register_channel(
        self,
        channel: str,
        thought_sink: Optional[ThoughtSink] = None,
        message_sink: Optional[MessageSink] = None,
    ) -> None:
        self._channel_sinks[channel] = (thought_sink, message_sink)
        self._append_journal_line("SYSTEM", "CHANNEL", f"registered {channel}", channel="system")

    def set_callback(self, callback: Callable[[str], Awaitable[None]]):
        self._callback = callback

    async def display_thought(self, thought: str):
        channel = self._current_channel.get()
        self._append_journal_line("THOUGHT", "ASSISTANT", thought, channel=channel)
        thought_sink, _ = self._channel_sinks.get(channel, (None, None))
        if thought_sink is not None:
            await self._maybe_await(thought_sink(thought))

    async def display_message(self, role: str, content: str):
        channel = self._current_channel.get()
        normalized_role = role if isinstance(role, str) else "assistant"
        self._append_journal_line("CHAT", normalized_role.upper(), content, channel=channel)
        collector = self._current_collector.get()
        if collector is not None:
            collector.append((normalized_role, content))
        _, message_sink = self._channel_sinks.get(channel, (None, None))
        if message_sink is not None:
            await self._maybe_await(message_sink(normalized_role, content))

    async def run(self):
        # The guichet itself has no event loop.
        return

    async def submit(self, channel: str, user_input: str) -> str:
        if self._callback is None:
            raise RuntimeError("No orchestrator callback configured.")

        async with self._submit_lock:
            self._append_journal_line("CHAT", "USER", user_input, channel=channel)
            collector: list[tuple[str, str]] = []
            token_channel = self._current_channel.set(channel)
            token_collector = self._current_collector.set(collector)
            try:
                await self._callback(user_input)
            finally:
                self._current_channel.reset(token_channel)
                self._current_collector.reset(token_collector)

            for role, content in reversed(collector):
                if role.lower() in ("assistant", "secretarius"):
                    return content
            for role, content in reversed(collector):
                if role.lower() == "system":
                    return content
            return "Aucune reponse finale n'a ete produite."

