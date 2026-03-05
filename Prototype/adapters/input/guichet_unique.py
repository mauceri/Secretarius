import asyncio
import inspect
import re
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from core.ports import InputGatewayInterface


ThoughtSink = Callable[[str], object]
MessageSink = Callable[[str, str], object]


class GuichetUnique(InputGatewayInterface):
    DEFAULT_RECENT_RESULT_TTL_S = 15.0

    def __init__(
        self,
        journal_path: Optional[str] = None,
        channel_journal_paths: Optional[dict[str, str]] = None,
        recent_result_ttl_s: float = DEFAULT_RECENT_RESULT_TTL_S,
    ):
        self._callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._journal_path = Path(journal_path) if journal_path else None
        self._channel_journal_paths: dict[str, Path] = {}
        self._current_channel: ContextVar[str] = ContextVar("current_channel", default="default")
        self._current_collector: ContextVar[Optional[list[tuple[str, str]]]] = ContextVar(
            "current_collector",
            default=None,
        )
        self._channel_sinks: dict[str, tuple[Optional[ThoughtSink], Optional[MessageSink]]] = {}
        self._inflight_lock = asyncio.Lock()
        self._inflight_submissions: dict[tuple[str, str], asyncio.Future[str]] = {}
        self._recent_results: dict[tuple[str, str], tuple[float, str]] = {}
        self._recent_result_ttl_s = max(0.0, float(recent_result_ttl_s))

        if self._journal_path:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        if channel_journal_paths:
            for channel, path in channel_journal_paths.items():
                if not channel or not path:
                    continue
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                self._channel_journal_paths[channel] = p

        if self._journal_path:
            self._append_journal_line("SYSTEM", "JOURNAL", "Guichet unique started", channel="system")

    def _append_journal_line(self, stream: str, role: str, text: str, channel: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        line = f"{timestamp}\t{channel}\t{stream}\t{role}\t{normalized}\n"

        if self._journal_path:
            try:
                with self._journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass

        channel_path = self._channel_journal_paths.get(channel)
        if channel_path is not None:
            if self._journal_path and channel_path.resolve() == self._journal_path.resolve():
                return
            try:
                with channel_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
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

    @staticmethod
    def _normalize_submission_text(user_input: str) -> str:
        text = (user_input or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    async def _run_submission(self, channel: str, user_input: str) -> str:
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

    async def submit(self, channel: str, user_input: str) -> str:
        if self._callback is None:
            raise RuntimeError("No orchestrator callback configured.")

        key = (channel, self._normalize_submission_text(user_input))
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        is_leader = False
        cached_result: str | None = None

        async with self._inflight_lock:
            future = self._inflight_submissions.get(key)
            if future is None:
                previous = self._recent_results.get(key)
                if previous is not None:
                    ts, result = previous
                    if now - ts <= self._recent_result_ttl_s:
                        cached_result = result
                    else:
                        del self._recent_results[key]
            if future is None and cached_result is None:
                future = loop.create_future()
                self._inflight_submissions[key] = future
                is_leader = True

        if cached_result is not None:
            self._append_journal_line(
                "SYSTEM",
                "DEDUP",
                "returned cached recent result",
                channel=channel,
            )
            return cached_result

        if not is_leader:
            return await future

        try:
            result = await self._run_submission(channel, user_input)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        else:
            if not future.done():
                future.set_result(result)
            async with self._inflight_lock:
                self._recent_results[key] = (time.monotonic(), result)
            return result
        finally:
            async with self._inflight_lock:
                current = self._inflight_submissions.get(key)
                if current is future:
                    del self._inflight_submissions[key]
                # Prune stale cached results.
                stale_before = time.monotonic() - self._recent_result_ttl_s
                stale_keys = [k for k, (ts, _) in self._recent_results.items() if ts < stale_before]
                for stale_key in stale_keys:
                    del self._recent_results[stale_key]
