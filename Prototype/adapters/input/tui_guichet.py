import asyncio
import json
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Awaitable, Optional

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, RichLog
from rich.markup import escape

from adapters.input.guichet_unique import GuichetUnique

class SecretariusTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #chat-container {
        height: 3fr;
        border: solid green;
    }
    #thought-container {
        height: 1fr;
        border: solid yellow;
        color: yellow;
    }
    Input {
        dock: bottom;
    }
    """

    def __init__(
        self,
        callback: Callable[[str], Awaitable[None]] = None,
        journal_hook: Optional[Callable[[str, str, str], None]] = None,
        show_thoughts: bool = True,
    ):
        super().__init__()
        self.callback = callback
        self.journal_hook = journal_hook
        self.show_thoughts = show_thoughts

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-container"):
            yield RichLog(id="chat-log", markup=True)
        if self.show_thoughts:
            with VerticalScroll(id="thought-container"):
                yield RichLog(id="thought-log", markup=True)
        yield Input(placeholder="Type your message to Secretarius...", id="user-input")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value
        if not user_text.strip():
            return
            
        event.input.value = ""
        
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold blue]User:[/bold blue] {escape(user_text)}")
        if self.journal_hook:
            self.journal_hook("CHAT", "USER", user_text)

        if self.callback:
            # Run the orchestrator cycle in background
            asyncio.create_task(self.callback(user_text))

    def log_thought(self, text: str):
        if not self.show_thoughts:
            return
        thought_log = self.query_one("#thought-log", RichLog)
        thought_log.write(f"[italic]{escape(text)}[/italic]")
        if self.journal_hook:
            self.journal_hook("THOUGHT", "ASSISTANT", text)

    def log_message(self, role: str, text: str):
        chat_log = self.query_one("#chat-log", RichLog)
        if role.lower() == "system":
            chat_log.write(f"[bold red]System:[/bold red] {escape(text)}")
            if self.journal_hook:
                self.journal_hook("CHAT", "SYSTEM", text)
        else:
            chat_log.write(f"[bold green]Secretarius:[/bold green] {escape(text)}")
            if self.journal_hook:
                self.journal_hook("CHAT", "ASSISTANT", text)


class TUIChannel:
    def __init__(self, guichet: GuichetUnique, channel_name: str = "tui", show_thoughts: bool = True):
        self._guichet = guichet
        self._channel_name = channel_name
        self._show_thoughts = show_thoughts
        self._app: SecretariusTUI = None
        self._guichet.register_channel(
            self._channel_name,
            thought_sink=self._on_thought if self._show_thoughts else None,
            message_sink=self._on_message,
        )

    def _dispatch_ui_update(self, callback: Callable[..., None], *args: Any) -> None:
        if not self._app:
            return
        app_thread_id = getattr(self._app, "_thread_id", None)
        current_thread_id = threading.get_ident()
        try:
            if app_thread_id is not None and current_thread_id == app_thread_id:
                # Already on UI thread: schedule safely in Textual's refresh cycle.
                self._app.call_after_refresh(callback, *args)
            else:
                # Called from a worker/task thread.
                self._app.call_from_thread(callback, *args)
        except RuntimeError:
            # Last-resort fallback to avoid losing visible output.
            callback(*args)

    def _on_thought(self, thought: str) -> None:
        self._dispatch_ui_update(self._app.log_thought, thought)

    def _on_message(self, role: str, content: str) -> None:
        self._dispatch_ui_update(self._app.log_message, role, content)

    async def _on_user_input(self, user_input: str) -> None:
        await self._guichet.submit(self._channel_name, user_input)

    async def run(self):
        self._app = SecretariusTUI(
            callback=self._on_user_input,
            journal_hook=None,
            show_thoughts=self._show_thoughts,
        )
        await self._app.run_async()


class TUIAPIClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0):
        self._url = f"{base_url.rstrip('/')}/tui/message"
        self._timeout_s = timeout_s

    async def submit(self, text: str) -> dict:
        return await asyncio.to_thread(self._submit_blocking, text)

    def _submit_blocking(self, text: str) -> dict:
        payload = json.dumps({"text": text}).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TUI API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TUI API unavailable: {exc.reason}") from exc


class RemoteTUIChannel:
    def __init__(self, api_base_url: str, show_thoughts: bool = True):
        self._api_client = TUIAPIClient(api_base_url, timeout_s=120.0)
        self._show_thoughts = show_thoughts
        self._app: SecretariusTUI | None = None

    async def _on_user_input(self, user_input: str) -> None:
        if self._app is None:
            return
        try:
            payload = await self._api_client.submit(user_input)
        except Exception as exc:
            self._app.log_message("system", str(exc))
            return

        if self._show_thoughts:
            for thought in payload.get("thoughts", []):
                self._app.log_thought(str(thought))

        messages = payload.get("messages", [])
        displayed_assistant = False
        for message in messages:
            role = str(message.get("role", "assistant"))
            content = str(message.get("content", ""))
            self._app.log_message(role, content)
            if role.lower() in ("assistant", "secretarius", "system"):
                displayed_assistant = True

        if not displayed_assistant:
            self._app.log_message("assistant", str(payload.get("reply_text", "")))

    async def run(self):
        self._app = SecretariusTUI(
            callback=self._on_user_input,
            journal_hook=None,
            show_thoughts=self._show_thoughts,
        )
        await self._app.run_async()
