import asyncio
import threading
from typing import Callable, Awaitable, Any, Optional

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
    ):
        super().__init__()
        self.callback = callback
        self.journal_hook = journal_hook

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-container"):
            yield RichLog(id="chat-log", markup=True)
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
    def __init__(self, guichet: GuichetUnique, channel_name: str = "tui"):
        self._guichet = guichet
        self._channel_name = channel_name
        self._app: SecretariusTUI = None
        self._guichet.register_channel(
            self._channel_name,
            thought_sink=self._on_thought,
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
        )
        await self._app.run_async()
