import aiohttp
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from core.ports import LLMInterface
from core.models import Message

logger = logging.getLogger(__name__)

class OllamaAdapter(LLMInterface):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:0.6b",
        think: bool = False,
        request_timeout_s: float = 180.0,
        raw_log_path: str | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.think = think
        self.api_url = f"{self.base_url}/api/chat"
        self.timeout = aiohttp.ClientTimeout(total=float(request_timeout_s))
        self.raw_log_path = Path(raw_log_path) if raw_log_path else None
        if self.raw_log_path:
            self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_raw_log(self, stream: str, text: str) -> None:
        if not self.raw_log_path:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        try:
            with self.raw_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp}\tMODEL\t{self.model}\n")
                handle.write(f"{timestamp}\tSTREAM\t{stream}\n")
                handle.write(normalized + "\n")
                handle.write("---\n")
        except OSError:
            return

    async def generate_response(self, messages: List[Message], system_prompt: str) -> str:
        # Prepare messages in Ollama format
        ollama_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            om = {"role": msg.role.value, "content": msg.content}
            # Although Ollama doesn't strictly use 'name', we format it inside content if needed,
            # or pass it along as best effort. For tool responses we might prefix.
            if msg.role.value == "tool" and msg.name:
                om["content"] = f"Tool {msg.name} returned:\n{msg.content}"
            ollama_messages.append(om)

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "think": self.think,
            # Instruct Ollama to use JSON output format if supported by the model
            "format": "json" 
        }
        self._append_raw_log("REQUEST", json.dumps(payload, ensure_ascii=False, indent=2))

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.api_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    self._append_raw_log("RESPONSE_RAW", json.dumps(data, ensure_ascii=False, indent=2))
                    content = data.get("message", {}).get("content", "")
                    self._append_raw_log("RESPONSE_CONTENT", content)
                    return content
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            raise
