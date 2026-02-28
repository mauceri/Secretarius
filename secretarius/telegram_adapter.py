from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from .channel_adapters import ChannelEvent, handle_channel_event


def _parse_allowed_chat_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(int(item))
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    poll_timeout_s: int
    poll_interval_s: float
    allowed_chat_ids: set[int]

    @staticmethod
    def from_env() -> "TelegramConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
        poll_timeout_s = int(os.environ.get("TELEGRAM_POLL_TIMEOUT_S", "25"))
        poll_interval_s = float(os.environ.get("TELEGRAM_POLL_INTERVAL_S", "0.5"))
        allowed_chat_ids = _parse_allowed_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS"))
        return TelegramConfig(
            token=token,
            poll_timeout_s=max(1, poll_timeout_s),
            poll_interval_s=max(0.0, poll_interval_s),
            allowed_chat_ids=allowed_chat_ids,
        )


class TelegramPollingAdapter:
    def __init__(self, config: TelegramConfig | None = None) -> None:
        self.config = config or TelegramConfig.from_env()
        self.base_url = f"https://api.telegram.org/bot{self.config.token}"
        self.offset = 0
        self.session = requests.Session()

    def run_forever(self) -> None:
        print("Telegram adapter started (polling).")
        while True:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
            except KeyboardInterrupt:
                print("Telegram adapter stopped.")
                return
            except Exception as exc:
                print(f"[telegram] polling error: {exc}")
                time.sleep(max(self.config.poll_interval_s, 1.0))
            time.sleep(self.config.poll_interval_s)

    def _get_updates(self) -> list[dict[str, Any]]:
        payload = {
            "timeout": self.config.poll_timeout_s,
            "offset": self.offset,
            "allowed_updates": ["message"],
        }
        resp = self.session.post(
            f"{self.base_url}/getUpdates",
            json=payload,
            timeout=self.config.poll_timeout_s + 10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {data}")
        result = data.get("result")
        if not isinstance(result, list):
            return []
        return [u for u in result if isinstance(u, dict)]

    def _handle_update(self, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self.offset = max(self.offset, update_id + 1)

        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        chat = message.get("chat")
        if not isinstance(chat, dict):
            return
        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            return

        if self.config.allowed_chat_ids and chat_id not in self.config.allowed_chat_ids:
            self._send_message(chat_id, "Acces refuse pour ce chat.")
            return

        from_user = message.get("from")
        user_id = str((from_user or {}).get("id", "unknown"))
        session_id = f"telegram:{chat_id}"
        event = ChannelEvent(
            channel="telegram",
            user_id=user_id,
            session_id=session_id,
            text=text.strip(),
            metadata={"chat_id": chat_id, "update_id": update_id},
        )
        try:
            result = handle_channel_event(event)
            output_text = str(result.get("output_text", "")).strip() or "(aucune reponse)"
        except Exception as exc:
            output_text = f"Erreur agent: {exc}"
        self._send_message(chat_id, output_text)

    def _send_message(self, chat_id: int, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text[:4000],  # limite Telegram
        }
        resp = self.session.post(
            f"{self.base_url}/sendMessage",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

