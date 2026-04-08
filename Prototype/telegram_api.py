"""Canal Telegram pour Secretarius — polling asynchrone via aiohttp."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp

from adapters.input.guichet_unique import GuichetUnique

logger = logging.getLogger("secretarius.telegram")


def _parse_allowed_chat_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    result: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        try:
            result.add(int(item))
        except ValueError:
            continue
    return result


class TelegramChannel:
    def __init__(
        self,
        guichet: GuichetUnique,
        channel_name: str = "telegram",
        poll_timeout_s: int = 25,
        poll_interval_s: float = 0.5,
        request_timeout_s: float = 120.0,
    ) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN manquant dans l'environnement")
        self._guichet = guichet
        self._channel = channel_name
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._poll_timeout_s = poll_timeout_s
        self._poll_interval_s = poll_interval_s
        self._request_timeout_s = request_timeout_s
        self._allowed_chat_ids = _parse_allowed_chat_ids(
            os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS")
        )
        self._offset = 0

    async def run(self) -> None:
        logger.info("Canal Telegram démarré (polling).")
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    updates = await self._get_updates(session)
                    for update in updates:
                        asyncio.create_task(self._handle_update(session, update))
                except asyncio.CancelledError:
                    logger.info("Canal Telegram arrêté.")
                    return
                except Exception as exc:
                    logger.error("Erreur polling Telegram : %s", exc)
                    await asyncio.sleep(max(self._poll_interval_s, 1.0))
                    continue
                await asyncio.sleep(self._poll_interval_s)

    async def _get_updates(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        payload = {
            "timeout": self._poll_timeout_s,
            "offset": self._offset,
            "allowed_updates": ["message"],
        }
        timeout = aiohttp.ClientTimeout(total=self._poll_timeout_s + 10)
        async with session.post(
            f"{self._base_url}/getUpdates", json=payload, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError(f"getUpdates échoué : {data}")
        result = data.get("result")
        if not isinstance(result, list):
            return []
        return [u for u in result if isinstance(u, dict)]

    async def _handle_update(self, session: aiohttp.ClientSession, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self._offset = max(self._offset, update_id + 1)

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

        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            await self._send_message(session, chat_id, "Accès refusé pour ce chat.")
            return

        from_user = message.get("from") or {}
        session_id = f"telegram:{chat_id}"
        logger.info("telegram message chat_id=%d session_id=%s", chat_id, session_id)
        try:
            reply = await asyncio.wait_for(
                self._guichet.submit(self._channel, text.strip()),
                timeout=self._request_timeout_s,
            )
        except asyncio.TimeoutError:
            reply = "Délai d'attente dépassé."
        except Exception as exc:
            logger.exception("Erreur guichet telegram session_id=%s", session_id)
            reply = f"Erreur agent : {exc}"
        await self._send_message(session, chat_id, reply)

    async def _send_message(self, session: aiohttp.ClientSession, chat_id: int, text: str) -> None:
        payload = {"chat_id": chat_id, "text": text[:4000]}
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with session.post(
                f"{self._base_url}/sendMessage", json=payload, timeout=timeout
            ) as resp:
                resp.raise_for_status()
        except Exception as exc:
            logger.error("Erreur sendMessage chat_id=%d : %s", chat_id, exc)
