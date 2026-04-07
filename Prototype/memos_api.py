from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

import aiohttp
from fastapi import FastAPI, HTTPException, Request

from adapters.input.guichet_unique import GuichetUnique
from secretarius_markdown import SecretariusMarkdownError, parse_secretarius_markdown, render_secretarius_command

logger = logging.getLogger(__name__)

SUPPORTED_COMMANDS = ("/exp", "/index", "/req", "/update")
SUPPORTED_EVENTS = {
    "memo.created",
    "memo.updated",
    "MEMO_CREATED",
    "MEMO_UPDATED",
}


def _extract_event_type(payload: Dict[str, Any]) -> str:
    for key in ("eventType", "event_type", "type", "event"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_memo(payload: Dict[str, Any]) -> Dict[str, Any]:
    memo = payload.get("memo")
    if isinstance(memo, dict):
        return memo

    data = payload.get("data")
    if isinstance(data, dict):
        nested = data.get("memo")
        if isinstance(nested, dict):
            return nested
        if any(key in data for key in ("name", "content", "parent", "creator", "visibility")):
            return data

    if any(key in payload for key in ("name", "content", "parent", "creator", "visibility")):
        return payload

    return {}


def _memo_id_from_name(memo_name: str) -> str:
    value = (memo_name or "").strip().strip("/")
    if not value:
        return ""
    if value.startswith("memos/"):
        return value.split("/", 1)[1].strip()
    return value


def _memo_content(memo: Dict[str, Any]) -> str:
    for key in ("content", "rawContent", "raw_content"):
        value = memo.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _memo_creator(memo: Dict[str, Any]) -> str:
    creator = memo.get("creator")
    if isinstance(creator, str):
        return creator.strip()
    if isinstance(creator, dict):
        for key in ("name", "id", "username", "email"):
            value = creator.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _starts_with_supported_command(text: str) -> bool:
    normalized = (text or "").lstrip()
    return any(normalized.startswith(command) for command in SUPPORTED_COMMANDS)


def _resolve_memos_input(text: str) -> str:
    parsed = parse_secretarius_markdown(text)
    if parsed is not None:
        return render_secretarius_command(parsed)
    return text


def _format_comment_body(reply_text: str) -> str:
    normalized = (reply_text or "").strip()
    if not normalized:
        normalized = "(reponse vide)"
    return f"Secretarius\n\n{normalized}"


class MemosClient:
    def __init__(
        self,
        base_url: str,
        access_token: str,
        timeout_s: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token.strip()
        self.timeout = aiohttp.ClientTimeout(total=float(timeout_s))

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def create_comment(self, memo_name: str, content: str, visibility: str) -> Dict[str, Any]:
        memo_id = _memo_id_from_name(memo_name)
        if not memo_id:
            raise ValueError("Cannot create a Memos comment without a memo identifier.")

        url = f"{self.base_url}/api/v1/memos/{memo_id}/comments"
        payload = {
            "state": "NORMAL",
            "content": content,
            "visibility": visibility or "PRIVATE",
        }
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload, headers=self._headers()) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Memos API error {response.status}: {text}")
                if not text.strip():
                    return {}
                return await response.json()


def create_memos_app(
    gateway: GuichetUnique,
    memos_base_url: str,
    memos_access_token: str,
    request_timeout_s: float = 120.0,
    publish_timeout_s: float = 30.0,
    response_visibility: str = "PRIVATE",
    webhook_token: str = "",
    ignored_creator: str = "",
    memos_client: Optional[MemosClient] = None,
) -> FastAPI:
    app = FastAPI(title="Secretarius Memos Channel")
    client = memos_client or MemosClient(
        base_url=memos_base_url,
        access_token=memos_access_token,
        timeout_s=publish_timeout_s,
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/memos/webhook")
    async def handle_webhook(http_request: Request) -> Dict[str, Any]:
        payload = await http_request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid webhook payload")

        expected_token = (webhook_token or "").strip()
        if expected_token:
            provided_token = (http_request.query_params.get("token") or "").strip()
            if provided_token != expected_token:
                raise HTTPException(status_code=401, detail="Invalid webhook token")

        request_id = f"req-{uuid4().hex[:12]}"
        client_ip = http_request.client.host if http_request.client else "unknown"
        event_type = _extract_event_type(payload)
        memo = _extract_memo(payload)
        memo_name = str(memo.get("name") or "").strip()
        memo_parent = str(memo.get("parent") or "").strip()
        memo_creator = _memo_creator(memo)
        text = _memo_content(memo)

        logger.info(
            "memos request_start request_id=%s client_ip=%s event_type=%s memo=%s",
            request_id,
            client_ip,
            event_type or "-",
            memo_name or "-",
        )

        if event_type and event_type not in SUPPORTED_EVENTS:
            logger.info("memos request_ignored request_id=%s reason=unsupported_event", request_id)
            return {"status": "ignored", "reason": "unsupported_event", "request_id": request_id}

        if memo_parent:
            logger.info("memos request_ignored request_id=%s reason=comment_event", request_id)
            return {"status": "ignored", "reason": "comment_event", "request_id": request_id}

        if ignored_creator and memo_creator == ignored_creator:
            logger.info("memos request_ignored request_id=%s reason=ignored_creator", request_id)
            return {"status": "ignored", "reason": "ignored_creator", "request_id": request_id}

        if not text:
            raise HTTPException(status_code=400, detail="Missing memo content")

        try:
            resolved_input = _resolve_memos_input(text)
        except SecretariusMarkdownError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid secretarius markdown: {exc}") from exc

        if not _starts_with_supported_command(resolved_input):
            logger.info("memos request_ignored request_id=%s reason=no_supported_command", request_id)
            return {"status": "ignored", "reason": "no_supported_command", "request_id": request_id}

        try:
            reply_text = await asyncio.wait_for(
                gateway.submit("memos", resolved_input),
                timeout=request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("memos request_timeout request_id=%s timeout_s=%.1f", request_id, request_timeout_s)
            raise HTTPException(status_code=504, detail="Agent timeout") from exc
        except Exception as exc:
            logger.exception("memos request_failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        if not memo_name:
            raise HTTPException(status_code=400, detail="Missing memo name")

        try:
            published = await client.create_comment(
                memo_name=memo_name,
                content=_format_comment_body(reply_text),
                visibility=response_visibility,
            )
        except Exception as exc:
            logger.exception("memos publish_failed request_id=%s memo=%s", request_id, memo_name)
            raise HTTPException(status_code=502, detail=f"Memos publish failed: {exc}") from exc

        response = {
            "status": "processed",
            "request_id": request_id,
            "event_type": event_type or "unknown",
            "memo_name": memo_name,
            "reply_text": reply_text,
            "comment_name": published.get("name"),
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "memos request_end request_id=%s memo=%s output_len=%d",
            request_id,
            memo_name,
            len(reply_text),
        )
        return response

    return app
