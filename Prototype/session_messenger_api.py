from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from adapters.input.guichet_unique import GuichetUnique

logger = logging.getLogger(__name__)


class SessionMessengerRequest(BaseModel):
    message_id: str
    sender_id: str
    text: str
    sent_at: Optional[str] = None
    bot_session_id: Optional[str] = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def create_session_messenger_app(
    gateway: GuichetUnique,
    request_timeout_s: float = 120.0,
) -> FastAPI:
    app = FastAPI(title="Secretarius Session Messenger Channel")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/session/message")
    async def handle_session_message(payload: SessionMessengerRequest, http_request: Request) -> Dict[str, Any]:
        text = (payload.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty user message")

        request_id = f"req-{uuid4().hex[:12]}"
        client_ip = http_request.client.host if http_request.client else "unknown"
        logger.info(
            "session_messenger request_start request_id=%s client_ip=%s sender_id=%s message_id=%s bot_session_id=%s attachments=%d",
            request_id,
            client_ip,
            payload.sender_id,
            payload.message_id,
            payload.bot_session_id or "-",
            len(payload.attachments),
        )

        try:
            output_text = await asyncio.wait_for(
                gateway.submit("session_messenger", text),
                timeout=request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "session_messenger request_timeout request_id=%s timeout_s=%.1f",
                request_id,
                request_timeout_s,
            )
            raise HTTPException(status_code=504, detail="Agent timeout") from exc
        except Exception as exc:
            logger.exception("session_messenger request_failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        response = {
            "request_id": request_id,
            "reply_text": output_text,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "session_messenger request_end request_id=%s output_len=%d",
            request_id,
            len(output_text),
        )
        return response

    return app
