from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from adapters.input.guichet_unique import GuichetUnique

logger = logging.getLogger(__name__)


class TUIMessageRequest(BaseModel):
    text: str


def create_tui_app(
    gateway: GuichetUnique,
    request_timeout_s: float = 120.0,
) -> FastAPI:
    app = FastAPI(title="Secretarius TUI Channel")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/tui/message")
    async def handle_tui_message(payload: TUIMessageRequest, http_request: Request) -> Dict[str, Any]:
        prompt = (payload.text or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty user message")

        request_id = f"req-{uuid4().hex[:12]}"
        client_ip = http_request.client.host if http_request.client else "unknown"
        logger.info("tui request_start request_id=%s client_ip=%s", request_id, client_ip)

        try:
            result = await asyncio.wait_for(
                gateway.submit_with_trace("tui", prompt),
                timeout=request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("tui request_timeout request_id=%s timeout_s=%.1f", request_id, request_timeout_s)
            raise HTTPException(status_code=504, detail="Agent timeout") from exc
        except Exception as exc:
            logger.exception("tui request_failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        payload_out = {
            "request_id": request_id,
            "reply_text": result["reply_text"],
            "thoughts": result["thoughts"],
            "messages": result["messages"],
        }
        logger.info(
            "tui request_end request_id=%s thoughts=%d messages=%d",
            request_id,
            len(result["thoughts"]),
            len(result["messages"]),
        )
        return payload_out

    return app
