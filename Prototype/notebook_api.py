from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from adapters.input.guichet_unique import GuichetUnique

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "secretarius-notebook"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: bool = False
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _chat_completion_response(model: str, content: str) -> Dict[str, Any]:
    created_ts = int(datetime.now(timezone.utc).timestamp())
    completion_id = f"chatcmpl-{uuid4()}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_chunks(model: str, content: str) -> AsyncIterator[str]:
    created_ts = int(datetime.now(timezone.utc).timestamp())
    completion_id = f"chatcmpl-{uuid4()}"
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    content_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def create_notebook_app(
    gateway: GuichetUnique,
    model_id: str = "secretarius-notebook",
    request_timeout_s: float = 120.0,
) -> FastAPI:
    app = FastAPI(title="Secretarius Notebook Channel")

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        prompt = (user_messages[-1].content or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty user message")

        request_id = f"req-{uuid4().hex[:12]}"
        client_ip = http_request.client.host if http_request.client else "unknown"
        logger.info(
            "notebook request_start request_id=%s client_ip=%s model=%s stream=%s",
            request_id,
            client_ip,
            request.model,
            request.stream,
        )

        try:
            output_text = await asyncio.wait_for(
                gateway.submit("notebook", prompt),
                timeout=request_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("notebook request_timeout request_id=%s timeout_s=%.1f", request_id, request_timeout_s)
            raise HTTPException(status_code=504, detail="Agent timeout") from exc
        except Exception as exc:
            logger.exception("notebook request_failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        if request.stream:
            logger.info("notebook request_end request_id=%s stream=true", request_id)
            return StreamingResponse(
                _stream_chunks(request.model, output_text),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id},
            )

        payload = _chat_completion_response(request.model, output_text)
        payload["x_request_id"] = request_id
        logger.info("notebook request_end request_id=%s stream=false", request_id)
        return payload

    return app
