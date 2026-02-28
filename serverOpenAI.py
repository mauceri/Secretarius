#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from secretarius.agent_runtime import get_runtime

app = FastAPI(title="Secretarius Agent API")
LOGGER = logging.getLogger("secretarius.api")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "secretarius-agent"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: bool = False
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def run_agent(messages: List[Message]) -> str:
    runtime = get_runtime()
    payload: list[dict[str, str]] = []
    use_last_user_only = os.environ.get("SECRETARIUS_LAST_USER_ONLY", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if use_last_user_only:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user is not None:
            payload = [{"role": "user", "content": last_user.content or ""}]
        else:
            payload = [{"role": m.role, "content": m.content or ""} for m in messages]
    else:
        payload = [{"role": m.role, "content": m.content or ""} for m in messages]
    return runtime.run(payload)


def _is_openwebui_followups_prompt(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return (
        "### Task:" in t
        and "follow_ups" in t
        and "### Chat History:" in t
        and "Suggest 3-5 relevant follow-up questions" in t
    )


MODEL_ID = "secretarius-agent"


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    request_id = f"req-{uuid4()}"
    message_roles = [m.role for m in request.messages] if isinstance(request.messages, list) else []
    total_chars = sum(len(m.content or "") for m in request.messages) if isinstance(request.messages, list) else 0
    LOGGER.info(
        "chat request_id=%s model=%s stream=%s user=%s message_count=%d total_chars=%d roles=%s",
        request_id,
        request.model,
        request.stream,
        request.user or "anonymous",
        len(request.messages) if isinstance(request.messages, list) else 0,
        total_chars,
        message_roles,
    )
    if not request.messages:
        LOGGER.warning("chat request_id=%s invalid no_messages", request_id)
        raise HTTPException(status_code=400, detail="No messages provided")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        LOGGER.warning("chat request_id=%s invalid no_user_message", request_id)
        raise HTTPException(status_code=400, detail="No user message found")

    prompt = user_messages[-1].content
    LOGGER.info("chat request_id=%s last_user_chars=%d", request_id, len(prompt or ""))

    # OpenWebUI secondary request for follow-up suggestions; do not send this to the agent.
    if _is_openwebui_followups_prompt(prompt):
        output_text = json.dumps({"follow_ups": []}, ensure_ascii=False)
        created_ts = int(datetime.now(timezone.utc).timestamp())
        completion_id = f"chatcmpl-{uuid4()}"
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        LOGGER.info("chat request_id=%s short_circuit=openwebui_followups", request_id)
        return response

    try:
        output_text = run_agent(request.messages)
    except Exception as exc:
        LOGGER.exception("chat request_id=%s agent_failed", request_id)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    created_ts = int(datetime.now(timezone.utc).timestamp())
    completion_id = f"chatcmpl-{uuid4()}"

    if request.stream:
        def event_stream():
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

            stop_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        LOGGER.info("chat request_id=%s stream_response output_len=%d", request_id, len(output_text))
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    LOGGER.info("chat request_id=%s response output_len=%d", request_id, len(output_text))
    return response


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.environ.get("SECRETARIUS_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("serverOpenAI:app", host=host, port=port, reload=False)
