import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import time
import re
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
    model: str = "secretarius-agent"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: bool = False
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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

def _extract_openwebui_query(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # OpenWebUI may wrap user input as:
    # History: ...
    # Query: <actual user request>
    # Keep only Query to avoid passing prior chat history to the model.
    m = re.search(r"(?im)^\s*query\s*:\s*(.+)\Z", t, flags=re.DOTALL)
    if m:
        query = m.group(1).strip()
        return query.strip("\"' \n")
    return t

def _detect_openwebui_aux_task(prompt: str, metadata: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit metadata when OpenWebUI provides it.
    task = str((metadata or {}).get("task") or "").strip().lower()
    if task in {"follow_ups", "followups", "follow-up"}:
        return "follow_ups"
    if task in {"title", "title_generation", "chat_title"}:
        return "title"
    if task in {"tags", "tag", "tags_generation"}:
        return "tags"

    t = (prompt or "").strip()
    tl = t.lower()
    if not tl:
        return None

    if _is_openwebui_followups_prompt(t):
        return "follow_ups"

    # Heuristic detection for OpenWebUI helper prompts that should not hit the agent.
    has_task_block = ("### task:" in tl and "### chat history:" in tl) or ("chat history" in tl)
    asks_title = bool(re.search(r"\btitle\b", tl)) and any(k in tl for k in ("generate", "suggest", "concise"))
    asks_tags = bool(re.search(r"\btags?\b", tl)) and any(k in tl for k in ("generate", "suggest", "list"))
    if has_task_block and asks_title:
        return "title"
    if has_task_block and asks_tags:
        return "tags"
    return None

def _normalize_for_fingerprint(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", "\n", t)
    return t.strip()


def _prompt_fingerprint(text: str) -> str:
    normalized = _normalize_for_fingerprint(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


def create_openwebui_app(
    gateway: GuichetUnique,
    model_id: str = "secretarius-agent",
    request_timeout_s: float = 90.0,
) -> FastAPI:
    app = FastAPI(title="Secretarius OpenWebUI Channel")
    retry_window_s = 10.0
    recent_requests: dict[tuple[str, str], tuple[float, str]] = {}
    recent_requests_lock = asyncio.Lock()

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

        prompt = _extract_openwebui_query(user_messages[-1].content)
        request_id = f"req-{uuid4().hex[:12]}"
        channel = "openwebui"
        prompt_fp = _prompt_fingerprint(prompt)
        client_ip = http_request.client.host if http_request.client else "unknown"
        user_agent = (http_request.headers.get("user-agent") or "").strip()
        idem_key = (http_request.headers.get("x-idempotency-key") or "").strip()

        now = time.monotonic()
        retry_suspect = False
        previous_request_id = ""
        previous_age_s = 0.0
        request_key = (channel, prompt_fp)
        async with recent_requests_lock:
            previous = recent_requests.get(request_key)
            if previous is not None:
                previous_ts, previous_request_id = previous
                previous_age_s = now - previous_ts
                retry_suspect = previous_age_s <= retry_window_s
            recent_requests[request_key] = (now, request_id)
            # Prune stale entries to keep map bounded.
            stale_cutoff = now - (retry_window_s * 3)
            stale_keys = [k for k, (ts, _) in recent_requests.items() if ts < stale_cutoff]
            for key in stale_keys:
                del recent_requests[key]

        logger.info(
            "openwebui request_start request_id=%s client_ip=%s model=%s stream=%s prompt_fp=%s retry_suspect=%s previous_request_id=%s previous_age_s=%.3f idem_key=%s ua=%s",
            request_id,
            client_ip,
            request.model,
            request.stream,
            prompt_fp,
            retry_suspect,
            previous_request_id or "-",
            previous_age_s,
            idem_key or "-",
            user_agent or "-",
        )

        aux_task = _detect_openwebui_aux_task(prompt, request.metadata)
        if aux_task == "follow_ups":
            payload = _chat_completion_response(
                request.model,
                json.dumps({"follow_ups": []}, ensure_ascii=False),
            )
            payload["x_request_id"] = request_id
            logger.info("openwebui request_end request_id=%s aux_task=follow_ups", request_id)
            return payload
        if aux_task == "title":
            payload = _chat_completion_response(
                request.model,
                json.dumps({"title": "Conversation"}, ensure_ascii=False),
            )
            payload["x_request_id"] = request_id
            logger.info("openwebui request_end request_id=%s aux_task=title", request_id)
            return payload
        if aux_task == "tags":
            payload = _chat_completion_response(
                request.model,
                json.dumps({"tags": []}, ensure_ascii=False),
            )
            payload["x_request_id"] = request_id
            logger.info("openwebui request_end request_id=%s aux_task=tags", request_id)
            return payload

        try:
            logger.info("openwebui gateway_submit_start request_id=%s", request_id)
            output_text = await asyncio.wait_for(
                gateway.submit("openwebui", prompt),
                timeout=request_timeout_s,
            )
            logger.info("openwebui gateway_submit_end request_id=%s output_len=%d", request_id, len(output_text))
        except asyncio.TimeoutError as exc:
            logger.warning("openwebui request_timeout request_id=%s timeout_s=%.1f", request_id, request_timeout_s)
            raise HTTPException(status_code=504, detail="Agent timeout") from exc
        except Exception as exc:
            logger.exception("openwebui request_failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

        if request.stream:
            logger.info("openwebui request_end request_id=%s stream=true", request_id)
            return StreamingResponse(
                _stream_chunks(request.model, output_text),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id},
            )
        payload = _chat_completion_response(request.model, output_text)
        payload["x_request_id"] = request_id
        logger.info("openwebui request_end request_id=%s stream=false", request_id)
        return payload

    return app
