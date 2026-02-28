#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from secretarius.memory_service import memory_add as do_memory_add
from secretarius.memory_service import memory_search as do_memory_search

app = FastAPI(title="Secretarius Memory API")
LOGGER = logging.getLogger("secretarius.memory_api")


class MemoryAddRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5


class MemorySearchRequest(BaseModel):
    text: Optional[str] = None
    expressions: Optional[List[str]] = None
    top_k: int = 5
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.post("/memory/add")
async def memory_add(request: MemoryAddRequest) -> Dict[str, Any]:
    request_id = f"memadd-{uuid4()}"
    LOGGER.info("memory.add request_id=%s text_len=%d", request_id, len(request.text or ""))
    try:
        payload = do_memory_add(text=request.text, metadata=request.metadata, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("memory.add request_id=%s failed", request_id)
        raise HTTPException(status_code=500, detail=f"memory add failed: {exc}") from exc
    payload["request_id"] = request_id
    return payload


@app.post("/memory/search")
async def memory_search(request: MemorySearchRequest) -> Dict[str, Any]:
    request_id = f"memsearch-{uuid4()}"
    LOGGER.info("memory.search request_id=%s text_len=%d", request_id, len((request.text or "").strip()))
    try:
        payload = do_memory_search(
            text=request.text,
            expressions=request.expressions,
            metadata=request.metadata,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("memory.search request_id=%s failed", request_id)
        raise HTTPException(status_code=500, detail=f"memory search failed: {exc}") from exc
    payload["request_id"] = request_id
    return payload


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.environ.get("SECRETARIUS_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    host = os.environ.get("MEMORY_HOST", "0.0.0.0")
    port = int(os.environ.get("MEMORY_PORT", "8011"))
    uvicorn.run("serverMemory:app", host=host, port=port, reload=False)

