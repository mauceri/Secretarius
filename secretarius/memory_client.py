from __future__ import annotations

import os
from typing import Any

import requests


def _base_url() -> str:
    return os.environ.get("SECRETARIUS_MEMORY_API_URL", "http://127.0.0.1:8011").rstrip("/")


def memory_add(*, text: str, metadata: dict[str, Any] | None = None, top_k: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": text}
    if isinstance(metadata, dict):
        payload["metadata"] = metadata
    if isinstance(top_k, int):
        payload["top_k"] = top_k
    resp = requests.post(f"{_base_url()}/memory/add", json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Invalid memory/add response shape")
    return data


def memory_search(
    *,
    text: str | None = None,
    expressions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(text, str):
        payload["text"] = text
    if isinstance(expressions, list):
        payload["expressions"] = expressions
    if isinstance(metadata, dict):
        payload["metadata"] = metadata
    if isinstance(top_k, int):
        payload["top_k"] = top_k
    resp = requests.post(f"{_base_url()}/memory/search", json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Invalid memory/search response shape")
    return data

