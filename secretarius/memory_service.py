from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

from .mcp_server import _handle_tool_call as _mcp_tool_call


def _tool_payload(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = _mcp_tool_call(name, arguments)
    payload = result.get("structuredContent")
    if isinstance(payload, dict):
        return payload
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                try:
                    loaded = json.loads(text)
                    if isinstance(loaded, dict):
                        return loaded
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
    return {}


def _default_collection() -> str:
    return os.environ.get("SECRETARIUS_MILVUS_COLLECTION", "secretarius_semantic_graph_384")


def _default_top_k() -> int:
    try:
        return int(os.environ.get("SECRETARIUS_MEMORY_TOP_K", "5"))
    except ValueError:
        return 5


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def _compact_hits(raw_hits: Any) -> dict[str, Any]:
    max_queries = _int_env("SECRETARIUS_MEMORY_MAX_QUERIES_RETURNED", 3)
    max_hits = _int_env("SECRETARIUS_MEMORY_MAX_HITS_PER_QUERY", 3)
    compact: list[list[dict[str, Any]]] = []
    total_hits = 0

    if not isinstance(raw_hits, list):
        return {"hits": compact, "total_hits": 0}

    for q_idx, hit_list in enumerate(raw_hits[:max_queries]):
        query_hits: list[dict[str, Any]] = []
        if isinstance(hit_list, list):
            total_hits += len(hit_list)
            for item in hit_list[:max_hits]:
                if not isinstance(item, dict):
                    continue
                entry: dict[str, Any] = {
                    "id": item.get("id"),
                    "distance": item.get("distance"),
                }
                entity = item.get("entity")
                if isinstance(entity, dict):
                    entry["source_idx"] = entity.get("source_idx")
                    payload_json = entity.get("payload_json")
                    if isinstance(payload_json, str):
                        try:
                            payload = json.loads(payload_json)
                            if isinstance(payload, dict):
                                entry["doc_id"] = payload.get("doc_id")
                                entry["type"] = payload.get("type")
                        except (TypeError, ValueError, json.JSONDecodeError):
                            pass
                query_hits.append(entry)
        compact.append(query_hits)

    return {"hits": compact, "total_hits": total_hits}


def _build_document(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    meta = metadata if isinstance(metadata, dict) else {}
    note_type = str(meta.get("type", "note"))
    doc_id = str(meta.get("doc_id", f"doc-{uuid4()}"))
    source = meta.get("source")
    source_obj = source if isinstance(source, dict) else {}
    user_fields: dict[str, Any] = {}
    for key in (
        "theme",
        "keywords",
        "tags",
        "status",
        "expires_at",
        "created_at",
        "updated_at",
        "confidence_user",
        "notes",
    ):
        if key in meta:
            user_fields[key] = meta[key]
    return {
        "schema": "secretarius.document.v0.1",
        "doc_id": doc_id,
        "type": note_type,
        "source": source_obj,
        "content": {"mode": "inline", "text": text},
        "user_fields": user_fields,
    }


def memory_add(*, text: str, metadata: dict[str, Any] | None = None, top_k: int | None = None) -> dict[str, Any]:
    text_value = (text or "").strip()
    if not text_value:
        raise ValueError("text is required")
    meta = metadata if isinstance(metadata, dict) else {}
    top_k_value = top_k if isinstance(top_k, int) and top_k > 0 else _default_top_k()
    collection_name = str(meta.get("collection_name", _default_collection()))
    document = _build_document(text_value, meta)

    extraction = _tool_payload(
        "extract_expressions",
        {"text": text_value, "document": document, "per_chunk_llm": False},
    )
    expressions = extraction.get("expressions", [])
    if not isinstance(expressions, list):
        expressions = []
    expressions = [x for x in expressions if isinstance(x, str)]
    if not expressions:
        return {
            "status": "ok",
            "doc_id": document.get("doc_id"),
            "expression_count": 0,
            "inserted_count": 0,
            "query_count": 0,
            "hits": [],
            "total_hits": 0,
            "warning": extraction.get("warning") or "no expressions extracted",
        }

    embeddings = _tool_payload(
        "expressions_to_embeddings",
        {"expressions": expressions, "document": extraction.get("document", document), "normalize": True},
    )
    vectors = embeddings.get("embeddings", [])
    if not isinstance(vectors, list):
        vectors = []

    semantic = _tool_payload(
        "semantic_graph_search",
        {
            "embeddings": vectors,
            "document": embeddings.get("document", extraction.get("document", document)),
            "upsert": True,
            "top_k": top_k_value,
            "collection_name": collection_name,
        },
    )
    compact = _compact_hits(semantic.get("hits", []))
    return {
        "status": "ok",
        "doc_id": document.get("doc_id"),
        "expression_count": len(expressions),
        "inserted_count": semantic.get("inserted_count", 0),
        "query_count": semantic.get("query_count", 0),
        "hits": compact["hits"],
        "total_hits": compact["total_hits"],
        "warning": semantic.get("warning") or embeddings.get("warning") or extraction.get("warning"),
    }


def memory_search(
    *,
    text: str | None = None,
    expressions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    meta = metadata if isinstance(metadata, dict) else {}
    exprs = expressions if isinstance(expressions, list) else []
    exprs = [x for x in exprs if isinstance(x, str) and x.strip()]
    text_value = (text or "").strip()
    if not exprs and not text_value:
        raise ValueError("Provide text or expressions")
    top_k_value = top_k if isinstance(top_k, int) and top_k > 0 else _default_top_k()
    collection_name = str(meta.get("collection_name", _default_collection()))

    extraction_warning = None
    if not exprs:
        extraction = _tool_payload(
            "extract_expressions",
            {"text": text_value, "per_chunk_llm": False},
        )
        extracted = extraction.get("expressions", [])
        if isinstance(extracted, list):
            exprs = [x for x in extracted if isinstance(x, str)]
        extraction_warning = extraction.get("warning")
        if not exprs:
            return {
                "status": "ok",
                "expressions": [],
                "expression_count": 0,
                "hits": [],
                "total_hits": 0,
                "graph": {"nodes": [], "edges": []},
                "query_count": 0,
                "warning": extraction_warning or "no expressions extracted",
            }

    semantic = _tool_payload(
        "semantic_graph_search",
        {
            "expressions": exprs,
            "upsert": False,
            "top_k": top_k_value,
            "collection_name": collection_name,
        },
    )
    compact = _compact_hits(semantic.get("hits", []))
    return {
        "status": "ok",
        "expressions": exprs,
        "expression_count": len(exprs),
        "hits": compact["hits"],
        "total_hits": compact["total_hits"],
        "graph": semantic.get("graph", {"nodes": [], "edges": []}),
        "query_count": semantic.get("query_count", 0),
        "warning": semantic.get("warning") or extraction_warning,
    }
