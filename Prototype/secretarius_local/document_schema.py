from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "secretarius.document.v0.1"


def normalize_document(payload: dict[str, Any] | None) -> dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    raw_source = src.get("source")
    raw_content = src.get("content")
    source_dict = dict(raw_source) if isinstance(raw_source, dict) else {}
    content_dict = dict(raw_content) if isinstance(raw_content, dict) else {}
    doc: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "doc_id": src.get("doc_id"),
        "type": src.get("type"),
        "source": source_dict,
        "content": content_dict,
        "user_fields": dict(src.get("user_fields") or {}),
        "derived": dict(src.get("derived") or {}),
        "indexing": dict(src.get("indexing") or {}),
    }

    # Accept envelope shortcuts at root level.
    if isinstance(src.get("url"), str) and src["url"].strip():
        doc["source"].setdefault("url", src["url"].strip())
    if isinstance(src.get("urls"), list):
        source_urls = doc["source"].get("urls")
        if not isinstance(source_urls, list):
            source_urls = []
            doc["source"]["urls"] = source_urls
        for value in src["urls"]:
            if isinstance(value, str) and value.strip() and value.strip() not in source_urls:
                source_urls.append(value.strip())
    if isinstance(raw_content, str) and raw_content.strip():
        doc["content"].setdefault("text", raw_content)
    if isinstance(src.get("note"), str) and src["note"].strip():
        doc["content"].setdefault("text", src["note"])
    if isinstance(src.get("text"), str) and src["text"].strip():
        doc["content"].setdefault("text", src["text"])
    if isinstance(src.get("content_ref"), str) and src["content_ref"].strip():
        doc["content"].setdefault("content_ref", src["content_ref"])

    _set_default_type(doc)
    _ensure_content_defaults(doc)
    _ensure_user_fields_defaults(doc)
    _ensure_indexing_defaults(doc)
    _ensure_source_defaults(doc)
    _ensure_stable_ids(doc)
    return doc


def set_indexing_state(doc: dict[str, Any], state: str) -> None:
    indexing = doc.setdefault("indexing", {})
    if not isinstance(indexing, dict):
        indexing = {}
        doc["indexing"] = indexing
    indexing["state"] = state
    touch_updated_at(doc)


def add_indexing_error(doc: dict[str, Any], stage: str, message: str) -> None:
    indexing = doc.setdefault("indexing", {})
    if not isinstance(indexing, dict):
        indexing = {}
        doc["indexing"] = indexing
    errors = indexing.get("errors")
    if not isinstance(errors, list):
        errors = []
        indexing["errors"] = errors
    errors.append({"at": _now_iso(), "stage": stage, "message": message})
    indexing["state"] = "error"
    touch_updated_at(doc)


def touch_updated_at(doc: dict[str, Any]) -> None:
    user_fields = doc.setdefault("user_fields", {})
    if not isinstance(user_fields, dict):
        user_fields = {}
        doc["user_fields"] = user_fields
    user_fields["updated_at"] = _now_iso()
    if not user_fields.get("created_at"):
        user_fields["created_at"] = user_fields["updated_at"]


def resolve_text_for_extraction(arguments: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    arg_text = arguments.get("text")
    arg_text_str = arg_text.strip() if isinstance(arg_text, str) and arg_text.strip() else None
    raw_doc = arguments.get("document")
    if not isinstance(raw_doc, dict):
        return arg_text_str, None
    doc = normalize_document(raw_doc)

    doc_text_str: str | None = None
    content = doc.get("content")
    if isinstance(content, dict):
        content_text = content.get("text")
        if isinstance(content_text, str) and content_text.strip():
            doc_text_str = content_text.strip()

    if arg_text_str and doc_text_str:
        # Prefer the richer text to avoid truncated orchestrator payloads.
        if len(doc_text_str) >= len(arg_text_str):
            return doc_text_str, doc
        return arg_text_str, doc
    if doc_text_str:
        return doc_text_str, doc
    if arg_text_str:
        return arg_text_str, doc
    return None, doc


def expressions_from_document(doc: dict[str, Any]) -> list[str]:
    derived = doc.get("derived")
    if not isinstance(derived, dict):
        return []
    expressions = derived.get("expressions")
    if not isinstance(expressions, list):
        return []
    out: list[str] = []
    for entry in expressions:
        if isinstance(entry, str) and entry.strip():
            out.append(entry.strip())
            continue
        if isinstance(entry, dict):
            value = entry.get("expression")
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
    return out


def enrich_document_with_extraction(
    doc: dict[str, Any],
    *,
    chunks: list[str],
    by_chunk: list[dict[str, Any]],
    expressions: list[str],
) -> dict[str, Any]:
    derived = doc.setdefault("derived", {})
    if not isinstance(derived, dict):
        derived = {}
        doc["derived"] = derived

    chunk_entries: list[dict[str, Any]] = []
    cursor = 0
    text = ((doc.get("content") or {}).get("text") if isinstance(doc.get("content"), dict) else None) or ""
    for idx, chunk in enumerate(chunks):
        start = text.find(chunk, cursor) if isinstance(text, str) else -1
        if start < 0:
            start = max(cursor, 0)
        end = start + len(chunk)
        cursor = end
        chunk_entries.append(
            {
                "chunk_id": _stable_hash("chunk", f"{doc.get('doc_id')}|{idx}|{chunk[:256]}"),
                "start": start,
                "end": end,
                "text_ref": None,
            }
        )
    derived["chunks"] = chunk_entries

    expr_entries: list[dict[str, Any]] = []
    for expr in expressions:
        span = _find_span(text, expr) if isinstance(text, str) else [0, 0]
        expr_entries.append(
            {
                "expression": expr,
                "span": span,
                "weight": None,
                "norm": expr.lower(),
                "embedding_ref": None,
            }
        )
    derived["expressions"] = expr_entries
    set_indexing_state(doc, "embedding")
    return doc


def enrich_document_with_embeddings(
    doc: dict[str, Any],
    *,
    expressions: list[str],
    embeddings: list[list[float]],
) -> dict[str, Any]:
    derived = doc.setdefault("derived", {})
    if not isinstance(derived, dict):
        derived = {}
        doc["derived"] = derived

    expr_entries = derived.get("expressions")
    if not isinstance(expr_entries, list):
        expr_entries = []
        derived["expressions"] = expr_entries

    index_by_expr: dict[str, int] = {}
    for idx, entry in enumerate(expr_entries):
        if isinstance(entry, dict):
            value = entry.get("expression")
            if isinstance(value, str) and value not in index_by_expr:
                index_by_expr[value] = idx

    for idx, expr in enumerate(expressions):
        emb_ref = _stable_hash("emb", f"{doc.get('doc_id')}|{idx}|{expr}")
        slot = index_by_expr.get(expr)
        if slot is None:
            expr_entries.append(
                {
                    "expression": expr,
                    "span": [0, 0],
                    "weight": None,
                    "norm": expr.lower(),
                    "embedding_ref": emb_ref,
                }
            )
        else:
            entry = expr_entries[slot]
            if isinstance(entry, dict):
                entry["embedding_ref"] = emb_ref

    set_indexing_state(doc, "done")
    return doc


def _set_default_type(doc: dict[str, Any]) -> None:
    value = doc.get("type")
    if isinstance(value, str) and value.strip():
        doc["type"] = value.strip()
        return
    source = doc.get("source")
    content = doc.get("content")
    if isinstance(source, dict) and isinstance(source.get("url"), str) and source["url"].strip():
        doc["type"] = "url"
        return
    if isinstance(content, dict) and isinstance(content.get("text"), str) and content["text"].strip():
        doc["type"] = "note"
        return
    doc["type"] = "other"


def _ensure_content_defaults(doc: dict[str, Any]) -> None:
    content = doc.get("content")
    if not isinstance(content, dict):
        content = {}
        doc["content"] = content
    text = content.get("text")
    content_ref = content.get("content_ref")
    if isinstance(text, str) and text:
        if not content.get("hash"):
            content["hash"] = _stable_hash("sha256", text)
    elif not isinstance(content_ref, str) or not content_ref:
        content.pop("hash", None)


def _ensure_source_defaults(doc: dict[str, Any]) -> None:
    source = doc.get("source")
    if not isinstance(source, dict):
        source = {}
        doc["source"] = source
    urls = source.get("urls")
    normalized_urls: list[str] = []
    if isinstance(urls, list):
        for value in urls:
            if isinstance(value, str) and value.strip() and value.strip() not in normalized_urls:
                normalized_urls.append(value.strip())
    source["urls"] = normalized_urls
    if normalized_urls and (not isinstance(source.get("url"), str) or not source["url"].strip()):
        source["url"] = normalized_urls[0]
    if isinstance(source.get("url"), str) and source["url"].strip() and source["url"].strip() not in source["urls"]:
        source["urls"].insert(0, source["url"].strip())


def _ensure_user_fields_defaults(doc: dict[str, Any]) -> None:
    user_fields = doc.get("user_fields")
    if not isinstance(user_fields, dict):
        user_fields = {}
        doc["user_fields"] = user_fields
    if not isinstance(user_fields.get("keywords"), list):
        user_fields["keywords"] = []
    touch_updated_at(doc)


def _ensure_indexing_defaults(doc: dict[str, Any]) -> None:
    indexing = doc.get("indexing")
    if not isinstance(indexing, dict):
        indexing = {}
        doc["indexing"] = indexing
    indexing.setdefault("pipeline_version", "v0.1")
    indexing.setdefault("state", "new")
    if not isinstance(indexing.get("errors"), list):
        indexing["errors"] = []


def _ensure_stable_ids(doc: dict[str, Any]) -> None:
    source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
    if not doc.get("doc_id"):
        source_urls = source.get("urls") if isinstance(source.get("urls"), list) else []
        basis = "|".join(
            [
                str(source.get("canonical_url") or ""),
                str(source.get("url") or ""),
                ",".join(str(value) for value in source_urls),
                str(content.get("hash") or ""),
                str((content.get("text") or "")[:512]),
            ]
        )
        doc["doc_id"] = _stable_hash("doc", basis)
    if isinstance(source, dict) and not source.get("source_id"):
        urls = source.get("urls") if isinstance(source.get("urls"), list) else []
        url_basis = str(source.get("canonical_url") or source.get("url") or ",".join(str(value) for value in urls) or doc.get("doc_id") or "")
        source["source_id"] = _stable_hash("src", url_basis)


def _stable_hash(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _find_span(text: str, expression: str) -> list[int]:
    idx = text.find(expression)
    if idx < 0:
        return [0, 0]
    return [idx, idx + len(expression)]


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
