from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any

from .document_schema import (
    add_indexing_error,
    enrich_document_with_embeddings,
    enrich_document_with_extraction,
    normalize_document,
    set_indexing_state,
)
from .embeddings import embed_expressions_multilingual
from .expression_extractor import extract_expressions
from .semantic_graph import semantic_graph_search_milvus
from .semantic_graph import semantic_graph_delete_doc


_QUERY_KEYWORD_RE = re.compile(r"(?<!\w)(?P<op>[+-]?)(?P<keyword>#[^\s#]+)")
logger = logging.getLogger(__name__)


def analyse_texte_documentaire(text: str, *, base_document: dict[str, Any] | None = None) -> dict[str, Any]:
    hashtag_re = re.compile(r"(?<!\w)#([^\s#]+)")
    date_re = re.compile(r"\b\d{2}-\d{2}-\d{4}\b")
    url_re = re.compile(r'\bhttps?://[^\s<>\"]+')
    doc_id_re = re.compile(r"(?mi)^[ \t]*doc_id[ \t]*:[ \t]*([^\n\r]+?)\s*$")
    type_note_re = re.compile(r"(?mi)^[ \t]*type_note[ \t]*:[ \t]*([^\n\r]+?)\s*$")
    allowed_type_notes = {"fugace", "lecture", "permanente"}

    def normalize(raw: str) -> str:
        cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def remove_spans(raw: str, spans: list[tuple[int, int]]) -> str:
        if not spans:
            return raw
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        out: list[str] = []
        cursor = 0
        for start, end in merged:
            out.append(raw[cursor:start])
            cursor = end
        out.append(raw[cursor:])
        return "".join(out)

    def compute_incipit_title(body: str, max_len: int = 40) -> str | None:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return None
        first = lines[0]
        if len(first) <= max_len:
            return first
        cut = first[:max_len]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut.rstrip(" ,;:-") + "..."

    def detect_explicit_title(lines: list[str]) -> tuple[str | None, int | None]:
        for idx in range(len(lines) - 1):
            current = lines[idx].strip()
            if not current:
                continue
            if hashtag_re.fullmatch(current):
                continue
            if date_re.fullmatch(current):
                continue
            if url_re.fullmatch(current):
                continue

            next_line = ""
            for look_ahead in range(idx + 1, len(lines)):
                candidate = lines[look_ahead].strip()
                if not candidate:
                    continue
                next_line = candidate
                break
            if not next_line:
                continue
            if len(current) <= 120 and not current.endswith((".", "!", "?", ";")) and len(next_line) >= len(current):
                return current, idx
        return None, None

    normalized = normalize(text or "")
    spans_to_remove: list[tuple[int, int]] = []
    keywords: list[str] = []

    for match in hashtag_re.finditer(normalized):
        keywords.append(f"#{match.group(1)}")
        spans_to_remove.append((match.start(), match.end()))

    date_match = date_re.search(normalized)
    date_value = date_match.group(0) if date_match else None
    if date_match:
        spans_to_remove.append((date_match.start(), date_match.end()))

    doc_id_match = doc_id_re.search(normalized)
    doc_id_value = doc_id_match.group(1).strip() if doc_id_match else None
    if doc_id_match and doc_id_value:
        spans_to_remove.append((doc_id_match.start(), doc_id_match.end()))

    type_note_match = type_note_re.search(normalized)
    type_note_value = "fugace"
    if type_note_match:
        candidate = type_note_match.group(1).strip().lower()
        if candidate in allowed_type_notes:
            type_note_value = candidate
            spans_to_remove.append((type_note_match.start(), type_note_match.end()))

    url_matches = list(url_re.finditer(normalized))
    url_values: list[str] = []
    for match in url_matches:
        value = match.group(0).strip()
        if value and value not in url_values:
            url_values.append(value)
        spans_to_remove.append((match.start(), match.end()))

    deduped_keywords: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        deduped_keywords.append(keyword)

    cleaned = remove_spans(normalized, spans_to_remove)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip(" \n\t-:")

    lines = cleaned.splitlines()
    title, title_idx = detect_explicit_title(lines)
    if title is not None and title_idx is not None:
        body = "\n".join(line for idx, line in enumerate(lines) if idx != title_idx).strip()
    else:
        body = cleaned.strip()
        title = compute_incipit_title(body)

    document_payload = copy.deepcopy(base_document) if isinstance(base_document, dict) else {}
    if isinstance(doc_id_value, str) and doc_id_value:
        document_payload["doc_id"] = doc_id_value
    content = document_payload.setdefault("content", {})
    if isinstance(content, dict):
        content["text"] = normalize(body)
    source = document_payload.setdefault("source", {})
    if isinstance(source, dict):
        existing_urls = source.get("urls")
        merged_urls: list[str] = []
        if isinstance(existing_urls, list):
            for value in existing_urls:
                if isinstance(value, str) and value.strip() and value.strip() not in merged_urls:
                    merged_urls.append(value.strip())
        for value in url_values:
            if value not in merged_urls:
                merged_urls.append(value)
        if merged_urls:
            source["urls"] = merged_urls
            if not isinstance(source.get("url"), str) or not source.get("url", "").strip():
                source["url"] = merged_urls[0]
    user_fields = document_payload.setdefault("user_fields", {})
    if isinstance(user_fields, dict):
        if not isinstance(user_fields.get("type_note"), str) or not user_fields.get("type_note", "").strip():
            user_fields["type_note"] = type_note_value
        if isinstance(title, str) and title.strip() and not user_fields.get("title"):
            user_fields["title"] = title.strip()
        if isinstance(date_value, str) and date_value.strip() and not user_fields.get("document_date"):
            user_fields["document_date"] = date_value.strip()
        if deduped_keywords and not user_fields.get("keywords"):
            user_fields["keywords"] = deduped_keywords
    return normalize_document(document_payload)


def index_document_text(
    text: str,
    *,
    base_document: dict[str, Any] | None = None,
    llama_cpp_url: str = "http://127.0.0.1:8989/v1/chat/completions",
    llama_cpp_model: str = "local-llama-cpp",
    timeout_s: float = 30.0,
    max_tokens: int = 20480,
    seed: int = 42,
    prompt_path: str | None = None,
    debug_return_raw: bool = False,
    embedding_model: str | None = None,
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    milvus_uri: str = "http://127.0.0.1:19530",
    milvus_token: str | None = None,
    collection_name: str = "secretarius_semantic_graph",
    metric_type: str = "COSINE",
    top_k: int | None = None,
) -> dict[str, Any]:
    resolved_top_k = _resolve_top_k(top_k)
    document = analyse_texte_documentaire(text, base_document=base_document)
    content = document.get("content") if isinstance(document.get("content"), dict) else {}
    document_text = content.get("text") if isinstance(content.get("text"), str) else ""
    if not document_text.strip():
        add_indexing_error(document, "extracting", "document has no inline text")
        return {
            "document": document,
            "extract": {"expressions": [], "warning": "document has no inline text"},
            "index": {
                "collection_name": collection_name,
                "inserted_count": 0,
                "query_count": 0,
                "hits": [],
                "warning": "document has no inline text",
            },
        }

    set_indexing_state(document, "extracting")
    extract_result = extract_expressions(
        document_text,
        llama_cpp_url=llama_cpp_url,
        llama_cpp_model=llama_cpp_model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        prompt_path=prompt_path,
        seed=seed,
        debug_return_raw=debug_return_raw,
    )
    expressions = extract_result.get("expressions", [])
    if not isinstance(expressions, list):
        expressions = []
    expressions = [expr for expr in expressions if isinstance(expr, str)]

    document = enrich_document_with_extraction(
        document,
        chunks=extract_result.get("chunks", []) if isinstance(extract_result.get("chunks"), list) else [],
        by_chunk=extract_result.get("by_chunk", []) if isinstance(extract_result.get("by_chunk"), list) else [],
        expressions=expressions,
    )

    if not expressions:
        warning = extract_result.get("warning") or "no expressions extracted"
        add_indexing_error(document, "extracting", str(warning))
        return {
            "document": document,
            "extract": extract_result,
            "index": {
                "collection_name": collection_name,
                "inserted_count": 0,
                "query_count": 0,
                "hits": [],
                "warning": str(warning),
            },
        }

    embedding_result = embed_expressions_multilingual(
        expressions,
        model=embedding_model,
        normalize=normalize_embeddings,
        batch_size=batch_size,
    )
    embeddings = embedding_result.get("embeddings", [])
    if not isinstance(embeddings, list):
        embeddings = []
    if not embeddings:
        warning = str(embedding_result.get("warning") or "no embeddings generated")
        logger.warning("index_document_text skipped semantic graph: %s", warning)
        add_indexing_error(document, "embedding", warning)
        return {
            "document": document,
            "extract": extract_result,
            "index": {
                "collection_name": collection_name,
                "inserted_count": 0,
                "query_count": 0,
                "hits": [],
                "warning": warning,
            },
        }
    document = enrich_document_with_embeddings(
        document,
        expressions=expressions,
        embeddings=embeddings,
    )

    documents = _documents_for_expression_embeddings(document, expressions)
    set_indexing_state(document, "upserting")
    index_result = semantic_graph_search_milvus(
        embeddings,
        documents=documents,
        top_k=resolved_top_k,
        uri=milvus_uri,
        token=milvus_token,
        collection_name=collection_name,
        metric_type=metric_type,
    )
    warning = index_result.get("warning")
    if warning:
        add_indexing_error(document, "upserting", str(warning))
    else:
        set_indexing_state(document, "done")
    return {
        "document": document,
        "extract": extract_result,
        "index": index_result,
    }


def update_document_text(
    text: str,
    *,
    base_document: dict[str, Any] | None = None,
    llama_cpp_url: str = "http://127.0.0.1:8989/v1/chat/completions",
    llama_cpp_model: str = "local-llama-cpp",
    timeout_s: float = 30.0,
    max_tokens: int = 20480,
    seed: int = 42,
    prompt_path: str | None = None,
    debug_return_raw: bool = False,
    embedding_model: str | None = None,
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    milvus_uri: str = "http://127.0.0.1:19530",
    milvus_token: str | None = None,
    collection_name: str = "secretarius_semantic_graph",
    metric_type: str = "COSINE",
    top_k: int | None = None,
) -> dict[str, Any]:
    resolved_top_k = _resolve_top_k(top_k)
    document = analyse_texte_documentaire(text, base_document=base_document)
    doc_id = document.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("'doc_id' must be provided for update")

    content = document.get("content") if isinstance(document.get("content"), dict) else {}
    document_text = content.get("text") if isinstance(content.get("text"), str) else ""

    deleted_count = semantic_graph_delete_doc(
        doc_id=doc_id.strip(),
        uri=milvus_uri,
        token=milvus_token,
        collection_name=collection_name,
    )

    if not document_text.strip():
        add_indexing_error(document, "extracting", "document has no inline text")
        return {
            "document": document,
            "extract": {"expressions": [], "warning": "document has no inline text"},
            "index": {
                "collection_name": collection_name,
                "inserted_count": 0,
                "deleted_count": deleted_count,
                "query_count": 0,
                "hits": [],
                "warning": "document has no inline text",
            },
        }

    set_indexing_state(document, "extracting")
    extract_result = extract_expressions(
        document_text,
        llama_cpp_url=llama_cpp_url,
        llama_cpp_model=llama_cpp_model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        prompt_path=prompt_path,
        seed=seed,
        debug_return_raw=debug_return_raw,
    )
    expressions = extract_result.get("expressions", [])
    if not isinstance(expressions, list):
        expressions = []
    expressions = [expr for expr in expressions if isinstance(expr, str)]

    document = enrich_document_with_extraction(
        document,
        chunks=extract_result.get("chunks", []) if isinstance(extract_result.get("chunks"), list) else [],
        by_chunk=extract_result.get("by_chunk", []) if isinstance(extract_result.get("by_chunk"), list) else [],
        expressions=expressions,
    )

    if not expressions:
        warning = extract_result.get("warning") or "no expressions extracted"
        add_indexing_error(document, "extracting", str(warning))
        return {
            "document": document,
            "extract": extract_result,
            "index": {
                "collection_name": collection_name,
                "inserted_count": 0,
                "deleted_count": deleted_count,
                "query_count": 0,
                "hits": [],
                "warning": str(warning),
            },
        }

    embedding_result = embed_expressions_multilingual(
        expressions,
        model=embedding_model,
        normalize=normalize_embeddings,
        batch_size=batch_size,
    )
    embeddings = embedding_result.get("embeddings", [])
    if not isinstance(embeddings, list):
        embeddings = []
    if not embeddings:
        warning = str(embedding_result.get("warning") or "no embeddings generated")
        logger.warning("update_document_text skipped semantic graph: %s", warning)
        add_indexing_error(document, "embedding", warning)
        return {
            "document": document,
            "extract": extract_result,
            "index": {
                "collection_name": collection_name,
                "inserted_count": 0,
                "deleted_count": deleted_count,
                "query_count": 0,
                "hits": [],
                "warning": warning,
            },
        }
    document = enrich_document_with_embeddings(
        document,
        expressions=expressions,
        embeddings=embeddings,
    )

    documents = _documents_for_expression_embeddings(document, expressions)
    set_indexing_state(document, "upserting")
    index_result = semantic_graph_search_milvus(
        embeddings,
        documents=documents,
        top_k=resolved_top_k,
        uri=milvus_uri,
        token=milvus_token,
        collection_name=collection_name,
        metric_type=metric_type,
    )
    index_result["deleted_count"] = index_result.get("deleted_count", 0) + deleted_count
    warning = index_result.get("warning")
    if warning:
        add_indexing_error(document, "upserting", str(warning))
    else:
        set_indexing_state(document, "done")
    return {
        "document": document,
        "extract": extract_result,
        "index": index_result,
    }


def search_documents_by_text(
    text: str,
    *,
    llama_cpp_url: str = "http://127.0.0.1:8989/v1/chat/completions",
    llama_cpp_model: str = "local-llama-cpp",
    timeout_s: float = 30.0,
    max_tokens: int = 20480,
    seed: int = 42,
    prompt_path: str | None = None,
    debug_return_raw: bool = False,
    embedding_model: str | None = None,
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    milvus_uri: str = "http://127.0.0.1:19530",
    milvus_token: str | None = None,
    collection_name: str = "secretarius_semantic_graph",
    metric_type: str = "COSINE",
    top_k: int | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    resolved_top_k = _resolve_top_k(top_k)
    query_document = analyse_texte_documentaire(text)
    parsed_query = _parse_keyword_query(text)
    query_text = parsed_query["text"]

    content = query_document.get("content")
    if isinstance(content, dict):
        content["text"] = query_text

    extract_result = extract_expressions(
        query_text,
        llama_cpp_url=llama_cpp_url,
        llama_cpp_model=llama_cpp_model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        prompt_path=prompt_path,
        seed=seed,
        debug_return_raw=debug_return_raw,
    )
    expressions = extract_result.get("expressions", [])
    if not isinstance(expressions, list):
        expressions = []
    expressions = [expr for expr in expressions if isinstance(expr, str)]

    if not expressions:
        return {
            "document": query_document,
            "extract": extract_result,
            "search": {
                "collection_name": collection_name,
                "query_count": 0,
                "hits": [],
                "warning": extract_result.get("warning") or "no expressions extracted",
            },
        }

    embedding_result = embed_expressions_multilingual(
        expressions,
        model=embedding_model,
        normalize=normalize_embeddings,
        batch_size=batch_size,
    )
    embeddings = embedding_result.get("embeddings", [])
    if not isinstance(embeddings, list):
        embeddings = []
    if not embeddings:
        warning = str(embedding_result.get("warning") or "no embeddings generated")
        logger.warning("search_documents_by_text skipped semantic graph: %s", warning)
        return {
            "document": query_document,
            "extract": extract_result,
            "search": {
                "collection_name": collection_name,
                "query_count": 0,
                "hits": [],
                "warning": warning,
            },
        }

    search_result = semantic_graph_search_milvus(
        embeddings,
        documents=[],
        top_k=resolved_top_k,
        required_keywords=parsed_query["required_keywords"],
        optional_keywords=parsed_query["optional_keywords"],
        excluded_keywords=parsed_query["excluded_keywords"],
        uri=milvus_uri,
        token=milvus_token,
        collection_name=collection_name,
        metric_type=metric_type,
        min_score=min_score,
    )
    return {
        "document": query_document,
        "extract": extract_result,
        "search": search_result,
    }


def _resolve_top_k(top_k: Any) -> int:
    if isinstance(top_k, int) and top_k >= 1:
        return top_k
    raw = (os.environ.get("SECRETARIUS_MILVUS_TOP_K") or "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed >= 1:
            return parsed
    return 10


def _documents_for_expression_embeddings(document: dict[str, Any], expressions: list[str]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for idx, expression in enumerate(expressions):
        doc_copy = copy.deepcopy(document)
        indexing = doc_copy.setdefault("indexing", {})
        if isinstance(indexing, dict):
            indexing["source"] = "document_pipeline:index_text"
            indexing["source_expression_idx"] = idx
            indexing["source_expression"] = expression
        documents.append(doc_copy)
    return documents


def _parse_keyword_query(text: str) -> dict[str, Any]:
    raw_text = text if isinstance(text, str) else ""
    optional_keywords: list[str] = []
    required_keywords: list[str] = []
    excluded_keywords: list[str] = []

    def replace_keyword(match: re.Match[str]) -> str:
        keyword = _normalize_keyword(match.group("keyword"))
        if not keyword:
            return " "
        operator = match.group("op")
        if operator == "+":
            if keyword not in required_keywords:
                required_keywords.append(keyword)
        elif operator == "-":
            if keyword not in excluded_keywords:
                excluded_keywords.append(keyword)
        elif keyword not in optional_keywords:
            optional_keywords.append(keyword)
        return " "

    cleaned = _QUERY_KEYWORD_RE.sub(replace_keyword, raw_text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip(" \n\t-+")
    return {
        "text": cleaned,
        "optional_keywords": optional_keywords,
        "required_keywords": required_keywords,
        "excluded_keywords": excluded_keywords,
    }


def _normalize_keyword(raw_keyword: str) -> str:
    if not isinstance(raw_keyword, str):
        return ""
    keyword = raw_keyword.strip()
    if not keyword:
        return ""
    if not keyword.startswith("#"):
        keyword = f"#{keyword.lstrip('#')}"
    return keyword
