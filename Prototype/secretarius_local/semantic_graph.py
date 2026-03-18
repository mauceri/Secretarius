from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _configure_logger() -> None:
    log_path = (os.environ.get("SECRETARIUS_SEMANTIC_GRAPH_LOG") or "").strip()
    if not log_path:
        return

    target_path = os.path.abspath(log_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == target_path:
            return

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        handler = logging.FileHandler(target_path, encoding="utf-8")
    except OSError:
        return

    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_configure_logger()

DEFAULT_MILVUS_URI = "http://127.0.0.1:19530"
DEFAULT_COLLECTION = "secretarius_semantic_graph"
DEFAULT_METRIC_TYPE = "COSINE"


def semantic_graph_search_milvus(
    embeddings: list[list[float]],
    *,
    documents: list[dict[str, Any]] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    required_keywords: list[str] | None = None,
    optional_keywords: list[str] | None = None,
    excluded_keywords: list[str] | None = None,
    uri: str = DEFAULT_MILVUS_URI,
    token: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    metric_type: str = DEFAULT_METRIC_TYPE,
) -> dict[str, Any]:
    resolved_top_k = _resolve_top_k(top_k)
    normalized = _normalize_embeddings(embeddings)
    if not normalized:
        logger.warning("semantic_graph_search skipped: no valid embeddings")
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": 0,
            "deleted_count": 0,
            "query_count": 0,
            "collection_name": collection_name,
            "metric_type": metric_type,
            "warning": "no valid embeddings",
        }

    docs = documents if isinstance(documents, list) else []
    dim = len(normalized[0])
    if any(len(v) != dim for v in normalized):
        logger.warning("semantic_graph_search skipped: inconsistent embedding dimensions")
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": 0,
            "deleted_count": 0,
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": metric_type,
            "warning": "embedding dimensions are inconsistent",
        }

    effective_metric = _resolve_metric_type(metric_type)

    client_class = _resolve_milvus_client()
    if client_class is None:
        logger.warning("semantic_graph_search unavailable: pymilvus import failed")
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": 0,
            "deleted_count": 0,
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": effective_metric,
            "warning": "pymilvus import failed",
        }

    try:
        client = client_class(uri=uri, token=token)
    except Exception as exc:
        logger.exception("semantic_graph_search connection failed collection=%s uri=%s", collection_name, uri)
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": 0,
            "deleted_count": 0,
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": effective_metric,
            "warning": f"milvus connection failed: {exc}",
        }

    try:
        _ensure_collection(
            client=client,
            collection_name=collection_name,
            dim=dim,
            metric_type=effective_metric,
        )
    except Exception as exc:
        logger.exception(
            "semantic_graph_search unable to ensure collection=%s dim=%s metric=%s",
            collection_name,
            dim,
            effective_metric,
        )
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": 0,
            "deleted_count": 0,
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": effective_metric,
            "warning": f"unable to ensure collection: {exc}",
        }

    inserted_count = 0
    deleted_count = 0
    if docs:
        try:
            upsert_stats = _upsert_documents(
                client=client,
                collection_name=collection_name,
                embeddings=normalized,
                documents=docs,
            )
            inserted_count = upsert_stats["upserted_count"]
            deleted_count = upsert_stats["deleted_count"]
        except Exception as exc:
            logger.exception(
                "semantic_graph_search upsert failed collection=%s documents=%s",
                collection_name,
                len(docs),
            )
            return {
                "graph": {"nodes": [], "edges": []},
                "hits": [],
                "inserted_count": 0,
                "deleted_count": 0,
                "query_count": len(normalized),
                "collection_name": collection_name,
                "metric_type": effective_metric,
                "warning": f"milvus upsert failed: {exc}",
            }

    try:
        search_filter = _build_keywords_filter(
            required_keywords=required_keywords,
            optional_keywords=optional_keywords,
            excluded_keywords=excluded_keywords,
        )
        raw_hits = client.search(
            collection_name=collection_name,
            data=normalized,
            limit=resolved_top_k,
            output_fields=["payload_json", "source_idx"],
            filter=search_filter,
        )
    except Exception as exc:
        logger.exception(
            "semantic_graph_search failed collection=%s queries=%s top_k=%s filter=%r",
            collection_name,
            len(normalized),
            resolved_top_k,
            search_filter,
        )
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": inserted_count,
            "deleted_count": deleted_count,
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": effective_metric,
            "warning": f"milvus search failed: {exc}",
        }

    filtered_hits = _filter_hits_by_min_score(raw_hits, min_score=min_score)
    graph = _build_graph_from_hits(filtered_hits)
    logger.info(
        "semantic_graph_search completed collection=%s queries=%s top_k=%s inserted=%s deleted=%s",
        collection_name,
        len(normalized),
        resolved_top_k,
        inserted_count,
        deleted_count,
    )
    return {
        "graph": graph,
        "hits": filtered_hits,
        "inserted_count": inserted_count,
        "deleted_count": deleted_count,
        "query_count": len(normalized),
        "collection_name": collection_name,
        "metric_type": effective_metric,
        "min_score": min_score,
        "top_k": resolved_top_k,
        "warning": None,
    }


def semantic_graph_delete_doc(
    *,
    doc_id: str,
    uri: str = DEFAULT_MILVUS_URI,
    token: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    normalized_doc_id = doc_id.strip() if isinstance(doc_id, str) else ""
    if not normalized_doc_id:
        logger.warning("semantic_graph_delete_doc skipped: empty doc_id")
        return 0

    client_class = _resolve_milvus_client()
    if client_class is None:
        logger.warning("semantic_graph_delete_doc unavailable: pymilvus import failed")
        return 0

    try:
        client = client_class(uri=uri, token=token)
    except Exception:
        logger.exception("semantic_graph_delete_doc connection failed collection=%s uri=%s", collection_name, uri)
        return 0

    if not client.has_collection(collection_name=collection_name):
        logger.info("semantic_graph_delete_doc skipped: missing collection=%s", collection_name)
        return 0

    existing = client.query(
        collection_name=collection_name,
        filter=f'doc_id == "{_escape_filter_string(normalized_doc_id)}"',
        output_fields=["expression_id"],
    )
    ids_to_delete: list[int] = []
    if isinstance(existing, list):
        for row in existing:
            if isinstance(row, dict) and isinstance(row.get("id"), int):
                ids_to_delete.append(row["id"])
    if not ids_to_delete:
        logger.info("semantic_graph_delete_doc no-op: doc_id=%s collection=%s", normalized_doc_id, collection_name)
        return 0
    result = client.delete(collection_name=collection_name, ids=ids_to_delete)
    deleted_count = int(result.get("delete_count", len(ids_to_delete))) if isinstance(result, dict) else len(ids_to_delete)
    logger.info(
        "semantic_graph_delete_doc completed doc_id=%s collection=%s deleted=%s",
        normalized_doc_id,
        collection_name,
        deleted_count,
    )
    return deleted_count


def _normalize_embeddings(embeddings: Any) -> list[list[float]]:
    if not isinstance(embeddings, list):
        return []
    valid: list[list[float]] = []
    for vec in embeddings:
        if not isinstance(vec, list) or not vec:
            continue
        try:
            valid.append([float(v) for v in vec])
        except (TypeError, ValueError):
            continue
    return valid


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


def _resolve_metric_type(metric_type: str) -> str:
    mt = (metric_type or "").strip().upper()
    if mt == "COSINE":
        # Milvus v2.2 standalone accepts IP/L2; with normalized vectors, IP ~= cosine.
        return "IP"
    if mt in ("IP", "L2"):
        return mt
    return "IP"


def _resolve_milvus_client() -> Any | None:
    client_class = globals().get("MilvusClient")
    if client_class is not None:
        return client_class
    try:
        from pymilvus import MilvusClient as imported_client_class
    except Exception:
        return None
    return imported_client_class


def _ensure_collection(*, client: Any, collection_name: str, dim: int, metric_type: str) -> None:
    if client.has_collection(collection_name=collection_name):
        existing_dim = _get_collection_dimension(client=client, collection_name=collection_name)
        if existing_dim is not None and existing_dim != dim:
            raise ValueError(
                f"collection '{collection_name}' dimension mismatch: existing={existing_dim}, requested={dim}"
            )
        return
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type=metric_type,
        consistency_level="Strong",
    )


def _get_collection_dimension(*, client: Any, collection_name: str) -> int | None:
    try:
        details = client.describe_collection(collection_name=collection_name)
    except Exception:
        return None
    if not isinstance(details, dict):
        return None
    for key in ("dimension", "dim"):
        value = details.get(key)
        if isinstance(value, int):
            return value
    schema = details.get("schema")
    if not isinstance(schema, dict):
        return None
    fields = schema.get("fields")
    if not isinstance(fields, list):
        return None
    for field in fields:
        if not isinstance(field, dict):
            continue
        if str(field.get("type", "")).upper().find("VECTOR") < 0 and str(field.get("data_type", "")).upper().find("VECTOR") < 0:
            continue
        params = field.get("params")
        if isinstance(params, dict):
            dim = params.get("dim")
            if isinstance(dim, int):
                return dim
        dim = field.get("dim")
        if isinstance(dim, int):
            return dim
    return None


def _upsert_documents(
    *,
    client: Any,
    collection_name: str,
    embeddings: list[list[float]],
    documents: list[dict[str, Any]],
) -> dict[str, int]:
    count = min(len(embeddings), len(documents))
    if count == 0:
        return {"upserted_count": 0, "deleted_count": 0}
    rows: list[dict[str, Any]] = []
    desired_expression_ids_by_doc: dict[str, set[str]] = {}
    for idx in range(count):
        payload = documents[idx]
        if not isinstance(payload, dict):
            payload = {"value": str(payload)}
        row = _build_row(payload=payload, embedding=embeddings[idx], source_idx=idx)
        doc_id = row.get("doc_id")
        expression_id = row.get("expression_id")
        if isinstance(doc_id, str) and doc_id and isinstance(expression_id, str) and expression_id:
            desired_expression_ids_by_doc.setdefault(doc_id, set()).add(expression_id)
        rows.append(row)

    rows = _dedupe_rows_by_id(rows)
    deleted_count = 0
    for doc_id, desired_expression_ids in desired_expression_ids_by_doc.items():
        stale_ids = _find_stale_primary_ids(
            client=client,
            collection_name=collection_name,
            doc_id=doc_id,
            desired_expression_ids=desired_expression_ids,
        )
        if stale_ids:
            result = client.delete(collection_name=collection_name, ids=stale_ids)
            if isinstance(result, dict):
                deleted_count += int(result.get("delete_count", len(stale_ids)))
            else:
                deleted_count += len(stale_ids)

    client.upsert(collection_name=collection_name, data=rows)
    return {"upserted_count": len(rows), "deleted_count": deleted_count}


def _build_row(*, payload: dict[str, Any], embedding: list[float], source_idx: int) -> dict[str, Any]:
    doc_id = _extract_doc_id(payload, source_idx=source_idx)
    expression = _extract_source_expression(payload, source_idx=source_idx)
    expression_norm = _normalize_expression(expression, source_idx=source_idx)
    expression_id = _stable_key("expr", f"{doc_id}|{expression_norm}")
    user_fields = payload.get("user_fields") if isinstance(payload.get("user_fields"), dict) else {}
    return {
        "id": _stable_int_id(f"{doc_id}|{expression_norm}"),
        "vector": embedding,
        "payload_json": json.dumps(payload, ensure_ascii=False),
        "source_idx": source_idx,
        "doc_id": doc_id,
        "expression_id": expression_id,
        "expression_norm": expression_norm,
        "keywords": _extract_keywords(user_fields.get("keywords")),
        "type_note": user_fields.get("type_note") if isinstance(user_fields.get("type_note"), str) else "fugace",
    }


def _dedupe_rows_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, int):
            deduped[row_id] = row
    return list(deduped.values())


def _find_stale_primary_ids(
    *,
    client: Any,
    collection_name: str,
    doc_id: str,
    desired_expression_ids: set[str],
) -> list[int]:
    if not doc_id:
        return []
    existing = client.query(
        collection_name=collection_name,
        filter=f'doc_id == "{_escape_filter_string(doc_id)}"',
        output_fields=["expression_id"],
    )
    stale_ids: list[int] = []
    if not isinstance(existing, list):
        return stale_ids
    for row in existing:
        item = row if isinstance(row, dict) else {}
        primary_id = item.get("id")
        expression_id = item.get("expression_id")
        if isinstance(primary_id, int) and (
            not isinstance(expression_id, str) or expression_id not in desired_expression_ids
        ):
            stale_ids.append(primary_id)
    return stale_ids


def _extract_doc_id(payload: dict[str, Any], *, source_idx: int) -> str:
    raw_doc_id = payload.get("doc_id")
    if isinstance(raw_doc_id, str) and raw_doc_id.strip():
        return raw_doc_id.strip()
    return _stable_key("doc", f"{payload}|{source_idx}")


def _extract_source_expression(payload: dict[str, Any], *, source_idx: int) -> str:
    indexing = payload.get("indexing")
    if isinstance(indexing, dict):
        value = indexing.get("source_expression")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"source_idx:{source_idx}"


def _normalize_expression(expression: str, *, source_idx: int) -> str:
    raw = expression.strip().lower() if isinstance(expression, str) else ""
    if raw:
        return " ".join(raw.split())
    return f"source_idx:{source_idx}"


def _stable_int_id(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return max(1, int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1))


def _stable_key(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _escape_filter_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _extract_keywords(raw_keywords: Any) -> list[str]:
    if not isinstance(raw_keywords, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_keywords:
        if not isinstance(value, str):
            continue
        keyword = value.strip()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        out.append(keyword)
    return out


def _build_keywords_filter(
    *,
    required_keywords: list[str] | None = None,
    optional_keywords: list[str] | None = None,
    excluded_keywords: list[str] | None = None,
) -> str | None:
    clauses: list[str] = []

    required = _extract_keywords(required_keywords)
    optional = _extract_keywords(optional_keywords)
    excluded = _extract_keywords(excluded_keywords)

    if required:
        clauses.append(
            "(" + " and ".join(_keyword_contains_clause(keyword) for keyword in required) + ")"
        )
    if optional:
        clauses.append(
            "(" + " or ".join(_keyword_contains_clause(keyword) for keyword in optional) + ")"
        )
    if excluded:
        clauses.extend(f"(not {_keyword_contains_clause(keyword)})" for keyword in excluded)

    if not clauses:
        return None
    return " and ".join(clauses)


def _keyword_contains_clause(keyword: str) -> str:
    return f'array_contains(keywords, "{_escape_filter_string(keyword)}")'


def _build_graph_from_hits(raw_hits: Any) -> dict[str, list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    if not isinstance(raw_hits, list):
        return {"nodes": nodes, "edges": edges}

    for q_idx, hit_list in enumerate(raw_hits):
        q_node_id = f"query:{q_idx}"
        if q_node_id not in seen_nodes:
            seen_nodes.add(q_node_id)
            nodes.append({"id": q_node_id, "type": "query", "query_index": q_idx})

        if not isinstance(hit_list, list):
            continue
        for rank, hit in enumerate(hit_list):
            if not isinstance(hit, dict):
                continue
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit
            doc_id = hit.get("id", entity.get("id", f"rank:{q_idx}:{rank}"))
            doc_node_id = f"doc:{doc_id}"
            if doc_node_id not in seen_nodes:
                seen_nodes.add(doc_node_id)
                payload_json = entity.get("payload_json")
                payload = None
                if isinstance(payload_json, str):
                    try:
                        payload = json.loads(payload_json)
                    except json.JSONDecodeError:
                        payload = payload_json
                nodes.append(
                    {
                        "id": doc_node_id,
                        "type": "document",
                        "payload": payload,
                        "source_idx": entity.get("source_idx"),
                    }
                )

            score = hit.get("score", entity.get("score", hit.get("distance", entity.get("distance"))))
            edges.append(
                {
                    "source": q_node_id,
                    "target": doc_node_id,
                    "rank": rank,
                    "score": score,
                }
            )

    return {"nodes": nodes, "edges": edges}


def _filter_hits_by_min_score(raw_hits: Any, *, min_score: float | None) -> list[list[dict[str, Any]]]:
    if not isinstance(raw_hits, list):
        return []
    if min_score is None:
        return [hit_list if isinstance(hit_list, list) else [] for hit_list in raw_hits]

    threshold = float(min_score)
    filtered: list[list[dict[str, Any]]] = []
    for hit_list in raw_hits:
        if not isinstance(hit_list, list):
            filtered.append([])
            continue
        kept: list[dict[str, Any]] = []
        for hit in hit_list:
            if not isinstance(hit, dict):
                continue
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit
            score = hit.get("score", entity.get("score", hit.get("distance", entity.get("distance"))))
            if isinstance(score, (int, float)) and float(score) >= threshold:
                kept.append(hit)
        filtered.append(kept)
    return filtered
