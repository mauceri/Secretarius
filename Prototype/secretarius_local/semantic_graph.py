from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULT_MILVUS_URI = "http://127.0.0.1:19530"
DEFAULT_COLLECTION = "secretarius_semantic_graph"
DEFAULT_METRIC_TYPE = "COSINE"


def semantic_graph_search_milvus(
    embeddings: list[list[float]],
    *,
    documents: list[dict[str, Any]] | None = None,
    top_k: int = 10,
    uri: str = DEFAULT_MILVUS_URI,
    token: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    metric_type: str = DEFAULT_METRIC_TYPE,
) -> dict[str, Any]:
    normalized = _normalize_embeddings(embeddings)
    if not normalized:
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

    try:
        from pymilvus import MilvusClient
    except Exception as exc:
        return {
            "graph": {"nodes": [], "edges": []},
            "hits": [],
            "inserted_count": 0,
            "deleted_count": 0,
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": effective_metric,
            "warning": f"pymilvus import failed: {exc}",
        }

    try:
        client = MilvusClient(uri=uri, token=token)
    except Exception as exc:
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
        raw_hits = client.search(
            collection_name=collection_name,
            data=normalized,
            limit=top_k,
            output_fields=["payload_json", "source_idx"],
        )
    except Exception as exc:
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

    graph = _build_graph_from_hits(raw_hits)
    return {
        "graph": graph,
        "hits": raw_hits,
        "inserted_count": inserted_count,
        "deleted_count": deleted_count,
        "query_count": len(normalized),
        "collection_name": collection_name,
        "metric_type": effective_metric,
        "warning": None,
    }


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


def _resolve_metric_type(metric_type: str) -> str:
    mt = (metric_type or "").strip().upper()
    if mt == "COSINE":
        # Milvus v2.2 standalone accepts IP/L2; with normalized vectors, IP ~= cosine.
        return "IP"
    if mt in ("IP", "L2"):
        return mt
    return "IP"


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
