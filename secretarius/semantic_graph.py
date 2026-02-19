from __future__ import annotations

import json
import time
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
            "query_count": len(normalized),
            "collection_name": collection_name,
            "metric_type": effective_metric,
            "warning": f"unable to ensure collection: {exc}",
        }

    inserted_count = 0
    if docs:
        try:
            inserted_count = _insert_documents(
                client=client,
                collection_name=collection_name,
                embeddings=normalized,
                documents=docs,
            )
        except Exception as exc:
            return {
                "graph": {"nodes": [], "edges": []},
                "hits": [],
                "inserted_count": 0,
                "query_count": len(normalized),
                "collection_name": collection_name,
                "metric_type": effective_metric,
                "warning": f"milvus insert failed: {exc}",
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
        return
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type=metric_type,
        consistency_level="Strong",
    )


def _insert_documents(
    *,
    client: Any,
    collection_name: str,
    embeddings: list[list[float]],
    documents: list[dict[str, Any]],
) -> int:
    count = min(len(embeddings), len(documents))
    if count == 0:
        return 0
    base_id = time.time_ns()
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        payload = documents[idx]
        if not isinstance(payload, dict):
            payload = {"value": str(payload)}
        rows.append(
            {
                "id": base_id + idx,
                "vector": embeddings[idx],
                "payload_json": json.dumps(payload, ensure_ascii=False),
                "source_idx": idx,
            }
        )
    client.insert(collection_name=collection_name, data=rows)
    return count


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
