"""
Pipeline RAG BGE-M3 multi-vector.

Trois opérations : index, search, update.

Stockage Milvus : 1 ligne par token, toutes les lignes d'un document
partagent le même doc_id. La recherche agrège par MaxSim (ColBERT).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np

from .embeddings import encode_multivector

logger = logging.getLogger(__name__)

MILVUS_URI        = os.environ.get("SECRETARIUS_MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION_NAME   = os.environ.get("SECRETARIUS_COLLECTION", "secretarius_rag")
VECTOR_DIM        = 1024
DEFAULT_TOP_K     = 10


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _doc_id_from_text(text: str) -> str:
    return "doc:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _stable_id(doc_id: str, token_idx: int) -> int:
    key = f"{doc_id}|{token_idx}"
    digest = hashlib.sha256(key.encode()).digest()
    return max(1, int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1))


def _get_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI)


def _ensure_collection(client: Any) -> None:
    if client.has_collection(COLLECTION_NAME):
        return
    from pymilvus import MilvusClient, DataType
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",          DataType.INT64,   is_primary=True)
    schema.add_field("vector",      DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("doc_id",      DataType.VARCHAR, max_length=64)
    schema.add_field("token_idx",   DataType.INT64)
    schema.add_field("payload_json",DataType.VARCHAR, max_length=4096)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="IP",
                           index_type="HNSW", params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="doc_id", index_type="Trie")

    client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
    logger.info("Collection créée : %s", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Indexation
# ---------------------------------------------------------------------------

def index(text: str, *, doc_id: str | None = None) -> dict[str, Any]:
    """
    Indexe un document texte.

    Si doc_id est fourni, il est utilisé ; sinon un hash SHA256 est calculé.
    Retourne {"doc_id": ..., "n_tokens": ..., "warning": None|str}.
    """
    if not text or not text.strip():
        return {"doc_id": None, "n_tokens": 0, "warning": "texte vide"}

    resolved_doc_id = (doc_id or "").strip() or _doc_id_from_text(text)

    try:
        client = _get_client()
        _ensure_collection(client)
    except Exception as exc:
        return {"doc_id": resolved_doc_id, "n_tokens": 0, "warning": f"Milvus indisponible : {exc}"}

    # Supprimer l'ancienne version si elle existe
    _delete_doc(client, resolved_doc_id)

    # Encoder
    token_vecs = encode_multivector([text])[0]  # (n_tokens, 1024)

    # Payload stocké sur chaque ligne (compact pour notes courtes)
    payload = json.dumps({"doc_id": resolved_doc_id, "text": text[:2000]}, ensure_ascii=False)

    rows = [
        {
            "id":           _stable_id(resolved_doc_id, i),
            "vector":       token_vecs[i].tolist(),
            "doc_id":       resolved_doc_id,
            "token_idx":    i,
            "payload_json": payload if i == 0 else "{}",
        }
        for i in range(len(token_vecs))
    ]

    try:
        client.insert(COLLECTION_NAME, rows)
        client.flush(collection_name=COLLECTION_NAME)
    except Exception as exc:
        return {"doc_id": resolved_doc_id, "n_tokens": 0, "warning": f"Erreur insertion : {exc}"}

    return {"doc_id": resolved_doc_id, "n_tokens": len(token_vecs), "warning": None}


# ---------------------------------------------------------------------------
# Recherche
# ---------------------------------------------------------------------------

def search(query: str, *, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """
    Recherche par similarité MaxSim (ColBERT).

    Retourne {"results": [{"doc_id", "text", "score"}, ...], "warning": None|str}.
    """
    if not query or not query.strip():
        return {"results": [], "warning": "requête vide"}

    try:
        client = _get_client()
        if not client.has_collection(COLLECTION_NAME):
            return {"results": [], "warning": "index vide — aucun document indexé"}
    except Exception as exc:
        return {"results": [], "warning": f"Milvus indisponible : {exc}"}

    query_vecs = encode_multivector([query])[0]  # (n_q_tokens, 1024)

    # Rechercher top_k*5 candidats par vecteur requête pour avoir assez de docs distincts
    candidate_limit = top_k * 5

    # max_scores[doc_id][q_token_idx] = meilleur score pour ce token de requête
    max_scores: dict[str, list[float]] = {}
    n_q = len(query_vecs)

    for q_idx in range(n_q):
        try:
            hits = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vecs[q_idx].tolist()],
                limit=candidate_limit,
                output_fields=["doc_id", "payload_json", "token_idx"],
            )[0]
        except Exception as exc:
            return {"results": [], "warning": f"Erreur recherche : {exc}"}

        for hit in hits:
            doc_id = hit["entity"]["doc_id"]
            score  = float(hit["distance"])
            if doc_id not in max_scores:
                max_scores[doc_id] = [0.0] * n_q
            if score > max_scores[doc_id][q_idx]:
                max_scores[doc_id][q_idx] = score

    if not max_scores:
        return {"results": [], "warning": None}

    # Score ColBERT = somme des maxima par token requête
    colbert_scores = {doc_id: sum(scores) for doc_id, scores in max_scores.items()}
    ranked = sorted(colbert_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Récupérer les payloads des documents retenus
    results = []
    for doc_id, score in ranked:
        try:
            rows = client.query(
                collection_name=COLLECTION_NAME,
                filter=f'doc_id == "{_escape(doc_id)}" and token_idx == 0',
                output_fields=["payload_json"],
                limit=1,
            )
            text = ""
            if rows:
                payload = json.loads(rows[0].get("payload_json", "{}"))
                text = payload.get("text", "")
        except Exception:
            text = ""
        results.append({"doc_id": doc_id, "text": text, "score": round(score, 4)})

    return {"results": results, "warning": None}


# ---------------------------------------------------------------------------
# Mise à jour
# ---------------------------------------------------------------------------

def update(doc_id: str, text: str) -> dict[str, Any]:
    """Met à jour un document existant (supprime + réindexe)."""
    if not doc_id or not doc_id.strip():
        return {"doc_id": None, "n_tokens": 0, "warning": "doc_id vide"}
    return index(text, doc_id=doc_id.strip())


# ---------------------------------------------------------------------------
# Suppression (interne)
# ---------------------------------------------------------------------------

def _delete_doc(client: Any, doc_id: str) -> int:
    try:
        if not client.has_collection(COLLECTION_NAME):
            return 0
        existing = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{_escape(doc_id)}"',
            output_fields=["id"],
            limit=16384,
        )
        ids = [row["id"] for row in existing if isinstance(row.get("id"), int)]
        if ids:
            client.delete(COLLECTION_NAME, ids=ids)
        return len(ids)
    except Exception as exc:
        logger.warning("Erreur suppression doc_id=%s : %s", doc_id, exc)
        return 0


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
