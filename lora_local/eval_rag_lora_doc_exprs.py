#!/usr/bin/env python3
"""
Expériences 4 et 5 — RAG classique vs indexation par expressions LoRA (côté document).

Même protocole que les expériences 1 et 2, mais les expressions utilisées
pour l'index (conditions B, C, D) sont extraites par LoRA depuis le texte
des chunks — et non les expressions DeepSeek pré-calculées du corpus.

Objectif : mesurer l'impact de la source des expressions documentaires
(DeepSeek corpus vs LoRA à la volée) sur la qualité de retrieval.

Expérience 4 : requêtes depuis texte brut (cache exp 1), expressions requête LoRA
Expérience 5 : requêtes depuis expressions (cache exp 2), expressions requête LoRA

Usage :
    source llenv/bin/activate
    source ~/.config/secrets.env

    # Expérience 4 (requêtes texte brut)
    python eval_rag_lora_doc_exprs.py \\
        --queries-cache eval_rag_queries.json \\
        --query-exprs-cache eval_rag_query_exprs.json \\
        --out eval_rag4_results.json

    # Expérience 5 (requêtes depuis expressions)
    python eval_rag_lora_doc_exprs.py \\
        --queries-cache eval_rag2_queries.json \\
        --query-exprs-cache eval_rag2_query_exprs.json \\
        --out eval_rag5_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib import request as urlrequest

import numpy as np

# ---------------------------------------------------------------------------
# Embedding BGE-M3 en CPU
# ---------------------------------------------------------------------------
_SECRETARIUS_LOCAL = Path(__file__).resolve().parents[1] / "Prototype" / "secretarius_local"

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    from sentence_transformers import SentenceTransformer
    if str(_SECRETARIUS_LOCAL) not in sys.path:
        sys.path.insert(0, str(_SECRETARIUS_LOCAL))
    from runtime_paths import resolve_sentence_model_path  # type: ignore
    model_path = resolve_sentence_model_path()
    if model_path is not None:
        _ENCODER = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    else:
        _ENCODER = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _ENCODER


def embed(texts: list[str], *, normalize: bool = True, batch_size: int = 32) -> np.ndarray:
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    encoder = _get_encoder()
    vecs = encoder.encode(
        cleaned,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LLAMA_URL     = "http://127.0.0.1:8989/v1/chat/completions"
CORPUS_DEFAULT = Path(__file__).parent / "data" / "corpus_synth_indexed_1000.jsonl"
TOP_K         = [1, 5, 10]

LORA_SYSTEM = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
LORA_PREFIX = "Quelles sont les expressions clés contenues à l'identique dans ce texte :\n"

# ---------------------------------------------------------------------------
# Chargement du corpus (texte + expressions corpus, même seed que les autres scripts)
# ---------------------------------------------------------------------------

def load_corpus(path: Path, n: int, seed: int = 42) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for chunk in doc.get("chunks", []):
                text  = chunk.get("chunk", "").strip()
                exprs = [e for e in chunk.get("expressions_caracteristiques", []) if isinstance(e, str) and e.strip()]
                if text and exprs:
                    chunks.append({"text": text, "expressions_corpus": exprs})
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return chunks[:n]

# ---------------------------------------------------------------------------
# Extraction LoRA
# ---------------------------------------------------------------------------

def _parse_json_list(text: str) -> list[str]:
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [x for x in result if isinstance(x, str)]
    except Exception:
        pass
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return [x for x in result if isinstance(x, str)]
        except Exception:
            pass
    return []


def lora_extract(text: str, timeout: float = 60.0) -> list[str]:
    payload = {
        "model": "local-llama-cpp",
        "messages": [
            {"role": "system", "content": LORA_SYSTEM},
            {"role": "user",   "content": LORA_PREFIX + text},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urlrequest.Request(
        LLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _parse_json_list(data["choices"][0]["message"]["content"].strip())
    except Exception as exc:
        print(f"\n  [LoRA] erreur : {exc}")
        return []


def load_or_extract_doc_expressions(chunks: list[dict], cache: Path) -> list[list[str]]:
    """Extrait les expressions LoRA pour chaque chunk document. Cache les résultats."""
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if len(cached) == len(chunks):
            print(f"Expressions documents chargées depuis le cache ({cache})")
            return cached

    exprs_list: list[list[str]] = []
    for i, chunk in enumerate(chunks):
        print(f"  Extraction expressions document {i+1}/{len(chunks)}…", end="\r", flush=True)
        exprs_list.append(lora_extract(chunk["text"]))
    print()
    cache.write_text(json.dumps(exprs_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Expressions documents sauvegardées dans {cache}")
    return exprs_list

# ---------------------------------------------------------------------------
# Construction des index (avec expressions LoRA côté document)
# ---------------------------------------------------------------------------

def build_indexes(
    chunks: list[dict],
    doc_exprs: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    n = len(chunks)

    print(f"Index A — embedding des {n} chunks bruts…")
    index_a = embed([c["text"] for c in chunks])

    print(f"Index B — embedding des expressions LoRA jointes…")
    joined  = [", ".join(e) if e else c["text"] for c, e in zip(chunks, doc_exprs)]
    index_b = embed(joined)

    print(f"Index C/D — embedding des expressions LoRA individuelles…")
    flat: list[str] = []
    boundaries: list[tuple[int, int]] = []
    for exprs, chunk in zip(doc_exprs, chunks):
        start = len(flat)
        if exprs:
            flat.extend(exprs)
        else:
            flat.append(chunk["text"])  # fallback : chunk entier
        boundaries.append((start, len(flat)))

    all_vecs = embed(flat, batch_size=64)
    index_c  = [all_vecs[s:e] for s, e in boundaries]

    print(f"  Dimension BGE-M3 : {index_a.shape[1]}")
    return index_a, index_b, index_c

# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def scores_dense(qvec: np.ndarray, index: np.ndarray) -> np.ndarray:
    return index @ qvec


def scores_late_interaction_asym(qvec: np.ndarray, index_c: list[np.ndarray]) -> np.ndarray:
    out = np.empty(len(index_c), dtype=np.float32)
    for i, vecs in enumerate(index_c):
        out[i] = float(np.max(vecs @ qvec))
    return out


def scores_late_interaction_sym(qe_vecs: np.ndarray, index_c: list[np.ndarray]) -> np.ndarray:
    out = np.empty(len(index_c), dtype=np.float32)
    for i, doc_vecs in enumerate(index_c):
        sim = qe_vecs @ doc_vecs.T
        out[i] = float(np.sum(np.max(sim, axis=1)))
    return out

# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(
    valid_pairs: list[tuple[int, str]],
    query_vecs: np.ndarray,
    query_expr_vecs: list[np.ndarray],
    index_a: np.ndarray,
    index_b: np.ndarray,
    index_c: list[np.ndarray],
) -> dict:
    accum: dict[str, dict] = {
        cond: {"rr": [], **{f"R@{k}": [] for k in TOP_K}}
        for cond in ("A", "B", "C", "D")
    }

    for j, (true_idx, _) in enumerate(valid_pairs):
        qv  = query_vecs[j]
        qev = query_expr_vecs[j]
        sc_d = scores_late_interaction_sym(qev, index_c) if len(qev) > 0 else None

        for cond, sc in [
            ("A", scores_dense(qv, index_a)),
            ("B", scores_dense(qv, index_b)),
            ("C", scores_late_interaction_asym(qv, index_c)),
            ("D", sc_d),
        ]:
            if sc is None:
                continue
            ranked = np.argsort(sc)[::-1]
            rank   = int(np.where(ranked == true_idx)[0][0])
            accum[cond]["rr"].append(1.0 / (rank + 1))
            for k in TOP_K:
                accum[cond][f"R@{k}"].append(1.0 if rank < k else 0.0)

    results = {}
    for cond, data in accum.items():
        n = len(data["rr"])
        if n == 0:
            continue
        results[cond] = {
            "MRR":    round(sum(data["rr"]) / n, 4),
            **{f"Recall@{k}": round(sum(data[f"R@{k}"]) / n, 4) for k in TOP_K},
            "n": n,
        }
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",             default=str(CORPUS_DEFAULT))
    parser.add_argument("--n",                  type=int, default=200)
    parser.add_argument("--doc-exprs-cache",    default="eval_rag_lora_doc_exprs.json")
    parser.add_argument("--queries-cache",      required=True)
    parser.add_argument("--query-exprs-cache",  required=True)
    parser.add_argument("--out",                required=True)
    args = parser.parse_args()

    # 1. Corpus
    print(f"\nChargement de {args.n} chunks…")
    chunks = load_corpus(Path(args.corpus), args.n)
    print(f"  {len(chunks)} chunks chargés")

    # 2. Expressions LoRA des documents
    print("\nExpressions LoRA des documents…")
    doc_exprs = load_or_extract_doc_expressions(chunks, Path(args.doc_exprs_cache))
    n_with = sum(1 for e in doc_exprs if e)
    print(f"  {n_with}/{len(chunks)} chunks avec expressions extraites")

    # 3. Requêtes (depuis cache)
    queries = json.loads(Path(args.queries_cache).read_text(encoding="utf-8"))
    valid_pairs = [(i, q) for i, q in enumerate(queries) if q and i < len(chunks)]
    print(f"  {len(valid_pairs)} requêtes valides")

    # 4. Expressions requêtes (depuis cache)
    all_query_exprs = json.loads(Path(args.query_exprs_cache).read_text(encoding="utf-8"))

    # 5. Index
    print("\nConstruction des index…")
    index_a, index_b, index_c = build_indexes(chunks, doc_exprs)

    # 6. Embedding requêtes
    print("\nEmbedding des requêtes…")
    query_texts = [q for _, q in valid_pairs]
    query_vecs  = embed(query_texts)

    # 7. Embedding expressions requêtes (pour D)
    flat_qe: list[str] = []
    qe_bounds: list[tuple[int, int]] = []
    for i, _ in valid_pairs:
        start = len(flat_qe)
        flat_qe.extend(all_query_exprs[i] if i < len(all_query_exprs) else [])
        qe_bounds.append((start, len(flat_qe)))

    if flat_qe:
        print("Embedding des expressions de requêtes…")
        all_qe = embed(flat_qe, batch_size=64)
        query_expr_vecs = [
            all_qe[s:e] if e > s else np.empty((0, all_qe.shape[1]), dtype=np.float32)
            for s, e in qe_bounds
        ]
    else:
        query_expr_vecs = [np.empty((0, 1024), dtype=np.float32)] * len(valid_pairs)

    # 8. Évaluation
    print("\nÉvaluation…")
    results = compute_metrics(valid_pairs, query_vecs, query_expr_vecs, index_a, index_b, index_c)

    # 9. Rapport
    n_q, n_d = len(valid_pairs), len(chunks)
    print(f"\n{'='*70}")
    print(f"RÉSULTATS — {n_q} requêtes, {n_d} documents (expressions LoRA côté doc)")
    print(f"{'='*70}")
    print(f"{'Condition':<36} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print(f"{'-'*70}")
    labels = {
        "A": "A) RAG classique",
        "B": "B) Expressions LoRA jointes",
        "C": "C) Late interaction asymétrique",
        "D": "D) Late interaction symétrique",
    }
    for cond, label in labels.items():
        if cond not in results:
            continue
        r = results[cond]
        n_label = f" (n={r['n']})" if r.get("n", n_q) < n_q else ""
        print(f"{label+n_label:<36} {r['MRR']:>7.4f} {r['Recall@1']:>7.4f} {r['Recall@5']:>7.4f} {r['Recall@10']:>7.4f}")
    print(f"{'='*70}\n")

    output = {
        "config":  {"n_chunks": n_d, "n_queries": n_q, "corpus": args.corpus,
                    "doc_exprs": "lora", "query_exprs": args.query_exprs_cache},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()
