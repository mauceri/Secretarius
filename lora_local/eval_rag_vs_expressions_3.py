#!/usr/bin/env python3
"""
Expérience 3 — RAG classique vs indexation par expressions.

Même protocole que l'expérience 1 (requêtes depuis le texte brut),
mais les expressions côté requête (condition D) sont extraites par
DeepSeek plutôt que par LoRA.

Objectif : isoler si c'est la qualité de l'extracteur LoRA (entraîné
sur des textes, pas des questions) qui limite la late interaction
symétrique dans l'expérience 1.

Quatre conditions :
  A) RAG classique              : embedding BGE-M3 du chunk brut
  B) Expressions jointes        : embedding BGE-M3 des expressions jointes (une seule chaîne)
  C) Late interaction asymét.   : score = max cos(query_vec, expr_doc_j)
  D) Late interaction symét.    : score = Σ_i max_j cos(expr_query_i, expr_doc_j)
                                  (expressions requête extraites par DeepSeek)

Protocole :
  - Corpus  : corpus_synth_indexed_1000.jsonl (chunks + expressions_caracteristiques)
  - Requête : question synthétique générée par DeepSeek à partir du texte brut (= exp 1)
  - Vérité terrain : la requête i doit retrouver le chunk i (tâche 1-to-1)
  - Métriques : Recall@1, @5, @10, MRR

Usage :
    source llenv/bin/activate
    source ~/.config/secrets.env
    python eval_rag_vs_expressions_3.py [--n 200] [--out eval_rag3_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib import request as urlrequest

import numpy as np

# ---------------------------------------------------------------------------
# Embedding BGE-M3 en CPU (contourne les incompatibilités ROCm de llenv)
# ---------------------------------------------------------------------------
_SECRETARIUS_LOCAL = Path(__file__).resolve().parents[1] / "Prototype" / "secretarius_local"

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    from sentence_transformers import SentenceTransformer
    import sys as _sys
    # Résolution du chemin local du modèle (même logique que runtime_paths.py)
    if str(_SECRETARIUS_LOCAL) not in _sys.path:
        _sys.path.insert(0, str(_SECRETARIUS_LOCAL))
    from runtime_paths import resolve_sentence_model_path  # type: ignore
    model_path = resolve_sentence_model_path()
    if model_path is not None:
        _ENCODER = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    else:
        _ENCODER = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _ENCODER


def embed_expressions_multilingual(texts: list[str], *, normalize: bool = True, batch_size: int = 32) -> dict:
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return {"embeddings": [], "dimension": 0}
    encoder = _get_encoder()
    vecs = encoder.encode(
        cleaned,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return {"embeddings": vecs.tolist(), "dimension": int(vecs.shape[1])}

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
CORPUS_DEFAULT = Path(__file__).parent / "data" / "corpus_synth_indexed_1000.jsonl"
TOP_K          = [1, 5, 10]
LLAMA_URL      = "http://127.0.0.1:8989/v1/chat/completions"

QUERY_GEN_SYSTEM = (
    "Voici un texte en français. Écris une question naturelle et précise "
    "qu'un utilisateur pourrait taper dans un moteur de recherche pour trouver "
    "ce texte. Réponds UNIQUEMENT par la question, sans explication ni ponctuation finale."
)

LORA_EXTRACT_SYSTEM = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
LORA_USER_PREFIX = "Quelles sont les expressions clés contenues à l'identique dans ce texte :\n"

# ---------------------------------------------------------------------------
# Chargement du corpus
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
                    chunks.append({"text": text, "expressions": exprs})
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return chunks[:n]

# ---------------------------------------------------------------------------
# Génération de requêtes synthétiques
# ---------------------------------------------------------------------------

def _deepseek_call(system: str, user: str, api_key: str, max_tokens: int = 120) -> str | None:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urlrequest.Request(
        DEEPSEEK_URL, data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"\n  [DeepSeek] erreur : {exc}")
        return None


def load_or_generate_queries(chunks: list[dict], api_key: str, cache: Path) -> list[str | None]:
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if len(cached) == len(chunks):
            print(f"Requêtes chargées depuis le cache ({cache})")
            return cached
        print("Cache incomplet, régénération…")

    queries: list[str | None] = []
    for i, chunk in enumerate(chunks):
        print(f"  Génération requête {i+1}/{len(chunks)}…", end="\r", flush=True)
        q = _deepseek_call(QUERY_GEN_SYSTEM, chunk["text"][:1200], api_key)
        queries.append(q)
        time.sleep(0.08)
    print()
    cache.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Requêtes sauvegardées dans {cache}")
    return queries

# ---------------------------------------------------------------------------
# Construction des index BGE-M3
# ---------------------------------------------------------------------------

def build_indexes(
    chunks: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Retourne :
      index_a : (N, D)           embeddings des chunks bruts
      index_b : (N, D)           embeddings des expressions jointes
      index_c : liste de N (k_i, D)  embeddings par expression
    """
    n = len(chunks)

    print(f"Index A — embedding des {n} chunks bruts…")
    res_a   = embed_expressions_multilingual([c["text"] for c in chunks], normalize=True)
    index_a = np.array(res_a["embeddings"], dtype=np.float32)

    print(f"Index B — embedding des expressions jointes…")
    joined  = [", ".join(c["expressions"]) for c in chunks]
    res_b   = embed_expressions_multilingual(joined, normalize=True)
    index_b = np.array(res_b["embeddings"], dtype=np.float32)

    print(f"Index C — embedding de chaque expression individuellement…")
    flat_exprs: list[str] = []
    boundaries: list[tuple[int, int]] = []
    for c in chunks:
        start = len(flat_exprs)
        flat_exprs.extend(c["expressions"])
        boundaries.append((start, len(flat_exprs)))

    res_c   = embed_expressions_multilingual(flat_exprs, normalize=True, batch_size=64)
    all_vecs = np.array(res_c["embeddings"], dtype=np.float32)
    index_c  = [all_vecs[s:e] for s, e in boundaries]

    dim = index_a.shape[1]
    print(f"  Dimension BGE-M3 : {dim}")
    return index_a, index_b, index_c

# ---------------------------------------------------------------------------
# Scores de retrieval
# ---------------------------------------------------------------------------

def scores_dense(qvec: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Cosine (= produit scalaire sur vecteurs normalisés). qvec : (D,), index : (N, D) → (N,)"""
    return index @ qvec


def scores_late_interaction(qvec: np.ndarray, index_c: list[np.ndarray]) -> np.ndarray:
    """Asymétrique : score(doc_i) = max_j cosine(qvec, expr_j^i)"""
    out = np.empty(len(index_c), dtype=np.float32)
    for i, expr_vecs in enumerate(index_c):
        out[i] = float(np.max(expr_vecs @ qvec))
    return out


def scores_late_interaction_sym(query_expr_vecs: np.ndarray, index_c: list[np.ndarray]) -> np.ndarray:
    """Symétrique : score(doc_i) = Σ_k max_j cosine(query_expr_k, doc_expr_j^i)"""
    out = np.empty(len(index_c), dtype=np.float32)
    for i, doc_vecs in enumerate(index_c):
        # sim_matrix : (n_query_exprs, n_doc_exprs)
        sim_matrix = query_expr_vecs @ doc_vecs.T
        out[i] = float(np.sum(np.max(sim_matrix, axis=1)))
    return out


# ---------------------------------------------------------------------------
# Extraction d'expressions via LoRA (llama.cpp)
# ---------------------------------------------------------------------------

def _parse_json_list(text: str) -> list[str]:
    import re
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
            {"role": "system", "content": LORA_EXTRACT_SYSTEM},
            {"role": "user",   "content": LORA_USER_PREFIX + text},
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
        print(f"\n  [LoRA] erreur extraction : {exc}")
        return []


DEEPSEEK_EXPR_SYSTEM = (
    "Voici une question en français. Extrais les expressions-clés, entités nommées, "
    "dates et lieux remarquables qu'un moteur de recherche devrait utiliser pour trouver "
    "un document pertinent. Réponds UNIQUEMENT par un tableau JSON de chaînes."
)


def deepseek_extract(query: str, api_key: str) -> list[str]:
    raw = _deepseek_call(DEEPSEEK_EXPR_SYSTEM, query, api_key, max_tokens=256)
    if not raw:
        return []
    return _parse_json_list(raw)


def load_or_extract_query_expressions(
    queries: list[str | None], cache: Path, api_key: str = ""
) -> list[list[str]]:
    """Extrait les expressions de chaque requête via DeepSeek. Cache les résultats."""
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if len(cached) == len(queries):
            print(f"Expressions requêtes chargées depuis le cache ({cache})")
            return cached

    exprs_list: list[list[str]] = []
    for i, q in enumerate(queries):
        print(f"  Extraction expressions requête {i+1}/{len(queries)}…", end="\r", flush=True)
        if q:
            exprs_list.append(deepseek_extract(q, api_key))
            time.sleep(0.08)
        else:
            exprs_list.append([])
    print()
    cache.write_text(json.dumps(exprs_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Expressions requêtes sauvegardées dans {cache}")
    return exprs_list

# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(
    valid_pairs: list[tuple[int, str]],
    query_vecs: np.ndarray,
    query_expr_vecs: list[np.ndarray],  # embeddings des expressions par requête (pour D)
    index_a: np.ndarray,
    index_b: np.ndarray,
    index_c: list[np.ndarray],
) -> dict:
    conds = ("A", "B", "C", "D")
    accum: dict[str, dict] = {
        cond: {"rr": [], **{f"R@{k}": [] for k in TOP_K}}
        for cond in conds
    }

    for j, (true_idx, _) in enumerate(valid_pairs):
        qv  = query_vecs[j]
        qev = query_expr_vecs[j]  # (n_exprs, D) ou tableau vide

        sc_d = scores_late_interaction_sym(qev, index_c) if len(qev) > 0 else None

        for cond, sc in [
            ("A", scores_dense(qv, index_a)),
            ("B", scores_dense(qv, index_b)),
            ("C", scores_late_interaction(qv, index_c)),
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
    parser = argparse.ArgumentParser(description="Évaluation RAG vs expressions")
    parser.add_argument("--corpus",              default=str(CORPUS_DEFAULT))
    parser.add_argument("--n",                   type=int, default=200)
    parser.add_argument("--queries-cache",        default="eval_rag_queries.json")
    parser.add_argument("--query-exprs-cache",    default="eval_rag3_query_exprs.json")
    parser.add_argument("--out",                  default="eval_rag3_results.json")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY manquant — lancez : source ~/.config/secrets.env")

    # 1. Corpus
    print(f"\nChargement de {args.n} chunks depuis {args.corpus}…")
    chunks = load_corpus(Path(args.corpus), args.n)
    print(f"  {len(chunks)} chunks chargés")

    # 2. Requêtes synthétiques
    print("\nRequêtes synthétiques…")
    queries = load_or_generate_queries(chunks, api_key, Path(args.queries_cache))

    valid_pairs = [(i, q) for i, q in enumerate(queries) if q]
    print(f"  {len(valid_pairs)} requêtes valides / {len(chunks)}")

    # 3. Index
    print("\nConstruction des index…")
    index_a, index_b, index_c = build_indexes(chunks)

    # 4. Embedding des requêtes (texte brut, pour A/B/C)
    print("\nEmbedding des requêtes…")
    query_texts = [q for _, q in valid_pairs]
    res_q       = embed_expressions_multilingual(query_texts, normalize=True)
    query_vecs  = np.array(res_q["embeddings"], dtype=np.float32)

    # 5. Extraction et embedding des expressions des requêtes (pour D)
    print("\nExtraction expressions des requêtes via LoRA…")
    all_query_exprs = load_or_extract_query_expressions(queries, Path(args.query_exprs_cache), api_key)

    # Embedding des expressions de requête pour les paires valides
    flat_qexprs: list[str] = []
    qexpr_boundaries: list[tuple[int, int]] = []
    for i, _ in valid_pairs:
        start = len(flat_qexprs)
        flat_qexprs.extend(all_query_exprs[i])
        qexpr_boundaries.append((start, len(flat_qexprs)))

    if flat_qexprs:
        print("Embedding des expressions de requêtes…")
        res_qe   = embed_expressions_multilingual(flat_qexprs, normalize=True, batch_size=64)
        all_qe_vecs = np.array(res_qe["embeddings"], dtype=np.float32)
        query_expr_vecs = [
            all_qe_vecs[s:e] if e > s else np.empty((0, all_qe_vecs.shape[1]), dtype=np.float32)
            for s, e in qexpr_boundaries
        ]
    else:
        query_expr_vecs = [np.empty((0, 1024), dtype=np.float32)] * len(valid_pairs)

    n_with_exprs = sum(1 for v in query_expr_vecs if len(v) > 0)
    print(f"  {n_with_exprs}/{len(valid_pairs)} requêtes avec expressions extraites")

    # 6. Évaluation
    print("\nÉvaluation…")
    results = compute_metrics(valid_pairs, query_vecs, query_expr_vecs, index_a, index_b, index_c)

    # 7. Rapport
    n_q = len(valid_pairs)
    n_d = len(chunks)
    print(f"\n{'='*70}")
    print(f"RÉSULTATS — {n_q} requêtes, {n_d} documents")
    print(f"{'='*70}")
    print(f"{'Condition':<36} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print(f"{'-'*70}")
    labels = {
        "A": "A) RAG classique",
        "B": "B) Expressions jointes",
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
        "config":  {"n_chunks": n_d, "n_queries": n_q, "corpus": args.corpus},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()
