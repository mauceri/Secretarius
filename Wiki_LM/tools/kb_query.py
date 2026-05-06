# tools/kb_query.py
"""Requête de proximité aux axes de la base de connaissance."""
from __future__ import annotations

import json
from pathlib import Path

import frontmatter
import numpy as np


def kb_query(
    vec: np.ndarray,
    kb_dir: Path,
    top_k: int = 3,
) -> list[dict]:
    """
    Retourne les top_k axes les plus proches du vecteur vec.

    vec doit être L2-normalisé (même convention que les embeddings BGE-M3).

    Returns:
        [{"id": "axis-0001", "title": "...", "score": 0.81, "tags": [...]}, ...]
        Liste vide si la base est inexistante ou vide.
    """
    axes_npy = kb_dir / "embeddings" / "axes.npy"
    axes_index = kb_dir / "embeddings" / "axes_index.json"

    if not axes_npy.exists() or not axes_index.exists():
        return []

    matrix = np.load(axes_npy)                                          # (K, dim)
    ids: list[str] = json.loads(axes_index.read_text(encoding="utf-8"))["ids"]

    scores = matrix @ vec                                                # (K,) cosine sim
    k = min(top_k, len(ids))
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        axis_id = ids[int(i)]
        axis_path = kb_dir / "axes" / f"{axis_id}.md"
        title = axis_id
        tags: list[str] = []
        if axis_path.exists():
            post = frontmatter.load(axis_path)
            title = str(post.get("title", axis_id))
            tags = list(post.get("tags", []))
        results.append({
            "id": axis_id,
            "title": title,
            "score": float(scores[int(i)]),
            "tags": tags,
        })
    return results
