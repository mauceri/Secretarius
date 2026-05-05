# tools/transfers.py
"""
Algorithme des transferts pour le clustering de documents.

Complexité : O(k × N × C) avec N docs, C clusters, k itérations.
Origine : algorithme de classification sur sacs de mots (Salton), adapté aux embeddings denses.
"""
from __future__ import annotations

import numpy as np

QUALITY_THRESHOLD: float = 0.20
MIN_PAGES_FOR_CLUSTERING: int = 50


def estimate_theta(
    sim: np.ndarray,
    percentile: float = 75.0,
    sample_size: int = 50_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estime θ par tirage aléatoire dans le triangle supérieur de sim."""
    if rng is None:
        rng = np.random.default_rng()
    n = sim.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    total = len(rows)
    if total <= sample_size:
        sample = sim[rows, cols]
    else:
        idx = rng.choice(total, size=sample_size, replace=False)
        sample = sim[rows[idx], cols[idx]]
    return float(np.percentile(sample, percentile))


def run_transfers(
    slugs: list[str],
    sim: np.ndarray,
    theta: float,
    max_k: int | None = None,
    force_assign: bool = False,
    dry_run: bool = False,
    initial_partition: dict[int, list[int]] | None = None,
    max_iter: int = 100,
    min_gain_delta: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> "dict[int, list[int]] | dict":
    n = len(slugs)
    if rng is None:
        rng = np.random.default_rng()

    labels = np.full(n, -1, dtype=np.intp)
    clusters: dict[int, set[int]] = {}
    next_id = [0]

    def _centroid(idx_set: set[int]) -> np.ndarray:
        return sim[list(idx_set)].mean(axis=0).astype(np.float32)

    centroids: dict[int, np.ndarray] = {}

    new_indices: list[int] = list(range(n))
    if initial_partition is not None:
        for cid, idx_list in initial_partition.items():
            clusters[cid] = set(idx_list)
            for idx in idx_list:
                if 0 <= idx < n:
                    labels[idx] = cid
            next_id[0] = max(next_id[0], cid + 1)
        centroids = {cid: _centroid(m) for cid, m in clusters.items()}
        new_indices = [i for i in range(n) if labels[i] == -1]

    def _add(x: int, cid: int) -> None:
        size = len(clusters[cid])
        centroids[cid] = ((centroids[cid] * size + sim[x]) / (size + 1)).astype(np.float32)
        clusters[cid].add(x)
        labels[x] = cid

    def _remove(x: int, cid: int) -> None:
        size = len(clusters[cid])
        clusters[cid].discard(x)
        if clusters[cid]:
            centroids[cid] = ((centroids[cid] * size - sim[x]) / (size - 1)).astype(np.float32)
        else:
            del clusters[cid]
            del centroids[cid]
        labels[x] = -1

    def _new_cluster(x: int) -> None:
        cid = next_id[0]
        next_id[0] += 1
        clusters[cid] = {x}
        centroids[cid] = sim[x].copy().astype(np.float32)
        labels[x] = cid

    def _best_other(x: int, exclude: int = -1) -> tuple[int, float]:
        best_cid, best_gain = -1, -np.inf
        for cid in clusters:
            if cid == exclude:
                continue
            g = float(centroids[cid][x])
            if g > best_gain:
                best_gain, best_cid = g, cid
        return best_cid, best_gain

    # --- Algo 1 : partition initiale ---
    order = rng.permutation(len(new_indices))
    for pos in order:
        x = new_indices[pos]
        if not clusters:
            _new_cluster(x)
            continue
        best_cid, best_gain = _best_other(x)
        if best_gain > theta:
            _add(x, best_cid)
        elif max_k is None or len(clusters) < max_k:
            _new_cluster(x)
        # else : stays -1 (poubelle)

    # --- Algo 2 : amélioration par transferts ---
    for _ in range(max_iter):
        changed = False
        for x in range(n):
            cx = int(labels[x])
            if cx == -1:
                continue
            size_cx = len(clusters[cx])
            if size_cx > 1:
                contrib = (size_cx * float(centroids[cx][x]) - float(sim[x, x])) / (size_cx - 1)
            else:
                contrib = 0.0
            best_cid, best_gain = _best_other(x, exclude=cx)
            if best_cid >= 0 and best_gain > contrib + min_gain_delta:
                old_cx = cx
                _remove(x, old_cx)
                if best_cid in clusters:
                    _add(x, best_cid)
                else:
                    _new_cluster(x)
                changed = True
        if not changed:
            break

    return {cid: list(m) for cid, m in clusters.items()}
