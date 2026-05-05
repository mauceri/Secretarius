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
    raise NotImplementedError
