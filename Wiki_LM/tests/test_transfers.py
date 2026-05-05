# tests/test_transfers.py
"""Tests pour transfers.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_sim(n: int = 10, rng_seed: int = 42) -> np.ndarray:
    """Matrice n×n : docs 0..n//2-1 groupe A, n//2..n-1 groupe B.
    Similarités intra ≈ 0.80, inter ≈ 0.20. Diagonale = 1.0."""
    rng = np.random.default_rng(rng_seed)
    half = n // 2
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            base = 0.80 if (i < half) == (j < half) else 0.20
            sim[i, j] = base + float(rng.standard_normal() * 0.02)
    sim = np.clip(sim, 0.0, 1.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim


def test_estimate_theta_in_unit_interval():
    from transfers import estimate_theta
    sim = _make_sim()
    theta = estimate_theta(sim, rng=np.random.default_rng(0))
    assert 0.0 <= theta <= 1.0


def test_estimate_theta_reproducible():
    from transfers import estimate_theta
    sim = _make_sim()
    t1 = estimate_theta(sim, rng=np.random.default_rng(7))
    t2 = estimate_theta(sim, rng=np.random.default_rng(7))
    assert t1 == t2


def test_estimate_theta_reflects_separation():
    """75th percentile doit séparer inter (~0.2) et intra (~0.8)."""
    from transfers import estimate_theta
    sim = _make_sim(n=20)
    theta = estimate_theta(sim, percentile=75.0, rng=np.random.default_rng(0))
    assert 0.2 < theta < 0.8


def test_estimate_theta_small_matrix_no_crash():
    """Matrice petite (total < sample_size) : pas d'erreur."""
    from transfers import estimate_theta
    sim = _make_sim(n=4)
    theta = estimate_theta(sim, sample_size=1_000_000, rng=np.random.default_rng(0))
    assert 0.0 <= theta <= 1.0


def test_run_transfers_two_clusters():
    """Données bien séparées → exactement 2 clusters, tous docs assignés."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    result = run_transfers(slugs, sim, theta=0.5, rng=np.random.default_rng(0))
    assert len(result) == 2
    all_idx = {i for m in result.values() for i in m}
    assert all_idx == set(range(10))


def test_run_transfers_group_coherence():
    """Chaque cluster ne contient que des membres du même groupe thématique."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    result = run_transfers(slugs, sim, theta=0.5, rng=np.random.default_rng(0))
    for members in result.values():
        groups = {i < 5 for i in members}
        assert len(groups) == 1, f"Cluster mélangé : {members}"


def test_run_transfers_high_theta_creates_singletons():
    """Sans max_k, theta=0.99 → Algo 1 crée des singletons.
    Algo 2 peut ensuite fusionner certains singletons par transfert.
    On vérifie : tous les docs sont assignés et chaque cluster est ≥ 1."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    result = run_transfers(slugs, sim, theta=0.99, rng=np.random.default_rng(0))
    all_idx = {i for m in result.values() for i in m}
    assert all_idx == set(range(10))
    assert len(result) >= 1
    for members in result.values():
        assert len(members) >= 1


def test_run_transfers_terminates_with_max_iter():
    """max_iter=0 désactive Algo 2 — la partition est celle d'Algo 1 seul."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    result = run_transfers(slugs, sim, theta=0.5, max_iter=0,
                           rng=np.random.default_rng(0))
    assert isinstance(result, dict)
    all_idx = {i for m in result.values() for i in m}
    assert all_idx == set(range(10))
    for members in result.values():
        assert len(members) >= 1


def test_run_transfers_min_gain_delta_stops_marginal():
    """min_gain_delta très élevé → Algo 2 n'effectue aucun transfert (résultat valide)."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    result = run_transfers(slugs, sim, theta=0.5, min_gain_delta=999.0,
                           rng=np.random.default_rng(0))
    all_idx = {i for m in result.values() for i in m}
    assert isinstance(result, dict)
    assert len(all_idx) == 10


def test_run_transfers_max_k():
    """max_k=1 : une seule classe autorisée, le reste reste en poubelle."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    result = run_transfers(slugs, sim, theta=0.5, max_k=1,
                           rng=np.random.default_rng(0))
    assert len(result) == 1
    in_cluster = sum(len(m) for m in result.values())
    assert in_cluster < 10  # des docs en poubelle


def test_run_transfers_force_assign_no_poubelle():
    """force_assign=True + max_k=1 : tous les docs assignés, aucun en poubelle."""
    from transfers import run_transfers
    sim = _make_sim(n=10)
    slugs = [f"s{i}" for i in range(10)]
    # max_k=1 : Algo 1 crée 1 cluster max, le reste va en poubelle.
    # force_assign=True : les docs en poubelle doivent être assignés au seul cluster.
    result = run_transfers(slugs, sim, theta=0.5, max_k=1, force_assign=True,
                           rng=np.random.default_rng(0))
    all_idx = {i for m in result.values() for i in m}
    assert all_idx == set(range(10))
