"""Tests pour cluster.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# ---------------------------------------------------------------------------
# _find_paragon
# ---------------------------------------------------------------------------

def test_find_paragon_selects_most_central():
    """Le parangon est le document avec la plus haute similarité moyenne."""
    from cluster import _find_paragon

    # index 1 : avg = (0.5 + 1.0 + 0.6) / 3 = 0.7 → le plus central
    sim = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.6],
        [0.2, 0.6, 1.0],
    ], dtype=np.float32)

    assert _find_paragon([0, 1, 2], sim) == 1


def test_find_paragon_single_element():
    from cluster import _find_paragon
    sim = np.array([[1.0]], dtype=np.float32)
    assert _find_paragon([0], sim) == 0


def test_find_paragon_with_offset_indices():
    """Les indices passés peuvent être un sous-ensemble non continu."""
    from cluster import _find_paragon

    sim = np.array([
        [1.0, 0.1, 0.9],
        [0.1, 1.0, 0.1],
        [0.9, 0.1, 1.0],
    ], dtype=np.float32)

    result = _find_paragon([0, 2], sim)
    assert result in (0, 2)


# ---------------------------------------------------------------------------
# _nearest_clusters
# ---------------------------------------------------------------------------

def test_nearest_clusters_returns_top_k():
    from cluster import _nearest_clusters

    centroids = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.9, 0.1], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([0.0, 0.9], dtype=np.float32),
    }

    nearest = _nearest_clusters(0, centroids, top_k=2)

    assert len(nearest) == 2
    assert nearest[0][0] == 1
    assert nearest[0][1] >= nearest[1][1]


def test_nearest_clusters_excludes_self():
    from cluster import _nearest_clusters

    centroids = {0: np.array([1.0, 0.0]), 1: np.array([0.5, 0.5])}
    nearest = _nearest_clusters(0, centroids, top_k=3)

    assert all(cid != 0 for cid, _ in nearest)


# ---------------------------------------------------------------------------
# _describe_cluster
# ---------------------------------------------------------------------------

def test_describe_cluster_with_mock_llm():
    from cluster import _describe_cluster

    class MockLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return "TITRE: Philosophie du langage\nDESCRIPTION: Ce groupe traite du langage."

    pages = {"src-a": {"title": "Titre A", "abstract": "Résumé A long."}}

    title, desc = _describe_cluster("src-a", pages, MockLLM())

    assert title == "Philosophie du langage"
    assert "langage" in desc


def test_describe_cluster_no_llm_returns_defaults():
    from cluster import _describe_cluster

    pages = {"src-a": {"title": "A", "abstract": ""}}

    title, desc = _describe_cluster("src-a", pages, llm=None)

    assert isinstance(title, str)
    assert isinstance(desc, str)
