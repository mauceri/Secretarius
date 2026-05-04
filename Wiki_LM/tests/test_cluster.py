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


# ---------------------------------------------------------------------------
# run_clustering — intégration
# ---------------------------------------------------------------------------

def _setup_wiki_with_embeds(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    """
    Crée n pages src- + embeddings factices formant 2 groupes bien séparés.
    Retourne (wiki_dir, embed_dir).
    """
    import json

    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    slugs = [f"src-{i:02d}" for i in range(n)]
    sources_dir = wiki_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    for i, slug in enumerate(slugs):
        (sources_dir / f"{slug}.md").write_text(
            f"---\ntitle: Source {i}\ncategory: source\ntags: [test]\n---\n\n"
            f"## Résumé\n\nTexte de test numéro {i}.\n",
            encoding="utf-8",
        )

    # Groupes : 0..n//2-1 vs n//2..n-1, vecteurs bien séparés
    rng = np.random.default_rng(0)
    vecs = np.zeros((n, 16), dtype=np.float32)
    half = n // 2
    vecs[:half, :8] = 1.0
    vecs[half:, 8:] = 1.0
    vecs += rng.standard_normal((n, 16)).astype(np.float32) * 0.05
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    np.save(embed_dir / "embeddings.npy", vecs)
    (embed_dir / "embeddings_index.json").write_text(
        json.dumps({"slugs": slugs}), encoding="utf-8"
    )

    return wiki_dir, embed_dir


def test_run_clustering_creates_output_dir(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    out_dir = wiki_dir / "clusterings" / "clustering-embeddings-hdbscan-2"
    assert out_dir.exists()


def test_run_clustering_creates_index_and_unclustered(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    out_dir = wiki_dir / "clusterings" / "clustering-embeddings-hdbscan-2"
    assert (out_dir / "index.md").exists()
    assert (out_dir / "unclustered.md").exists()


def test_run_clustering_cluster_files_have_members(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    stats = run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    assert stats["clusters"] >= 1
    out_dir = wiki_dir / "clusterings" / "clustering-embeddings-hdbscan-2"
    cluster_files = [f for f in out_dir.glob("cluster-*.md")]
    assert len(cluster_files) == stats["clusters"]

    for f in cluster_files:
        content = f.read_text(encoding="utf-8")
        assert "[[src-" in content


def test_run_clustering_stats_dict(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    stats = run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    assert "clusters" in stats
    assert "noise" in stats
    assert stats["clusters"] + stats["noise"] <= 6
    assert stats["signal"] == "embeddings"
    assert stats["param"] == 2


def test_run_clustering_returns_two_clusters_for_well_separated_data(tmp_path):
    """Données bien séparées en 2 groupes → exactement 2 clusters (param=2)."""
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path, n=10)
    stats = run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    assert stats["clusters"] == 2


# ---------------------------------------------------------------------------
# Endpoints serveur /cluster + /cluster-status
# ---------------------------------------------------------------------------

def test_cluster_status_initial(tmp_path):
    """GET /cluster-status retourne running=False initialement."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

    import server as srv_module

    # Réinitialiser le statut
    srv_module._cluster_status["running"] = False
    srv_module._cluster_status["last"] = None
    srv_module._cluster_status["error"] = None

    client = srv_module.app.test_client()
    resp = client.get("/cluster-status")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["running"] is False
    assert data["last"] is None


def test_cluster_endpoint_missing_param(tmp_path):
    """POST /cluster sans 'param' retourne une erreur 400."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

    import server as srv_module

    client = srv_module.app.test_client()
    resp = client.post("/cluster", json={"signal": "embeddings"})

    assert resp.status_code == 400
    assert "param" in resp.get_json().get("error", "")
