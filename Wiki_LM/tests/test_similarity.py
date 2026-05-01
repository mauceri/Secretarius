"""Tests pour similarity.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_dir(tmp_path: Path, slugs: list[str], dim: int = 8) -> Path:
    """Crée un répertoire d'embeddings factice avec vecteurs L2-normalisés."""
    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(slugs), dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(embed_dir / "embeddings.npy", vecs)
    (embed_dir / "embeddings_index.json").write_text(
        json.dumps({"slugs": slugs}), encoding="utf-8"
    )
    return embed_dir


# ---------------------------------------------------------------------------
# EmbeddingSimilarity
# ---------------------------------------------------------------------------

def test_embedding_similarity_diagonal_is_one(tmp_path):
    slugs = ["src-a", "src-b", "src-c"]
    embed_dir = _make_embed_dir(tmp_path, slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(slugs)

    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)


def test_embedding_similarity_values_in_range(tmp_path):
    slugs = ["src-a", "src-b", "src-c"]
    embed_dir = _make_embed_dir(tmp_path, slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(slugs)

    assert float(mat.min()) >= 0.0
    assert float(mat.max()) <= 1.0 + 1e-5


def test_embedding_similarity_symmetry(tmp_path):
    slugs = ["src-a", "src-b", "src-c"]
    embed_dir = _make_embed_dir(tmp_path, slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(slugs)

    np.testing.assert_allclose(mat, mat.T, atol=1e-5)


def test_embedding_similarity_missing_dir(tmp_path):
    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(tmp_path / "nonexistent").compute(["src-a", "src-b"])

    assert mat.shape == (2, 2)
    assert float(mat.sum()) == 0.0


def test_embedding_similarity_subset(tmp_path):
    """Requête sur un sous-ensemble des slugs indexés."""
    all_slugs = ["src-a", "src-b", "src-c", "src-d"]
    embed_dir = _make_embed_dir(tmp_path, all_slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(["src-a", "src-c"])

    assert mat.shape == (2, 2)
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)
