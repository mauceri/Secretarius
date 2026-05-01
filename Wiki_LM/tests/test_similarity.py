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


# ---------------------------------------------------------------------------
# CoLinkSimilarity
# ---------------------------------------------------------------------------

def _write_page(wiki_dir: Path, slug: str, body: str, tags: list[str] | None = None) -> None:
    tags_yaml = f"tags: {tags}\n" if tags else ""
    (wiki_dir / f"{slug}.md").write_text(
        f"---\ntitle: {slug}\n{tags_yaml}---\n\n{body}", encoding="utf-8"
    )


def test_colink_identical_pages(tmp_path):
    """Deux pages partageant exactement les mêmes liens → Jaccard = 1.0."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-foo]] [[e-bar]]")
    _write_page(wiki_dir, "src-b", "[[c-foo]] [[e-bar]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1.0)


def test_colink_no_overlap(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-foo]]")
    _write_page(wiki_dir, "src-b", "[[c-bar]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(0.0)


def test_colink_partial_overlap(tmp_path):
    """Jaccard({a,b}, {b,c}) = 1/3."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-a]] [[c-b]]")
    _write_page(wiki_dir, "src-b", "[[c-b]] [[c-c]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1 / 3, abs=1e-5)


def test_colink_symmetry(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-foo]] [[c-bar]]")
    _write_page(wiki_dir, "src-b", "[[c-bar]] [[e-baz]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(mat[1, 0])


# ---------------------------------------------------------------------------
# TagSimilarity
# ---------------------------------------------------------------------------

def test_tag_similarity_identical(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "", tags=["foo", "bar"])
    _write_page(wiki_dir, "src-b", "", tags=["foo", "bar"])

    from similarity import TagSimilarity
    mat = TagSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1.0)


def test_tag_similarity_partial(tmp_path):
    """Jaccard({foo,bar}, {bar,baz}) = 1/3."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "", tags=["foo", "bar"])
    _write_page(wiki_dir, "src-b", "", tags=["bar", "baz"])

    from similarity import TagSimilarity
    mat = TagSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1 / 3, abs=1e-5)


def test_tag_similarity_no_tags(tmp_path):
    """Pages sans tags → similarité 0."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "rien")
    _write_page(wiki_dir, "src-b", "rien")

    from similarity import TagSimilarity
    mat = TagSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CombinedSimilarity
# ---------------------------------------------------------------------------

def test_combined_equal_weights():
    """Avec deux signaux égaux à 0.4 et 0.8 → résultat = 0.6."""
    from similarity import BaseSimilarity, CombinedSimilarity

    class _Fixed(BaseSimilarity):
        def __init__(self, val: float) -> None:
            self._val = val
        def compute(self, slugs: list[str]) -> np.ndarray:
            n = len(slugs)
            return np.full((n, n), self._val, dtype=np.float32)

    sig = CombinedSimilarity([_Fixed(0.4), _Fixed(0.8)])
    mat = sig.compute(["a", "b"])

    np.testing.assert_allclose(mat, 0.6, atol=1e-5)


def test_combined_custom_weights():
    """Poids 1:3 → résultat = 0.25*0.4 + 0.75*0.8 = 0.7."""
    from similarity import BaseSimilarity, CombinedSimilarity

    class _Fixed(BaseSimilarity):
        def __init__(self, val: float) -> None:
            self._val = val
        def compute(self, slugs: list[str]) -> np.ndarray:
            n = len(slugs)
            return np.full((n, n), self._val, dtype=np.float32)

    sig = CombinedSimilarity([_Fixed(0.4), _Fixed(0.8)], weights=[1.0, 3.0])
    mat = sig.compute(["a", "b"])

    np.testing.assert_allclose(mat, 0.7, atol=1e-5)
