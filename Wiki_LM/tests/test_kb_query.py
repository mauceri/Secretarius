# tests/test_kb_query.py
"""Tests pour kb_query.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_kb(tmp_path: Path, n_axes: int = 3, dim: int = 8) -> tuple[Path, np.ndarray, list[str]]:
    kb_dir = tmp_path / "kb"
    (kb_dir / "axes").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir(parents=True)

    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_axes, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(kb_dir / "embeddings" / "axes.npy", mat)

    ids = [f"axis-{i + 1:04d}" for i in range(n_axes)]
    (kb_dir / "embeddings" / "axes_index.json").write_text(
        json.dumps({"ids": ids}), encoding="utf-8"
    )
    for i, aid in enumerate(ids):
        (kb_dir / "axes" / f"{aid}.md").write_text(
            f"---\ntitle: Axe {i + 1}\ntags: [tag{i}]\n---\n\nDescription.\n",
            encoding="utf-8",
        )
    return kb_dir, mat, ids


def test_kb_query_returns_top_k(tmp_path):
    from kb_query import kb_query
    kb_dir, mat, ids = _make_kb(tmp_path, n_axes=3)
    vec = mat[0].copy()                        # proche de axis-0001
    results = kb_query(vec, kb_dir, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "axis-0001"     # plus proche = lui-même
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_kb_query_scores_decreasing(tmp_path):
    from kb_query import kb_query
    kb_dir, mat, _ = _make_kb(tmp_path, n_axes=3)
    vec = mat[1].copy()
    results = kb_query(vec, kb_dir, top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_kb_query_includes_title_and_tags(tmp_path):
    from kb_query import kb_query
    kb_dir, mat, _ = _make_kb(tmp_path, n_axes=3)
    results = kb_query(mat[0].copy(), kb_dir, top_k=1)
    assert results[0]["title"] == "Axe 1"
    assert results[0]["tags"] == ["tag0"]


def test_kb_query_empty_kb(tmp_path):
    from kb_query import kb_query
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    result = kb_query(np.ones(8, dtype=np.float32), kb_dir, top_k=3)
    assert result == []


def test_kb_query_top_k_greater_than_axes(tmp_path):
    """top_k > nombre d'axes → retourne tous les axes disponibles."""
    from kb_query import kb_query
    kb_dir, mat, ids = _make_kb(tmp_path, n_axes=2)
    results = kb_query(mat[0].copy(), kb_dir, top_k=10)
    assert len(results) == 2


def test_kb_query_axis_without_md_file(tmp_path):
    """Axe sans fichier .md → title=axis_id, tags=[]."""
    from kb_query import kb_query
    kb_dir = tmp_path / "kb"
    (kb_dir / "axes").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir(parents=True)
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(8).astype(np.float32)
    vec /= np.linalg.norm(vec)
    mat = vec.reshape(1, -1)
    np.save(kb_dir / "embeddings" / "axes.npy", mat)
    (kb_dir / "embeddings" / "axes_index.json").write_text(
        json.dumps({"ids": ["axis-0001"]}), encoding="utf-8"
    )
    # Pas de fichier axis-0001.md créé
    results = kb_query(vec, kb_dir, top_k=1)
    assert results[0]["title"] == "axis-0001"
    assert results[0]["tags"] == []


def test_kb_query_dim_mismatch_raises(tmp_path):
    """Dimension du vecteur incompatible → ValueError."""
    from kb_query import kb_query
    kb_dir, mat, _ = _make_kb(tmp_path, n_axes=2, dim=8)
    bad_vec = np.ones(16, dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        kb_query(bad_vec, kb_dir, top_k=1)
