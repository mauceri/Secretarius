# tests/test_kb_tags.py
"""Tests pour kb_tags.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_tag_vecs(
    groups: list[list[str]],
    isolated: list[str],
    dim: int = 8,
) -> dict[str, np.ndarray]:
    """
    Crée des vecteurs synthétiques L2-normalisés.
    Tags d'un même groupe : proches (base + bruit faible).
    Tags isolés : aléatoires.
    """
    rng = np.random.default_rng(0)
    vecs: dict[str, np.ndarray] = {}
    for g_idx, group in enumerate(groups):
        base = np.zeros(dim, dtype=np.float32)
        base[g_idx % dim] = 1.0
        for tag in group:
            v = base + rng.standard_normal(dim).astype(np.float32) * 0.02
            v /= np.linalg.norm(v)
            vecs[tag] = v
    for tag in isolated:
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        vecs[tag] = v
    return vecs


def test_build_tag_groups_synonyms():
    """Tags du même groupe → même canonique."""
    from kb_tags import build_tag_groups
    tags = {"python": 10, "Python": 8, "cuisine": 5, "Cuisine": 3}
    vecs = _make_tag_vecs([["python", "Python"], ["cuisine", "Cuisine"]], [])
    result = build_tag_groups(tags, vecs, threshold=0.90)
    canonical_map = {v: k for k, variants in result.items() for v in variants}
    assert canonical_map["python"] == canonical_map["Python"]
    assert canonical_map["cuisine"] == canonical_map["Cuisine"]
    assert canonical_map["python"] != canonical_map["cuisine"]


def test_build_tag_groups_hapax_excluded():
    """Hapaxes (count=1) exclus avec min_count=2."""
    from kb_tags import build_tag_groups
    tags = {"rare": 1, "commun": 5}
    vecs = _make_tag_vecs([], ["rare", "commun"])
    result = build_tag_groups(tags, vecs, threshold=0.90, min_count=2)
    all_variants = [v for variants in result.values() for v in variants]
    assert "rare" not in all_variants
    assert "commun" in all_variants


def test_build_tag_groups_keep_hapax():
    """min_count=1 → hapaxes conservés."""
    from kb_tags import build_tag_groups
    tags = {"rare": 1, "commun": 5}
    vecs = _make_tag_vecs([], ["rare", "commun"])
    result = build_tag_groups(tags, vecs, threshold=0.90, min_count=1)
    all_variants = [v for variants in result.values() for v in variants]
    assert "rare" in all_variants


def test_save_tag_dict(tmp_path):
    """save_tag_dict écrit tags_dict.json et tags_embeddings.npy."""
    from kb_tags import save_tag_dict
    kb_dir = tmp_path / "kb"
    groups = {"python": ["python", "Python"], "cuisine": ["cuisine"]}
    vecs = {
        "python": np.array([1.0, 0.0], dtype=np.float32),
        "Python": np.array([0.99, 0.01], dtype=np.float32),
        "cuisine": np.array([0.0, 1.0], dtype=np.float32),
    }
    save_tag_dict(kb_dir, groups, vecs)
    assert (kb_dir / "tags" / "tags_dict.json").exists()
    assert (kb_dir / "tags" / "tags_embeddings.npy").exists()
    d = json.loads((kb_dir / "tags" / "tags_dict.json").read_text(encoding="utf-8"))
    assert "python" in d
    assert "Python" in d["python"]
