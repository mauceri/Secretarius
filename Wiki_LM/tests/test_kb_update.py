# tests/test_kb_update.py
"""Tests pour kb_update.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import frontmatter
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_archived_wiki(
    tmp_path: Path,
    n_clusters: int = 2,
    members_per_cluster: int = 5,
    dim: int = 16,
) -> tuple[Path, Path, str]:
    """
    Crée un mini-wiki archivé avec n_clusters clusters et embeddings synthétiques.
    Retourne (wiki_root, embed_dir, clustering_name).
    """
    wiki_root = tmp_path / "wiki_arch"
    wiki_root.mkdir()
    wiki_dir = wiki_root
    sources_dir = wiki_dir / "sources"
    sources_dir.mkdir(parents=True)

    all_slugs: list[str] = []
    for c in range(n_clusters):
        for m in range(members_per_cluster):
            slug = f"src-{c:02d}{m:02d}"
            all_slugs.append(slug)
            (sources_dir / f"{slug}.md").write_text(
                f"---\ntitle: Source {c}-{m}\ncategory: source\n"
                f"tags: [theme-{c}, test]\n---\n\n## Résumé\n\nTexte {c}-{m}.\n",
                encoding="utf-8",
            )

    clustering_name = "clustering-embeddings-transfers-0.500"
    clustering_dir = wiki_dir / "clusterings" / clustering_name
    clustering_dir.mkdir(parents=True)

    for c in range(n_clusters):
        member_slugs = [f"src-{c:02d}{m:02d}" for m in range(members_per_cluster)]
        members_lines = "\n".join(
            f"- [[{s}]] — Source {c}-{i}" for i, s in enumerate(member_slugs)
        )
        cluster_path = clustering_dir / f"cluster-embeddings-transfers-30-{c:04d}.md"
        cluster_path.write_text(
            f"---\ncategory: cluster\nsignal: embeddings\nalgo: transfers\n"
            f"param: 30\nmembers: {members_per_cluster}\n"
            f"paragon: src-{c:02d}00\ncreated: 2026-05-06\n---\n\n"
            f"# Thème {c}\n\nDescription du thème {c}.\n\n"
            f"## Parangon\n\n[[src-{c:02d}00]] — Source {c}-0\n\n"
            f"## Documents membres\n\n{members_lines}\n\n"
            f"## Clusters proches\n\n",
            encoding="utf-8",
        )

    rng = np.random.default_rng(0)
    total = n_clusters * members_per_cluster
    vecs = np.zeros((total, dim), dtype=np.float32)
    half = dim // 2
    for c in range(n_clusters):
        start, end = c * members_per_cluster, (c + 1) * members_per_cluster
        vecs[start:end, c * half : (c + 1) * half] = 1.0
    vecs += rng.standard_normal((total, dim)).astype(np.float32) * 0.05
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    np.save(embed_dir / "embeddings.npy", vecs)
    (embed_dir / "embeddings_index.json").write_text(
        json.dumps({"slugs": all_slugs}), encoding="utf-8"
    )

    return wiki_root, embed_dir, clustering_name


def test_update_kb_creates_axes_dir(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert (kb_dir / "axes").exists()
    assert (kb_dir / "embeddings" / "axes.npy").exists()
    assert (kb_dir / "embeddings" / "axes_index.json").exists()


def test_update_kb_creates_two_axes(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    stats = update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert stats["created"] == 2
    assert stats["updated"] == 0
    axes = list((kb_dir / "axes").glob("axis-*.md"))
    assert len(axes) == 2


def test_update_kb_axis_has_correct_frontmatter(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    axis_path = kb_dir / "axes" / "axis-0001.md"
    post = frontmatter.load(axis_path)
    assert post.get("title") == "Thème 0"
    assert post.get("members_count") == 5
    assert "wiki_arch" in post.get("source_wikis", [])
    assert isinstance(post.get("cohesion"), float)


def test_update_kb_axes_npy_shape(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    mat = np.load(kb_dir / "embeddings" / "axes.npy")
    assert mat.shape == (2, 16)


def test_update_kb_exclusion_too_small(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(
        tmp_path, n_clusters=2, members_per_cluster=2
    )
    kb_dir = tmp_path / "kb"
    stats = update_kb(wiki_root, clustering_name, embed_dir, kb_dir, min_size=3)
    assert stats["excluded"] == 2
    excluded = json.loads((kb_dir / "excluded.json").read_text(encoding="utf-8"))
    assert any("size" in e["reason"] for e in excluded)


def test_update_kb_fusion_on_second_call(tmp_path):
    """Deuxième appel avec même clustering → axes mis à jour, pas dupliqués."""
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    stats2 = update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert stats2["created"] == 0
    assert stats2["updated"] == 2
    axes = list((kb_dir / "axes").glob("axis-*.md"))
    assert len(axes) == 2
    # Vérifier que members_count a doublé
    post = frontmatter.load(kb_dir / "axes" / "axis-0001.md")
    assert post.get("members_count") == 10   # 5 + 5


def test_update_kb_source_wikis_accumulated(tmp_path):
    """source_wikis s'accumule entre les appels depuis deux wikis différents."""
    from kb_update import update_kb
    import shutil
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)

    wiki_root2 = tmp_path / "wiki_arch2"
    shutil.copytree(wiki_root, wiki_root2)
    update_kb(wiki_root2, clustering_name, embed_dir, kb_dir)

    post = frontmatter.load(kb_dir / "axes" / "axis-0001.md")
    src_wikis = post.get("source_wikis", [])
    assert len(src_wikis) == 2
    assert "wiki_arch" in src_wikis
    assert "wiki_arch2" in src_wikis


def test_update_kb_index_md_created(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert (kb_dir / "index.md").exists()
    content = (kb_dir / "index.md").read_text(encoding="utf-8")
    assert "Thème 0" in content
    assert "Thème 1" in content


def test_update_kb_missing_clustering_raises(tmp_path):
    """Clustering inexistant → FileNotFoundError."""
    from kb_update import update_kb
    wiki_root, embed_dir, _ = _make_archived_wiki(tmp_path)
    kb_dir = tmp_path / "kb"
    with pytest.raises(FileNotFoundError):
        update_kb(wiki_root, "clustering-embeddings-transfers-0.999", embed_dir, kb_dir)
