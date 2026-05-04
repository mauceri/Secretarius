"""Tests pour wiki_paths.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def test_subdir_for_src():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("src-foo-bar") == "sources"


def test_subdir_for_concept():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("c-zettelkasten") == "concepts"


def test_subdir_for_entity():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("e-bush") == "entités"


def test_subdir_for_cluster():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("cluster-embeddings-0000") == "clusterings"


def test_slug_to_path_src(tmp_path):
    from wiki_paths import slug_to_path
    wiki = tmp_path / "wiki"
    assert slug_to_path(wiki, "src-foo") == wiki / "sources" / "src-foo.md"


def test_slug_to_path_concept(tmp_path):
    from wiki_paths import slug_to_path
    wiki = tmp_path / "wiki"
    assert slug_to_path(wiki, "c-zettelkasten") == wiki / "concepts" / "c-zettelkasten.md"


def test_slug_to_path_entity(tmp_path):
    from wiki_paths import slug_to_path
    wiki = tmp_path / "wiki"
    assert slug_to_path(wiki, "e-bush") == wiki / "entités" / "e-bush.md"


def test_find_page_exists(tmp_path):
    from wiki_paths import find_page
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "src-foo.md").write_text("---\ntitle: Foo\n---\n", encoding="utf-8")
    p = find_page(wiki, "src-foo")
    assert p == wiki / "sources" / "src-foo.md"


def test_find_page_not_exists(tmp_path):
    from wiki_paths import find_page
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    assert find_page(wiki, "src-nonexistent") is None


def test_iter_pages_all_subdirs(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    for sd in ("sources", "concepts", "entités"):
        (wiki / sd).mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("a", encoding="utf-8")
    (wiki / "concepts" / "c-b.md").write_text("b", encoding="utf-8")
    (wiki / "entités" / "e-c.md").write_text("c", encoding="utf-8")
    (wiki / "index.md").write_text("index", encoding="utf-8")  # ne doit PAS être inclus

    paths = list(iter_pages(wiki))
    names = {p.name for p in paths}
    assert "src-a.md" in names
    assert "c-b.md" in names
    assert "e-c.md" in names
    assert "index.md" not in names
    assert len(paths) == 3


def test_iter_pages_with_prefix(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("a", encoding="utf-8")
    (wiki / "sources" / "src-b.md").write_text("b", encoding="utf-8")
    paths = list(iter_pages(wiki, prefix="src-"))
    assert len(paths) == 2


def test_iter_pages_with_subdirs(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("a", encoding="utf-8")
    (wiki / "concepts" / "c-b.md").write_text("b", encoding="utf-8")
    paths = list(iter_pages(wiki, subdirs=["concepts"]))
    assert len(paths) == 1
    assert paths[0].name == "c-b.md"


def test_iter_pages_missing_subdir(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    # Aucun sous-répertoire créé
    paths = list(iter_pages(wiki))
    assert paths == []
