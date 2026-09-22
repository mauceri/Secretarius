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


class TestVaultMirrors:
    def test_parses_single_entry(self, monkeypatch, tmp_path):
        from wiki_paths import vault_mirrors
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={tmp_path}")
        assert vault_mirrors() == {"Arbath": tmp_path}

    def test_parses_multiple_entries(self, monkeypatch, tmp_path):
        from wiki_paths import vault_mirrors
        a, b = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={a},Autre={b}")
        assert vault_mirrors() == {"Arbath": a, "Autre": b}

    def test_empty_when_unset(self, monkeypatch):
        from wiki_paths import vault_mirrors
        monkeypatch.delenv("WIKI_VAULT_MIRRORS", raising=False)
        assert vault_mirrors() == {}

    def test_ignores_malformed_pairs(self, monkeypatch, tmp_path):
        from wiki_paths import vault_mirrors
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"sans-egal,Arbath={tmp_path}")
        assert vault_mirrors() == {"Arbath": tmp_path}


class TestMirrorPage:
    def _make_wiki(self, tmp_path):
        wiki_root = tmp_path / "canonical"
        (wiki_root / "wiki" / "concepts").mkdir(parents=True)
        page = wiki_root / "wiki" / "concepts" / "c-x.md"
        page.write_text("---\ntitle: X\n---\n\nContenu.", encoding="utf-8")
        return wiki_root, page

    def test_copies_page_to_known_mirror(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, page = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, "Arbath", "c-x")

        dest = mirror / "Wiki_LM" / "wiki" / "concepts" / "c-x.md"
        assert dest.read_text(encoding="utf-8") == page.read_text(encoding="utf-8")

    def test_overwrites_existing_mirror_copy(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, page = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        dest_dir = mirror / "Wiki_LM" / "wiki" / "concepts"
        dest_dir.mkdir(parents=True)
        (dest_dir / "c-x.md").write_text("ancienne version", encoding="utf-8")
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, "Arbath", "c-x")

        assert (dest_dir / "c-x.md").read_text(encoding="utf-8") == page.read_text(encoding="utf-8")

    def test_noop_when_vault_name_is_none(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, None, "c-x")

        assert not mirror.exists()

    def test_noop_when_vault_unknown(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        monkeypatch.delenv("WIKI_VAULT_MIRRORS", raising=False)

        mirror_page(wiki_root, "Coffre inconnu", "c-x")
        # Ne lève pas — c'est le seul comportement observable ici.

    def test_noop_when_source_page_missing(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        mirror_page(wiki_root, "Arbath", "c-absent")

        assert not (mirror / "Wiki_LM" / "wiki" / "concepts" / "c-absent.md").exists()

    def test_noop_when_mirror_resolves_to_canonical(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, page = self._make_wiki(tmp_path)
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Secretarius={wiki_root}")

        mirror_page(wiki_root, "Secretarius", "c-x")

        # Rien d'autre à vérifier que l'absence de doublon écrit hors de wiki_root :
        # aucun répertoire "Wiki_LM" imbriqué ne doit apparaître sous wiki_root.
        assert not (wiki_root / "Wiki_LM").exists()

    def test_write_failure_is_silently_ignored(self, monkeypatch, tmp_path):
        from wiki_paths import mirror_page
        wiki_root, _ = self._make_wiki(tmp_path)
        mirror = tmp_path / "mirror-vault"
        monkeypatch.setenv("WIKI_VAULT_MIRRORS", f"Arbath={mirror}")

        def boom(*a, **k):
            raise OSError("disque plein")

        monkeypatch.setattr(Path, "write_text", boom)
        mirror_page(wiki_root, "Arbath", "c-x")  # ne lève pas
