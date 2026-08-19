from __future__ import annotations

from pathlib import Path

import frontmatter


def _write_page(wiki_dir: Path, subdir: str, slug: str, sources: list[str] | None = None) -> Path:
    front = (
        "---\n"
        f"title: {slug}\n"
        f"category: {'source' if subdir == 'sources' else 'concept'}\n"
        "tags: [test]\n"
        "created: 2026-01-01\n"
    )
    if sources is not None:
        front += f"sources: {sources}\n"
    front += "---\n\n# Test\n"
    page = wiki_dir / subdir / f"{slug}.md"
    page.write_text(front, encoding="utf-8")
    return page


class TestCascadeDeletedSlugs:
    def test_returns_affected_concept_now_orphaned(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        _write_page(wiki_dir, "concepts", "c-related", sources=["src-orphan"])

        affected = ingestor._cascade_deleted_slugs({"src-orphan"}, dry_run=False)

        assert "c-related" in affected
        assert (wiki_dir / "poubelle" / "c-related.md").exists()

    def test_does_not_touch_index_tags_or_log(self, ingestor, wiki_dir: Path):
        """_cascade_deleted_slugs ne fait que la cascade c-/e- — pas de finalisation
        (c'est _finalize_deletions qui s'en charge, appelé séparément)."""
        _write_page(wiki_dir, "sources", "src-orphan")
        _write_page(wiki_dir, "concepts", "c-related", sources=["src-orphan"])
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[c-related]] | concept | Related\n", encoding="utf-8"
        )

        ingestor._cascade_deleted_slugs({"src-orphan"}, dry_run=False)

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "c-related" in index  # pas encore retiré, _finalize_deletions n'a pas été appelé
        log = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "dé-ingestion" not in log


class TestFinalizeDeletions:
    def test_removes_from_index(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[src-orphan]] | source | Orphan\n- [[src-keep]] | source | Keep\n",
            encoding="utf-8",
        )
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=False)

        ingestor._finalize_deletions(["src-orphan"], dry_run=False)

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "src-orphan" not in index
        assert "src-keep" in index

    def test_rebuilds_tags_md_excluding_trashed(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=False)

        ingestor._finalize_deletions(["src-orphan"], dry_run=False)

        tags = (wiki_dir / "tags.md").read_text(encoding="utf-8")
        assert "src-orphan" not in tags

    def test_appends_to_log(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=False)

        ingestor._finalize_deletions(["src-orphan"], dry_run=False)

        log = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "dé-ingestion" in log
        assert "src-orphan" in log

    def test_dry_run_does_not_write(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[src-orphan]] | source | Orphan\n", encoding="utf-8"
        )
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=True)

        ingestor._finalize_deletions(["src-orphan"], dry_run=True)

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "src-orphan" in index  # inchangé

    def test_keeps_a_reviser_page_in_index_but_removes_trashed(self, ingestor, wiki_dir: Path):
        """Cas mixte : une page c-/e- qui perd seulement UNE partie de ses sources
        reste sur disque en status: à-réviser et doit rester dans l'index, tandis
        qu'une page qui perd toutes ses sources est mise en poubelle et retirée."""
        _write_page(wiki_dir, "sources", "src-a")
        _write_page(wiki_dir, "sources", "src-b")
        _write_page(wiki_dir, "sources", "src-c")
        _write_page(wiki_dir, "concepts", "c-multi", sources=["src-a", "src-b"])
        _write_page(wiki_dir, "concepts", "c-single", sources=["src-c"])
        (wiki_dir / "index.md").write_text(
            "# Index\n\n"
            "- [[c-multi]] | concept | Multi\n"
            "- [[c-single]] | concept | Single\n",
            encoding="utf-8",
        )

        affected = ingestor._cascade_deleted_slugs({"src-a", "src-c"}, dry_run=False)
        ingestor._finalize_deletions(affected, dry_run=False)

        assert (wiki_dir / "concepts" / "c-multi.md").exists()
        post = frontmatter.loads((wiki_dir / "concepts" / "c-multi.md").read_text(encoding="utf-8"))
        assert post.get("status") == "à-réviser"
        assert not (wiki_dir / "concepts" / "c-single.md").exists()
        assert (wiki_dir / "poubelle" / "c-single.md").exists()

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "c-multi" in index  # toujours valide sur disque, doit rester
        assert "c-single" not in index  # réellement mis en poubelle


class TestManualRemove:
    def test_trash_page_full_removes_slug_everywhere(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-manual")
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[src-manual]] | source | Manual\n", encoding="utf-8"
        )

        affected = ingestor._trash_page_full("src-manual", dry_run=False)

        assert "src-manual" in affected
        assert (wiki_dir / "poubelle" / "src-manual.md").exists()
        assert not (wiki_dir / "sources" / "src-manual.md").exists()
        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "src-manual" not in index

    def test_trash_page_full_cascades_to_referencing_concepts(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-manual")
        _write_page(wiki_dir, "concepts", "c-related", sources=["src-manual"])

        ingestor._trash_page_full("src-manual", dry_run=False)

        assert (wiki_dir / "poubelle" / "c-related.md").exists()

    def test_trash_page_full_missing_slug_raises(self, ingestor, wiki_dir: Path):
        import pytest
        with pytest.raises(FileNotFoundError):
            ingestor._trash_page_full("src-does-not-exist", dry_run=False)
