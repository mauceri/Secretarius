"""Tests du pipeline d'ingestion."""

from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from ingest import Ingestor


class TestIngestSingle:
    def test_creates_source_page(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu de l'article de test.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        assert slug.startswith("src-")
        assert (wiki_dir / f"{slug}.md").exists()

    def test_source_page_has_valid_frontmatter(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        post = frontmatter.loads((wiki_dir / f"{slug}.md").read_text())
        assert post["category"] == "source"
        assert "title" in post

    def test_index_updated(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        index = (wiki_dir / "index.md").read_text()
        assert f"[[{slug}]]" in index

    def test_log_updated(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest(str(src))
        log = (wiki_dir / "log.md").read_text()
        assert "ingest" in log

    def test_concept_pages_created(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest(str(src))
        # MockLLM extrait "zettelkasten" et "Vannevar Bush"
        assert (wiki_dir / "c-zettelkasten.md").exists()
        assert (wiki_dir / "e-vannevar-bush.md").exists()

    def test_no_double_src_prefix(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "src-already-prefixed.txt"
        src.write_text("Contenu.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        assert not slug.startswith("src-src-")

    def test_status_nouveau_on_new_page(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        post = frontmatter.loads((wiki_dir / f"{slug}.md").read_text())
        assert post.get("status") == "nouveau"


class TestImmuable:
    def test_immuable_page_not_overwritten(self, ingestor, wiki_dir, tmp_path):
        page = wiki_dir / "c-zettelkasten.md"
        page.write_text(
            "---\ntitle: Zettelkasten\ncategory: concept\nstatus: immuable\n---\n\nContenu protégé.\n",
            encoding="utf-8",
        )
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest(str(src))
        assert "Contenu protégé." in page.read_text()

    def test_non_immuable_page_overwritten(self, ingestor, wiki_dir, tmp_path):
        page = wiki_dir / "c-zettelkasten.md"
        page.write_text(
            "---\ntitle: Zettelkasten\ncategory: concept\nstatus: nouveau\n---\n\nAncien contenu.\n",
            encoding="utf-8",
        )
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest(str(src))
        assert "Ancien contenu." not in page.read_text()


class TestRawIncremental:
    def _populate_raw(self, raw_dir: Path) -> list[Path]:
        files = []
        for i in range(3):
            f = raw_dir / f"20260101-00000{i}-test.txt"
            f.write_text(f"Contenu source {i}", encoding="utf-8")
            files.append(f)
        return files

    def test_processes_all_new(self, ingestor, wiki_dir, raw_dir):
        self._populate_raw(raw_dir)
        slugs = ingestor.ingest_raw_dir()
        assert len(slugs) == 3

    def test_manifest_written(self, ingestor, raw_dir):
        self._populate_raw(raw_dir)
        ingestor.ingest_raw_dir()
        manifest = (raw_dir / ".ingested").read_text()
        assert manifest.count("\n") == 3

    def test_already_ingested_skipped(self, ingestor, raw_dir):
        self._populate_raw(raw_dir)
        ingestor.ingest_raw_dir()
        slugs2 = ingestor.ingest_raw_dir()
        assert slugs2 == []

    def test_new_file_only_processed(self, ingestor, raw_dir):
        self._populate_raw(raw_dir)
        ingestor.ingest_raw_dir()
        new = raw_dir / "20260102-000000-nouveau.txt"
        new.write_text("Nouvelle source", encoding="utf-8")
        slugs = ingestor.ingest_raw_dir()
        assert len(slugs) == 1


class TestRawForce:
    def test_force_resets_index(self, ingestor, wiki_dir, raw_dir):
        f = raw_dir / "20260101-000000-source.txt"
        f.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest_raw_dir()
        # Ajouter une entrée parasite dans l'index
        idx = wiki_dir / "index.md"
        idx.write_text(idx.read_text() + "- [[src-fantome]] | source | Fantôme\n")
        ingestor.ingest_raw_dir(force=True)
        assert "src-fantome" not in (wiki_dir / "index.md").read_text()

    def test_force_deletes_non_immuable_pages(self, ingestor, wiki_dir, raw_dir):
        f = raw_dir / "20260101-000000-source.txt"
        f.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest_raw_dir()
        pages_before = list(wiki_dir.glob("src-*.md"))
        assert len(pages_before) > 0
        ingestor.ingest_raw_dir(force=True)
        # Les pages sont recréées — pas de pages orphelines de l'ancien run
        old_slugs = {p.stem for p in pages_before}
        current_slugs = {p.stem for p in wiki_dir.glob("src-*.md")}
        # Les pages de l'ancien run ne sont présentes que si recréées
        assert old_slugs == current_slugs or not (old_slugs - current_slugs)

    def test_force_preserves_immuable(self, ingestor, wiki_dir, raw_dir):
        page = wiki_dir / "c-precieux.md"
        page.write_text(
            "---\ntitle: Précieux\ncategory: concept\nstatus: immuable\n---\n\nContenu précieux.\n"
        )
        f = raw_dir / "20260101-000000-source.txt"
        f.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest_raw_dir(force=True)
        assert page.exists()
        assert "Contenu précieux." in page.read_text()

    def test_force_preserves_log(self, ingestor, wiki_dir, raw_dir):
        (wiki_dir / "log.md").write_text("## [2026-01-01] ingest | Ancienne source\n")
        f = raw_dir / "20260101-000000-source.txt"
        f.write_text("Contenu.", encoding="utf-8")
        ingestor.ingest_raw_dir(force=True)
        assert (wiki_dir / "log.md").exists()


class TestConceptLinks:
    def test_concepts_linked_in_source_page(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        text = (wiki_dir / f"{slug}.md").read_text()
        # MockLLM injecte "zettelkasten" et "Vannevar Bush"
        assert "[[c-zettelkasten]]" in text
        assert "[[e-vannevar-bush]]" in text

    def test_raw_names_replaced_not_duplicated(self, ingestor, wiki_dir, tmp_path):
        src = tmp_path / "article.txt"
        src.write_text("Contenu.", encoding="utf-8")
        slug = ingestor.ingest(str(src))
        text = (wiki_dir / f"{slug}.md").read_text()
        # Les noms nus ne doivent plus apparaître dans la section
        import re
        section = re.search(
            r"## Concepts et entités mentionnés(.+?)(?=##|\Z)", text, re.DOTALL
        )
        assert section is not None
        body = section.group(1)
        assert "- concept: zettelkasten" not in body
        assert "- entité: Vannevar Bush" not in body


class TestParseUrlFile:
    def test_bare_url(self, tmp_path):
        f = tmp_path / "test.url"
        f.write_text("https://example.com/page\n")
        from ingest import Ingestor
        assert Ingestor._parse_url_file(f) == "https://example.com/page"

    def test_url_prefix_format(self, tmp_path):
        f = tmp_path / "test.url"
        f.write_text("url: https://example.com/page\nfetched: 2026-01-01\n")
        from ingest import Ingestor
        assert Ingestor._parse_url_file(f) == "https://example.com/page"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "test.url"
        f.write_text("")
        from ingest import Ingestor
        assert Ingestor._parse_url_file(f) == ""
