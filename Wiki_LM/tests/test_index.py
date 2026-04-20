"""Tests de _update_index."""

from pathlib import Path

import pytest

from ingest import Ingestor


class TestUpdateIndex:
    def test_adds_entry(self, ingestor, wiki_dir):
        ingestor._update_index("src-test", "Test Source", "source")
        text = (wiki_dir / "index.md").read_text()
        assert "[[src-test]]" in text
        assert "source" in text
        assert "Test Source" in text

    def test_updates_existing_entry(self, ingestor, wiki_dir):
        ingestor._update_index("src-test", "Titre Original", "source")
        ingestor._update_index("src-test", "Titre Mis à Jour", "source")
        text = (wiki_dir / "index.md").read_text()
        assert "Titre Mis à Jour" in text
        assert "Titre Original" not in text
        assert text.count("[[src-test]]") == 1

    def test_multiple_entries(self, ingestor, wiki_dir):
        ingestor._update_index("src-a", "Source A", "source")
        ingestor._update_index("c-zettelkasten", "Zettelkasten", "concept")
        ingestor._update_index("e-vannevar-bush", "Vannevar Bush", "entité")
        text = (wiki_dir / "index.md").read_text()
        assert "[[src-a]]" in text
        assert "[[c-zettelkasten]]" in text
        assert "[[e-vannevar-bush]]" in text

    def test_creates_index_if_missing(self, ingestor, wiki_dir):
        (wiki_dir / "index.md").unlink()
        ingestor._update_index("src-test", "Test", "source")
        assert (wiki_dir / "index.md").exists()
