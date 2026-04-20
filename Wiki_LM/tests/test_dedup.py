"""Tests de déduplication dans ingest et capture."""

from pathlib import Path

import pytest

from ingest import _dedup_files, _file_hash


def _url_file(path: Path, url: str) -> Path:
    path.write_text(url + "\n", encoding="utf-8")
    return path


def _text_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestDedupFiles:
    def test_no_duplicates(self, tmp_path):
        a = _url_file(tmp_path / "a.url", "https://example.com/page1")
        b = _url_file(tmp_path / "b.url", "https://example.com/page2")
        assert _dedup_files([a, b]) == [a, b]

    def test_duplicate_url(self, tmp_path):
        a = _url_file(tmp_path / "a.url", "https://example.com/page")
        b = _url_file(tmp_path / "b.url", "https://example.com/page")
        result = _dedup_files([a, b])
        assert len(result) == 1
        assert result[0] == a  # premier conservé

    def test_duplicate_url_with_prefix(self, tmp_path):
        """Format 'url: https://...' et URL nue — même URL."""
        a = _url_file(tmp_path / "a.url", "https://example.com/page")
        b = (tmp_path / "b.url")
        b.write_text("url: https://example.com/page\nfetched: 2026-01-01\n")
        result = _dedup_files([a, b])
        assert len(result) == 1

    def test_duplicate_file_content(self, tmp_path):
        content = "Contenu identique pour les deux fichiers.\n"
        a = _text_file(tmp_path / "a.txt", content)
        b = _text_file(tmp_path / "b.txt", content)
        result = _dedup_files([a, b])
        assert len(result) == 1

    def test_different_file_content(self, tmp_path):
        a = _text_file(tmp_path / "a.txt", "Contenu A\n")
        b = _text_file(tmp_path / "b.txt", "Contenu B\n")
        assert len(_dedup_files([a, b])) == 2

    def test_url_fragment_ignored(self, tmp_path):
        """Deux URLs identiques sauf fragment → doublon."""
        a = _url_file(tmp_path / "a.url", "https://example.com/page#section1")
        b = _url_file(tmp_path / "b.url", "https://example.com/page#section2")
        result = _dedup_files([a, b])
        assert len(result) == 1

    def test_pdf_same_hash_different_name(self, tmp_path):
        """PDF copié avec nom différent (cas return-of-the-god-hypothesis)."""
        content = b"%PDF fake content " + b"x" * 1000
        a = tmp_path / "original.pdf"
        b = tmp_path / "src-original-truncated.pdf"
        a.write_bytes(content)
        b.write_bytes(content)
        result = _dedup_files([a, b])
        assert len(result) == 1

    def test_empty_list(self):
        assert _dedup_files([]) == []

    def test_manifest_file_ignored_by_caller(self, tmp_path):
        """Le fichier .ingested ne doit pas être dans la liste (filtré en amont)."""
        a = _url_file(tmp_path / "a.url", "https://example.com")
        result = _dedup_files([a])
        assert result == [a]


class TestFileHash:
    def test_same_content_same_hash(self, tmp_path):
        content = b"contenu test"
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(content)
        b.write_bytes(content)
        assert _file_hash(a) == _file_hash(b)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert _file_hash(a) != _file_hash(b)

    def test_deterministic(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"stable content")
        assert _file_hash(f) == _file_hash(f)
