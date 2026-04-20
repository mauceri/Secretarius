"""Tests de capture.py."""

from pathlib import Path

import pytest

from capture import capture_urls, capture_comment, capture_file, _normalize_url


class TestNormalizeUrl:
    def test_removes_utm(self):
        url = "https://example.com/page?utm_source=email&utm_medium=newsletter"
        assert "utm_source" not in _normalize_url(url)

    def test_removes_fragment(self):
        url = "https://example.com/page#section"
        assert "#" not in _normalize_url(url)

    def test_removes_trailing_slash(self):
        assert _normalize_url("https://example.com/") == _normalize_url("https://example.com")

    def test_preserves_path(self):
        url = "https://example.com/article/123"
        assert "/article/123" in _normalize_url(url)

    def test_removes_tried_redirect(self):
        url = "https://substack.com/post/1?inbox=true&triedRedirect=true"
        norm = _normalize_url(url)
        assert "triedRedirect" not in norm
        assert "inbox" not in norm


class TestCaptureUrls:
    def test_single_url(self, tmp_path):
        files = capture_urls(["https://example.com/page"], tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".url"
        assert files[0].read_text().strip() == "https://example.com/page"

    def test_multiple_urls(self, tmp_path):
        urls = ["https://a.com", "https://b.com"]
        files = capture_urls(urls, tmp_path)
        assert len(files) == 2

    def test_duplicate_url_rejected(self, tmp_path):
        url = "https://example.com/page"
        capture_urls([url], tmp_path)
        files2 = capture_urls([url], tmp_path)
        assert len(files2) == 0
        assert len(list(tmp_path.glob("*.url"))) == 1

    def test_duplicate_url_tracking_params(self, tmp_path):
        """Même URL avec paramètres tracking différents → doublon."""
        capture_urls(["https://example.com/page"], tmp_path)
        files2 = capture_urls(["https://example.com/page?utm_source=email"], tmp_path)
        assert len(files2) == 0

    def test_different_urls_both_kept(self, tmp_path):
        capture_urls(["https://a.com/page1"], tmp_path)
        files2 = capture_urls(["https://a.com/page2"], tmp_path)
        assert len(files2) == 1

    def test_filename_includes_domain(self, tmp_path):
        files = capture_urls(["https://arxiv.org/abs/1234"], tmp_path)
        assert "arxiv" in files[0].name


class TestCaptureComment:
    def test_creates_md_file(self, tmp_path):
        path = capture_comment("Note sur BM25", tmp_path)
        assert path.suffix == ".md"
        assert path.exists()

    def test_content_preserved(self, tmp_path):
        text = "Réflexion sur le modèle vectoriel"
        path = capture_comment(text, tmp_path)
        assert path.read_text(encoding="utf-8").strip() == text

    def test_slug_in_filename(self, tmp_path):
        path = capture_comment("Idée importante", tmp_path)
        assert "idee" in path.name or "importante" in path.name


class TestCaptureFile:
    def test_copies_file(self, tmp_path):
        src = tmp_path / "source" / "doc.pdf"
        src.parent.mkdir()
        src.write_bytes(b"%PDF fake content")
        raw = tmp_path / "raw"
        raw.mkdir()
        dest, note = capture_file(src, raw)
        assert dest is not None
        assert dest.exists()
        assert dest.read_bytes() == src.read_bytes()

    def test_duplicate_rejected(self, tmp_path):
        src = tmp_path / "source" / "doc.pdf"
        src.parent.mkdir()
        src.write_bytes(b"%PDF fake content identical")
        raw = tmp_path / "raw"
        raw.mkdir()
        capture_file(src, raw)
        dest2, _ = capture_file(src, raw)
        assert dest2 is None
        assert len(list(raw.glob("*.pdf"))) == 1

    def test_comment_creates_md(self, tmp_path):
        src = tmp_path / "source" / "doc.pdf"
        src.parent.mkdir()
        src.write_bytes(b"%PDF content")
        raw = tmp_path / "raw"
        raw.mkdir()
        dest, note = capture_file(src, raw, comment="Article intéressant")
        assert note is not None
        assert note.suffix == ".md"
        assert "Article intéressant" in note.read_text(encoding="utf-8")
