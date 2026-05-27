"""Tests du serveur MCP Wiki_LM."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import mcp_server


@pytest.fixture
def wiki_env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("WIKI_PATH", str(tmp_path))
    monkeypatch.setenv("WIKI_RAW_PATH", str(raw))
    return {"root": tmp_path, "raw": raw}


class TestWikiCapture:
    def test_creates_url_file_for_url(self, wiki_env):
        result = mcp_server.wiki_capture("https://example.com")
        assert len(result["files"]) == 1
        assert result["files"][0].endswith(".url")
        url_file = wiki_env["raw"] / result["files"][0]
        assert "https://example.com" in url_file.read_text()

    def test_parses_hashtags_as_tags(self, wiki_env):
        result = mcp_server.wiki_capture("#linguistique https://example.com")
        url_file = wiki_env["raw"] / result["files"][0]
        assert "linguistique" in url_file.read_text()

    def test_creates_md_for_note_only(self, wiki_env):
        result = mcp_server.wiki_capture("Ceci est une note sans URL")
        assert len(result["files"]) == 1
        assert result["files"][0].endswith(".md")

    def test_url_deduplication(self, wiki_env):
        mcp_server.wiki_capture("https://example.com")
        result2 = mcp_server.wiki_capture("https://example.com")
        # Doublon ignoré, pas de nouveau fichier
        assert len(result2["files"]) == 0
