"""Tests de la façade CLI wiki.py (agent wiki SLM)."""

import importlib


def _wiki(monkeypatch, tmp_path):
    monkeypatch.setenv("WIKI_PATH", str(tmp_path))
    (tmp_path / "raw").mkdir(exist_ok=True)
    import wiki
    importlib.reload(wiki)
    return wiki


def test_capture_url_with_tags(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_capture("#a #b https://example.com note libre")
    assert "files" in out and len(out["files"]) >= 1
    url_files = [f for f in out["files"] if f.endswith(".url")]
    assert url_files, out
    content = (tmp_path / "raw" / url_files[0]).read_text()
    assert "https://example.com" in content
    assert "tags: a, b" in content
