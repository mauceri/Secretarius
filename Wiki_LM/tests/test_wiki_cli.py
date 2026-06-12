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


def test_query_returns_synthesis(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = "Synthèse."
        references = ["src-a"]

    class _Q:
        def __init__(self, *a, **k):
            pass

        def query(self, q, top_k=5):
            return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    out = wiki.op_query("question ?")
    assert out == {"synthesis": "Synthèse.", "references": ["src-a"]}


def test_query_empty_kb(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = ""
        references = []

    class _Q:
        def __init__(self, *a, **k):
            pass

        def query(self, q, top_k=5):
            return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    assert "error" in wiki.op_query("q")


def test_status_empty(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_status()
    assert out["running"] is False
    assert out["last_run"] is None
    assert out["pending"] == 0


def test_status_running_with_pending(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    wiki._write_state({"running": True, "last_run": None})
    out = wiki.op_status()
    assert out["running"] is True
    assert out["pending"] == 1


def test_status_reports_blocked_files(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "bad.url.error").write_text("https://x\n")
    out = wiki.op_status()
    assert out["blocked_files"] == ["bad.url.error"]
    assert out["pending"] == 0  # un .url.error n'est pas "pending"


def test_ingest_nothing_to_do(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    assert wiki.op_ingest() == {"status": "nothing_to_do", "queued": 0}


def test_ingest_started(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    calls = {}
    monkeypatch.setattr(wiki.subprocess, "Popen",
                        lambda *a, **k: calls.setdefault("spawned", True))
    out = wiki.op_ingest()
    assert out["status"] == "started" and out["queued"] == 1
    assert calls.get("spawned") is True
    assert wiki._read_state()["running"] is True


def test_ingest_already_running(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    wiki._write_state({"running": True, "last_run": None})
    assert wiki.op_ingest() == {"status": "already_running"}


def test_do_ingest_writes_last_run(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _Ing:
        def __init__(self, *a, **k):
            pass

        def ingest_raw_dir(self, *a, **k):
            return ["src-x", ""]

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    st = wiki._read_state()
    assert st["running"] is False
    assert st["last_run"]["ingested"] == 1
    assert st["last_run"]["errors"] == 1
