"""Tests de la façade CLI wiki.py (agent wiki SLM)."""

import importlib
import os


def _wiki(monkeypatch, tmp_path):
    monkeypatch.setenv("WIKI_PATH", str(tmp_path))
    (tmp_path / "raw").mkdir(exist_ok=True)
    import wiki
    importlib.reload(wiki)
    return wiki


def test_bootstrap_api_key_from_file(monkeypatch, tmp_path):
    key_file = tmp_path / "euria-key"
    key_file.write_text("secret-xyz\n")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _wiki(monkeypatch, tmp_path)
    assert os.environ["OPENAI_API_KEY"] == "secret-xyz"


def test_bootstrap_keeps_existing_api_key(monkeypatch, tmp_path):
    key_file = tmp_path / "euria-key"
    key_file.write_text("from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    _wiki(monkeypatch, tmp_path)
    assert os.environ["OPENAI_API_KEY"] == "already-set"


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


def test_ingest_ready_without_launching(monkeypatch, tmp_path):
    # op_ingest = pré-vérification seule : ne lance rien, ne pose pas le verrou.
    # Le lancement du worker en arrière-plan est fait par l'agent (exec background).
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    out = wiki.op_ingest()
    assert out == {"status": "ready", "queued": 1}
    assert wiki._read_state()["running"] is False


def test_ingest_already_running(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    wiki._write_state({"running": True, "last_run": None})
    assert wiki.op_ingest() == {"status": "already_running", "queued": 1}


def test_ingest_worker_skips_when_already_running(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    wiki._write_state({"running": True, "last_run": "sentinel"})
    called = {"ingest": False}

    class _Ing:
        _MANIFEST = ".ingested"

        def __init__(self, *a, **k):
            pass

        def _load_manifest(self):
            return {}

        def ingest_raw_dir(self, *a, **k):
            called["ingest"] = True
            return []

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    assert wiki.op_ingest_worker() == {"status": "already_running"}
    assert called["ingest"] is False
    assert wiki._read_state()["last_run"] == "sentinel"


def test_do_ingest_writes_last_run(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    # Deux fichiers en attente ; ingest_raw_dir ne renvoie que les succès
    # (les échecs n'apparaissent pas dans la liste — vrai contrat d'ingest.py).
    (tmp_path / "raw" / "a.url").write_text("https://a.example\n")
    (tmp_path / "raw" / "b.md").write_text("note\n")

    class _Ing:
        _MANIFEST = ".ingested"

        def __init__(self, *a, **k):
            pass

        def _load_manifest(self):
            return {}

        def ingest_raw_dir(self, *a, **k):
            return ["src-a"]  # un seul succès ; le second a échoué (absent)

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    st = wiki._read_state()
    assert st["running"] is False
    assert st["last_run"]["total"] == 2
    assert st["last_run"]["ingested"] == 1
    assert st["last_run"]["errors"] == 1


def test_main_unknown_op(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    assert "error" in wiki.main(["nope"])


def test_main_no_args(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    assert "error" in wiki.main([])


def test_main_capture_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.main(["capture", "https://example.com"])
    assert "files" in out


def test_cli_subprocess_outputs_json(monkeypatch, tmp_path):
    import subprocess as sp
    import sys as _sys
    env = {**__import__("os").environ, "WIKI_PATH": str(tmp_path)}
    (tmp_path / "raw").mkdir(exist_ok=True)
    wiki_py = __import__("pathlib").Path(__file__).parent.parent / "tools" / "wiki.py"
    r = sp.run([_sys.executable, str(wiki_py), "status"],
               capture_output=True, text=True, env=env)
    assert r.returncode == 0
    import json as _json
    data = _json.loads(r.stdout)
    assert data["running"] is False
