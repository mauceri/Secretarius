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
    # texte + URL → un seul .url combiné (note embarquée, pas de .md séparé)
    assert len(out["files"]) == 1
    fname = out["files"][0]
    assert fname.endswith(".url")
    content = (tmp_path / "raw" / fname).read_text()
    assert "https://example.com" in content
    assert "tags: a, b" in content
    assert "note libre" in content
    assert not list((tmp_path / "raw").glob("*.md"))


def test_query_returns_synthesis(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = "Synthèse."
        references = ["src-a"]
        history_slug = "20260908-120000-question"
        brief = "Résumé."

    class _Q:
        def __init__(self, *a, **k):
            pass

        def query(self, q, top_k=5):
            return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    out = wiki.op_query("question ?")
    assert out["synthesis"] == "Synthèse."
    assert out["references"] == ["src-a"]
    assert out["brief"] == "Résumé."
    # chemin relatif à la racine du coffre : <nom du dossier wiki>/historique/<slug>.md
    assert out["history_path"] == f"{tmp_path.name}/historique/20260908-120000-question.md"


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


def test_history_path_is_relative_to_vault_root_in_sandbox(monkeypatch, tmp_path):
    # Dans le sandbox, WIKI_PATH pointe directement sur /Wiki_LM : le chemin
    # rendu doit rester relatif à la racine du coffre, pas absolu.
    wiki = _wiki(monkeypatch, tmp_path)
    from pathlib import Path
    monkeypatch.setattr(wiki, "_wiki_root", lambda: Path("/Wiki_LM"))
    assert wiki._history_path("20260908-120000-question") == (
        "Wiki_LM/historique/20260908-120000-question.md")


def test_search_returns_results(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        def __init__(self, title, excerpt):
            self.title = title
            self.excerpt = excerpt

    class _S:
        def __init__(self, *a, **k):
            pass

        def search(self, q, top_k=5):
            return [_R("Titre A", "extrait A"), _R("Titre B", "extrait B")]

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.op_search("mots-clés")
    assert out == {"results": [
        {"title": "Titre A", "excerpt": "extrait A"},
        {"title": "Titre B", "excerpt": "extrait B"},
    ]}


def test_search_no_results(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _S:
        def __init__(self, *a, **k):
            pass

        def search(self, q, top_k=5):
            return []

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.op_search("mots-clés")
    assert out == {"results": []}


def test_search_exception_returns_error(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _S:
        def __init__(self, *a, **k):
            raise FileNotFoundError("wiki/ introuvable")

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.op_search("mots-clés")
    assert "error" in out


def test_main_search_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _S:
        def __init__(self, *a, **k):
            pass

        def search(self, q, top_k=5):
            return []

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.main(["search", "mots-clés"])
    assert "results" in out


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


def test_ingest_launches_detached_worker(monkeypatch, tmp_path):
    # op_ingest lance le worker détaché lui-même (un seul exec synchrone côté agent).
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    spawned = {"n": 0}
    monkeypatch.setattr(wiki, "_spawn_detached_worker",
                        lambda: spawned.__setitem__("n", spawned["n"] + 1), raising=False)
    out = wiki.op_ingest()
    assert out == {"status": "launched", "queued": 1}
    assert spawned["n"] == 1


def test_ingest_already_running_does_not_launch(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    wiki._write_state({"running": True,
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       "last_run": None})
    spawned = {"n": 0}
    monkeypatch.setattr(wiki, "_spawn_detached_worker",
                        lambda: spawned.__setitem__("n", spawned["n"] + 1), raising=False)
    assert wiki.op_ingest() == {"status": "already_running", "queued": 1}
    assert spawned["n"] == 0


def test_ingest_stale_lock_relaunches(monkeypatch, tmp_path):
    # Un verrou running vieux (worker tué) est considéré périmé -> on relance.
    from datetime import datetime, timezone, timedelta
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "raw" / "x.url").write_text("https://example.com\n")
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    wiki._write_state({"running": True, "started_at": old, "last_run": None})
    spawned = {"n": 0}
    monkeypatch.setattr(wiki, "_spawn_detached_worker",
                        lambda: spawned.__setitem__("n", spawned["n"] + 1), raising=False)
    out = wiki.op_ingest()
    assert out == {"status": "launched", "queued": 1}
    assert spawned["n"] == 1


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


def test_ingest_worker_backend_par_defaut_non_surchargé(monkeypatch, tmp_path):
    # WIKI_INGEST_LLM_BACKEND absent -> Ingestor reçoit llm=None (son propre
    # défaut LLM()/WIKI_LLM_BACKEND s'applique, comportement inchangé).
    wiki = _wiki(monkeypatch, tmp_path)
    monkeypatch.delenv("WIKI_INGEST_LLM_BACKEND", raising=False)
    captured = {}

    class _Ing:
        _MANIFEST = ".ingested"

        def __init__(self, *a, **k):
            captured.update(k)

        def _load_manifest(self):
            return {}

        def ingest_raw_dir(self, *a, **k):
            return []

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    assert captured.get("llm") is None


def test_ingest_worker_backend_surchargeable(monkeypatch, tmp_path):
    # WIKI_INGEST_LLM_BACKEND=ollama -> Ingestor reçoit une instance LLM dédiée,
    # distincte de celle utilisée par /q (WIKI_LLM_BACKEND reste inchangée).
    wiki = _wiki(monkeypatch, tmp_path)
    monkeypatch.setenv("WIKI_INGEST_LLM_BACKEND", "ollama")
    monkeypatch.setenv("WIKI_INGEST_LLM_MODEL", "qwen3:8b")
    captured = {}

    class _Ing:
        _MANIFEST = ".ingested"

        def __init__(self, *a, **k):
            captured.update(k)

        def _load_manifest(self):
            return {}

        def ingest_raw_dir(self, *a, **k):
            return []

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    llm = captured.get("llm")
    assert llm is not None
    assert llm._backend.__class__.__name__ == "_OllamaBackend"
    assert llm._backend.model == "qwen3:8b"


def test_ingest_worker_sans_fallback_configure(monkeypatch, tmp_path):
    # WIKI_INGEST_LLM_FALLBACK_BACKEND absente -> pas de repli (comportement
    # inchangé), même avec un backend d'ingestion dédié.
    wiki = _wiki(monkeypatch, tmp_path)
    monkeypatch.setenv("WIKI_INGEST_LLM_BACKEND", "ollama")
    captured = {}

    class _Ing:
        _MANIFEST = ".ingested"

        def __init__(self, *a, **k):
            captured.update(k)

        def _load_manifest(self):
            return {}

        def ingest_raw_dir(self, *a, **k):
            return []

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    assert captured["llm"]._fallback is None


def test_ingest_worker_fallback_configure(monkeypatch, tmp_path):
    # Les 4 variables WIKI_INGEST_LLM_FALLBACK_* construisent un LLM de repli
    # attaché au LLM d'ingestion — cas réel : Qwen3-8B local (Ollama) en
    # principal, proxy OpenAI local vers Qwen3-14B obfusqué (Modal) en repli.
    wiki = _wiki(monkeypatch, tmp_path)
    monkeypatch.setenv("WIKI_INGEST_LLM_BACKEND", "ollama")
    monkeypatch.setenv("WIKI_INGEST_LLM_FALLBACK_BACKEND", "openai")
    monkeypatch.setenv("WIKI_INGEST_LLM_FALLBACK_MODEL", "qwen3-14b-h128-a1-h02")
    monkeypatch.setenv("WIKI_INGEST_LLM_FALLBACK_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("WIKI_INGEST_LLM_FALLBACK_API_KEY", "local")
    captured = {}

    class _Ing:
        _MANIFEST = ".ingested"

        def __init__(self, *a, **k):
            captured.update(k)

        def _load_manifest(self):
            return {}

        def ingest_raw_dir(self, *a, **k):
            return []

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    fallback = captured["llm"]._fallback
    assert fallback is not None
    assert fallback._backend.model == "qwen3-14b-h128-a1-h02"
    assert str(fallback._backend._client.base_url).rstrip("/") == "http://127.0.0.1:8001/v1"


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


def test_op_kb_update_kb_dir_suit_wiki_path(monkeypatch, tmp_path):
    # Régression (2026-09-10) : op_kb_update passait _DEFAULT_KB_DIR
    # (kb_update.py), qui pointe en dur vers ~/Documents/Arbath/Wiki_LM,
    # ignorant WIKI_PATH — /kbupdate échouait en prod (mkdir sur un
    # répertoire Arbath inexistant dans le sandbox).
    wiki = _wiki(monkeypatch, tmp_path)
    (tmp_path / "wiki" / "clusterings" / "clustering-embeddings-transfers-0.4").mkdir(parents=True)
    captured = {}

    def fake_update_kb(**kwargs):
        captured.update(kwargs)
        return {"created": 0, "updated": 0, "excluded": 0}

    monkeypatch.setattr(wiki, "update_kb", fake_update_kb)
    result = wiki.op_kb_update()

    assert result["status"] == "ok"
    assert captured["kb_dir"] == tmp_path / "knowledge_base"


def _write_wiki_page(tmp_path, subdir: str, slug: str, sources=None):
    import frontmatter
    post = frontmatter.Post("# Test", title=slug, category=subdir.rstrip("s"))
    if sources is not None:
        post["sources"] = sources
    d = tmp_path / "wiki" / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.md").write_text(frontmatter.dumps(post), encoding="utf-8")


def test_delete_preview_ne_modifie_rien(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_wiki_page(tmp_path, "sources", "src-a")
    _write_wiki_page(tmp_path, "concepts", "c-related", sources=["src-a"])

    out = wiki.op_delete_preview("src-a")

    assert out["status"] == "ok"
    assert "src-a" in out["affected"]
    assert "c-related" in out["affected"]
    # Rien n'a bougé : essai à blanc.
    assert (tmp_path / "wiki" / "sources" / "src-a.md").exists()
    assert (tmp_path / "wiki" / "concepts" / "c-related.md").exists()


def test_delete_applique_pour_de_vrai(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_wiki_page(tmp_path, "sources", "src-a")
    _write_wiki_page(tmp_path, "concepts", "c-related", sources=["src-a"])

    out = wiki.op_delete("src-a")

    assert out["status"] == "ok"
    assert set(out["affected"]) == {"src-a", "c-related"}
    assert not (tmp_path / "wiki" / "sources" / "src-a.md").exists()
    assert (tmp_path / "wiki" / "poubelle" / "src-a.md").exists()
    assert (tmp_path / "wiki" / "poubelle" / "c-related.md").exists()


def test_delete_slug_introuvable(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_delete_preview("src-inexistant")
    assert "error" in out


def test_delete_sans_slug(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_delete_preview("")
    assert "error" in out


def test_main_delete_preview_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_wiki_page(tmp_path, "sources", "src-a")
    out = wiki.main(["delete_preview", "src-a"])
    assert out["status"] == "ok"
    assert (tmp_path / "wiki" / "sources" / "src-a.md").exists()


def test_main_delete_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_wiki_page(tmp_path, "sources", "src-a")
    out = wiki.main(["delete", "src-a"])
    assert out["status"] == "ok"
    assert not (tmp_path / "wiki" / "sources" / "src-a.md").exists()


def _write_source_page(tmp_path, slug: str, resume: bool = True, verifie=None, mtime=None):
    import frontmatter
    body = "# Test\n\n## Résumé\n\nTexte.\n" if resume else "# Note\n\nTexte verbatim.\n"
    post = frontmatter.Post(body, title=slug, category="source")
    if verifie is not None:
        post["vérifié"] = verifie
    d = tmp_path / "wiki" / "sources"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{slug}.md"
    p.write_text(frontmatter.dumps(post), encoding="utf-8")
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


def test_review_rien_en_attente(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_review()
    assert out["status"] == "empty"


def test_review_ignore_les_pages_sans_resume(monkeypatch, tmp_path):
    # Note locale verbatim (pas de ## Résumé) : jamais résumée par le LLM,
    # aucun risque à relire — hors périmètre de la file.
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-note", resume=False)
    out = wiki.op_review()
    assert out["status"] == "empty"


def test_review_ignore_les_pages_deja_verifiees(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-a", verifie=True)
    out = wiki.op_review()
    assert out["status"] == "empty"


def test_review_page_sans_champ_verifie_compte_comme_en_attente(monkeypatch, tmp_path):
    # Pages antérieures à la fonctionnalité : pas de champ du tout -> en attente.
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-a")
    out = wiki.op_review()
    assert out["status"] == "ok"
    assert out["slug"] == "src-a"
    assert "content" in out


def test_review_ordre_fifo_plus_ancienne_dabord(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-recent", verifie=False, mtime=2000)
    _write_source_page(tmp_path, "src-ancien", verifie=False, mtime=1000)
    out = wiki.op_review()
    assert out["slug"] == "src-ancien"


def test_verify_marque_la_page(monkeypatch, tmp_path):
    import frontmatter
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-a", verifie=False)
    out = wiki.op_verify("src-a")
    assert out["status"] == "ok"
    post = frontmatter.loads((tmp_path / "wiki" / "sources" / "src-a.md").read_text())
    assert post["vérifié"] is True


def test_verify_disparait_de_la_file(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-a", verifie=False)
    wiki.op_verify("src-a")
    out = wiki.op_review()
    assert out["status"] == "empty"


def test_verify_slug_introuvable(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_verify("src-inexistant")
    assert "error" in out


def test_verify_sans_slug(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.op_verify("")
    assert "error" in out


def test_main_review_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-a")
    out = wiki.main(["review", ""])
    assert out["status"] == "ok"


def test_main_verify_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    _write_source_page(tmp_path, "src-a", verifie=False)
    out = wiki.main(["verify", "src-a"])
    assert out["status"] == "ok"
