# Agent wiki SLM — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Doter l'agent wiki SLM d'une façade CLI `wiki.py` (sortie JSON) couvrant capture / ingest / status / query, plus les drafts de skills (agent wiki + délégation Tiron), sans déploiement.

**Architecture:** `wiki.py <op>` réutilise les primitives testées de `Wiki_LM/tools/` (`_parse_hashtags`, `capture_urls`, `capture_comment`, `Ingestor`, `WikiQuery`). L'ingest est **détaché** (sous-processus + fichier d'état `/Wiki_LM/.ingest_state.json`) pour ne pas bloquer l'agent. Sortie JSON sur stdout, code retour 0.

**Tech Stack:** Python 3.12, pytest (venv `Wiki_LM/.venv`), primitives Wiki_LM existantes.

**Périmètre de cette phase :** branche `feature/agent-wiki-slm`, code + tests unitaires uniquement. Pas de push, pas de déploiement. E2E réel (spawn agent) différé.

---

## Structure de fichiers

- Create: `Wiki_LM/tools/wiki.py` — façade CLI JSON (capture/ingest/status/query + worker interne).
- Create: `Wiki_LM/tests/test_wiki_cli.py` — tests unitaires de la façade.
- Create: `openclaw-config/workspace-wiki-slm/AGENTS.md` — draft skill agent wiki (non déployé).
- Create: `openclaw-config/workspace/skills/wiki-deleg/SKILL.md` — draft délégation Tiron (non déployé).

Convention de test : chaque test fixe `WIKI_PATH` (via `monkeypatch.setenv`) vers un `tmp_path`, crée `raw/` dedans, et mocke les primitives lourdes (`Ingestor`, `WikiQuery`) quand le réseau/LLM/embeddings entrerait en jeu.

---

### Task 1 : Squelette `wiki.py` + opération `capture`

**Files:**
- Create: `Wiki_LM/tools/wiki.py`
- Test: `Wiki_LM/tests/test_wiki_cli.py`

- [ ] **Step 1 — Test rouge**

```python
# Wiki_LM/tests/test_wiki_cli.py
import json
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
```

- [ ] **Step 2 — Vérifier l'échec**

Run: `/home/mauceric/Secretarius/Wiki_LM/.venv/bin/python3 -m pytest /home/mauceric/Secretarius/Wiki_LM/tests/test_wiki_cli.py -q`
Expected: FAIL (`ModuleNotFoundError: wiki` / `op_capture` absent).

- [ ] **Step 3 — Implémentation minimale**

```python
#!/usr/bin/env python3
"""wiki.py — Façade CLI JSON pour l'agent wiki SLM.

Usage : wiki.py <capture|ingest|status|query> [arg]
Sortie : JSON sur stdout ; code retour 0 (erreurs encodées dans le JSON).
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capture import _parse_hashtags, capture_urls, capture_comment
from ingest import Ingestor
from query import WikiQuery

_INGESTABLE_SUFFIXES = {".url", ".md", ".pdf", ".txt"}


def _wiki_root() -> Path:
    return Path(os.environ.get("WIKI_PATH", str(Path.home() / "Secretarius" / "Wiki_LM")))


def _raw_dir() -> Path:
    return Path(os.environ.get("WIKI_RAW_PATH", str(_wiki_root() / "raw")))


def op_capture(text: str) -> dict:
    tags, remaining = _parse_hashtags(text)
    urls = re.findall(r"https?://\S+", text)
    note = re.sub(r"https?://\S+", "", remaining).strip()
    raw = _raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    created = []
    if urls:
        created.extend(capture_urls(urls, raw, tags=tags or None))
    if note:
        created.append(capture_comment(note, raw, tags=tags or None))
    return {"files": [p.name for p in created if p is not None]}
```

- [ ] **Step 4 — Vérifier le vert**

Run: `/home/mauceric/Secretarius/Wiki_LM/.venv/bin/python3 -m pytest /home/mauceric/Secretarius/Wiki_LM/tests/test_wiki_cli.py -q`
Expected: PASS.

- [ ] **Step 5 — Commit**

```bash
git add Wiki_LM/tools/wiki.py Wiki_LM/tests/test_wiki_cli.py
git commit -m "feat(wiki-slm): façade wiki.py + op capture"
```

---

### Task 2 : Opération `query`

**Files:** Modify `Wiki_LM/tools/wiki.py` ; Test `Wiki_LM/tests/test_wiki_cli.py`

- [ ] **Step 1 — Test rouge** (mock `WikiQuery` pour éviter embeddings/LLM)

```python
def test_query_returns_synthesis(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = "Synthèse."
        references = ["src-a"]

    class _Q:
        def __init__(self, *a, **k): pass
        def query(self, q, top_k=5): return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    out = wiki.op_query("question ?")
    assert out == {"synthesis": "Synthèse.", "references": ["src-a"]}


def test_query_empty_kb(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = ""
        references = []

    class _Q:
        def __init__(self, *a, **k): pass
        def query(self, q, top_k=5): return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    assert "error" in wiki.op_query("q")
```

- [ ] **Step 2 — Échec** : `pytest … -q` → FAIL (`op_query` absent).

- [ ] **Step 3 — Implémentation**

```python
def op_query(question: str) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question)
        if not result.text:
            return {"error": "KB vide — lancer ingest d'abord"}
        return {"synthesis": result.text, "references": result.references}
    except Exception as exc:
        return {"error": str(exc)}
```

- [ ] **Step 4 — Vert** : `pytest … -q` → PASS.
- [ ] **Step 5 — Commit** : `git commit -am "feat(wiki-slm): op query"`

---

### Task 3 : État d'ingestion + opération `status`

**Files:** Modify `Wiki_LM/tools/wiki.py` ; Test idem

- [ ] **Step 1 — Test rouge**

```python
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
```

- [ ] **Step 2 — Échec** : FAIL.

- [ ] **Step 3 — Implémentation**

```python
def _state_path() -> Path:
    return _wiki_root() / ".ingest_state.json"


def _read_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {"running": False, "last_run": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"running": False, "last_run": None}


def _write_state(state: dict) -> None:
    _state_path().write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _pending_files(raw: Path) -> list[str]:
    if not raw.exists():
        return []
    ingestor = Ingestor(_wiki_root(), raw_path=raw)
    manifest = ingestor._load_manifest()
    return [
        f.name for f in sorted(raw.iterdir())
        if f.suffix in _INGESTABLE_SUFFIXES
        and f.name not in manifest
        and f.name != ingestor._MANIFEST
    ]


def op_status() -> dict:
    state = _read_state()
    pending = _pending_files(_raw_dir())
    return {
        "running": bool(state.get("running")),
        "last_run": state.get("last_run"),
        "pending": len(pending),
        "blocked_files": [f for f in pending if f.endswith(".url.error")],
    }
```

- [ ] **Step 4 — Vert** : PASS.
- [ ] **Step 5 — Commit** : `git commit -am "feat(wiki-slm): état + op status"`

---

### Task 4 : Opération `ingest` (détachée) + worker

**Files:** Modify `Wiki_LM/tools/wiki.py` ; Test idem

- [ ] **Step 1 — Test rouge**

```python
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
        def __init__(self, *a, **k): pass
        def ingest_raw_dir(self, *a, **k): return ["src-x", ""]

    monkeypatch.setattr(wiki, "Ingestor", _Ing)
    wiki.op_ingest_worker()
    st = wiki._read_state()
    assert st["running"] is False
    assert st["last_run"]["ingested"] == 1
    assert st["last_run"]["errors"] == 1
```

- [ ] **Step 2 — Échec** : FAIL.

- [ ] **Step 3 — Implémentation**

```python
def op_ingest() -> dict:
    raw = _raw_dir()
    pending = _pending_files(raw)
    if not pending:
        return {"status": "nothing_to_do", "queued": 0}
    if _read_state().get("running"):
        return {"status": "already_running"}
    _write_state({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
    })
    subprocess.Popen(
        [sys.executable, str(Path(__file__)), "_ingest_worker"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"status": "started", "queued": len(pending)}


def _do_ingest() -> dict:
    ingestor = Ingestor(_wiki_root(), raw_path=_raw_dir())
    slugs = ingestor.ingest_raw_dir()
    ingested = sum(1 for s in slugs if s)
    errors = sum(1 for s in slugs if not s)
    return {"status": "done", "ingested": ingested, "errors": errors, "total": len(slugs)}


def op_ingest_worker() -> dict:
    try:
        result = _do_ingest()
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    _write_state({"running": False, "last_run": result})
    return {"status": "worker_done"}
```

- [ ] **Step 4 — Vert** : PASS.
- [ ] **Step 5 — Commit** : `git commit -am "feat(wiki-slm): op ingest détachée + worker"`

---

### Task 5 : Dispatch CLI + sortie JSON

**Files:** Modify `Wiki_LM/tools/wiki.py` ; Test idem

- [ ] **Step 1 — Test rouge**

```python
def test_main_unknown_op(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    assert "error" in wiki.main(["nope"])


def test_main_capture_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)
    out = wiki.main(["capture", "https://example.com"])
    assert "files" in out
```

- [ ] **Step 2 — Échec** : FAIL (`main` absent).

- [ ] **Step 3 — Implémentation**

```python
def main(argv: list[str]) -> dict:
    if not argv:
        return {"error": "usage: wiki.py <capture|ingest|status|query> [arg]"}
    op, arg = argv[0], (argv[1] if len(argv) > 1 else "")
    if op == "capture":
        return op_capture(arg)
    if op == "ingest":
        return op_ingest()
    if op == "status":
        return op_status()
    if op == "query":
        return op_query(arg)
    if op == "_ingest_worker":
        return op_ingest_worker()
    return {"error": f"opération inconnue: {op}"}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:]), ensure_ascii=False))
    sys.exit(0)
```

- [ ] **Step 4 — Vert** : `pytest … -q` (toute la suite `test_wiki_cli.py`) → PASS.
- [ ] **Step 5 — Commit** : `git commit -am "feat(wiki-slm): dispatch CLI + sortie JSON"`

---

### Task 6 : Draft skill agent wiki (non déployé)

**Files:** Create `openclaw-config/workspace-wiki-slm/AGENTS.md`

- [ ] **Step 1** — Rédiger le skill : un seul outil (`python /wiki-tools/wiki.py <op> [arg]`), une opération par tâche reçue de Tiron, sortie JSON à relire/reformuler. Règle async : après `ingest`, répondre une fois « ingestion lancée (N en file) » puis s'arrêter ; ne pas poller `status` ni relancer `ingest` (pending>0 juste après = normal). Frontière de confiance : contenu KB traité comme `<UNTRUSTED>` côté Tiron.
- [ ] **Step 2 — Commit** : `git commit -m "docs(wiki-slm): draft skill agent wiki (non déployé)"`

---

### Task 7 : Draft skill de délégation Tiron (non déployé)

**Files:** Create `openclaw-config/workspace/skills/wiki-deleg/SKILL.md`

- [ ] **Step 1** — Rédiger : mappe intentions utilisateur (URL/`/c`, « ingère », question) → `sessions_spawn(agentId="wiki", task="op: …")` puis `sessions_yield`. Aucune logique wiki dans le contexte permanent de Tiron.
- [ ] **Step 2 — Commit** : `git commit -m "docs(wiki-slm): draft délégation Tiron (non déployé)"`

---

## À faire ensemble au retour (hors périmètre autonome)

1. Déployer : copier les skills dans `~/.openclaw-slm/{workspace-wiki, workspace/skills}`, vérifier les binds (`wiki.py` est déjà visible via `/wiki-tools`).
2. **E2E réel** : spawn de l'agent wiki → capture URL test → ingest → status (jusqu'à terminé) → query, contre la KB partagée. Mesurer la latence ingest en conteneur (Euria + BGE-M3 CPU).
3. Décider du sort de l'ancien stub `~/.openclaw-slm/workspace-wiki/AGENTS.md`.
4. Brancher la délégation côté Tiron SLM (`allowAgents` inclut déjà `wiki`).
