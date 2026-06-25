#!/usr/bin/env python3
"""wiki.py — Façade CLI JSON pour l'agent wiki SLM.

Usage : wiki.py <capture|ingest|status|query> [arg]
Sortie : JSON sur stdout ; code retour 0 (erreurs encodées dans le JSON).
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

def _bootstrap_api_key() -> None:
    # OpenClaw 6.1 efface les valeurs de secret du templating de sandbox.docker.env :
    # ${EURIA_API_KEY} arrive vide dans le conteneur. La clé est donc fournie via un
    # fichier monté (convention Docker _FILE), lu ici avant toute construction de LLM.
    key_file = os.environ.get("OPENAI_API_KEY_FILE")
    if key_file and not os.environ.get("OPENAI_API_KEY"):
        try:
            os.environ["OPENAI_API_KEY"] = Path(key_file).read_text(encoding="utf-8").strip()
        except OSError:
            pass


_bootstrap_api_key()

sys.path.insert(0, str(Path(__file__).parent))
from capture import _parse_hashtags, capture_urls, capture_comment, slugify, timestamp, _write_note
from ingest import Ingestor
from query import WikiQuery
from kb_tags import collect_tags
from kb_update import update_kb, _DEFAULT_EMBED_DIR, _DEFAULT_KB_DIR

_INGESTABLE_SUFFIXES = {".url", ".md", ".pdf", ".txt"}


def _wiki_root() -> Path:
    return Path(os.environ.get("WIKI_PATH", str(Path.home() / "Secretarius" / "Wiki_LM")))


def _raw_dir() -> Path:
    return Path(os.environ.get("WIKI_RAW_PATH", str(_wiki_root() / "raw")))


def op_capture(text: str) -> dict:
    directives = [m.group(1).lower() for m in re.finditer(r"@(\w+)", text)]
    text = re.sub(r"@\w+\s*", "", text).strip()
    tags, remaining = _parse_hashtags(text)
    urls = re.findall(r"https?://\S+", remaining)
    note = re.sub(r"https?://\S+", "", remaining).strip()
    refs = re.findall(r"\bref:(\S+)", note)
    if refs:
        note = re.sub(r"\s*\bref:\S+", "", note).strip()
    file_paths = re.findall(r"\bfile:(\S+)", note)
    if file_paths:
        note = re.sub(r"\s*\bfile:\S+", "", note).strip()
        for fpath in file_paths:
            try:
                content = Path(fpath).read_text(encoding="utf-8")
                if fpath.endswith(".md"):
                    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL).strip()
                if content:
                    note = (note + "\n\n" + content).strip() if note else content.strip()
            except OSError:
                pass
    wiki_root = _wiki_root()
    if "simple" in directives:
        sources_dir = wiki_root / "wiki" / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp()
        slug = slugify(note or (Path(refs[0]).stem if refs else "note"))
        path = sources_dir / f"src-{ts}-{slug}.md"
        _write_note(path, note, tags or None, refs or None, wiki_root)
        return {"files": [path.name], "dest": "sources"}
    raw = _raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    created = []
    if urls:
        created.extend(capture_urls(urls, raw, tags=tags or None))
    if note or refs:
        created.append(capture_comment(note, raw, tags=tags or None, refs=refs or None))
    return {"files": [p.name for p in created if p is not None]}


def op_query(question: str) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question)
        if not result.text:
            return {"error": "KB vide — lancer ingest d'abord"}
        return {"synthesis": result.text, "references": result.references}
    except Exception as exc:
        return {"error": str(exc)}


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


def _blocked_files(raw: Path) -> list[str]:
    if not raw.exists():
        return []
    return sorted(f.name for f in raw.iterdir() if f.name.endswith(".url.error"))


def op_status() -> dict:
    state = _read_state()
    raw = _raw_dir()
    return {
        "running": bool(state.get("running")),
        "last_run": state.get("last_run"),
        "pending": len(_pending_files(raw)),
        "blocked_files": _blocked_files(raw),
    }


def op_ingest() -> dict:
    # Pré-vérification seule. Le worker (_ingest_worker) est lancé en arrière-plan
    # par l'agent via exec(background:true) — l'auto-détache (subprocess.Popen) ne
    # survit pas au nettoyage de l'outil exec d'OpenClaw.
    pending = _pending_files(_raw_dir())
    if not pending:
        return {"status": "nothing_to_do", "queued": 0}
    if _read_state().get("running"):
        return {"status": "already_running", "queued": len(pending)}
    return {"status": "ready", "queued": len(pending)}


def _do_ingest() -> dict:
    # ingest_raw_dir ne renvoie que les slugs des fichiers ingérés avec succès ;
    # les échecs sont marqués au manifeste mais absents de la liste. On dérive
    # donc le total des fichiers en attente avant traitement.
    raw = _raw_dir()
    queued = len(_pending_files(raw))
    ingestor = Ingestor(_wiki_root(), raw_path=raw)
    slugs = ingestor.ingest_raw_dir()
    ingested = sum(1 for s in slugs if s)
    return {"status": "done", "ingested": ingested, "errors": queued - ingested, "total": queued}


def op_ingest_worker() -> dict:
    # Lancé en arrière-plan par l'agent. Auto-gardé : pose le verrou running au
    # début, traite le batch (synchrone en interne), libère le verrou à la fin.
    if _read_state().get("running"):
        return {"status": "already_running"}
    _write_state({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
    })
    try:
        result = _do_ingest()
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    _write_state({"running": False, "last_run": result})
    return {"status": "worker_done"}


def op_tags() -> dict:
    tags = collect_tags(_wiki_root() / "wiki")
    return {"tags": sorted(tags.keys())}


def _kb_update_state() -> Path:
    return _wiki_root() / ".kb_update_state.json"


def op_kb_update() -> dict:
    wiki_dir = _wiki_root() / "wiki"
    clusterings_dir = wiki_dir / "clusterings"
    if not clusterings_dir.exists():
        return {"status": "error", "reason": "répertoire clusterings/ introuvable"}
    candidates = sorted(
        (c for c in clusterings_dir.iterdir() if c.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"status": "error", "reason": "aucun clustering disponible"}
    clustering_name = candidates[0].name
    stats = update_kb(
        wiki_root=wiki_dir,
        clustering_name=clustering_name,
        embed_dir=_DEFAULT_EMBED_DIR,
        kb_dir=_DEFAULT_KB_DIR,
    )
    return {"status": "ok", "clustering": clustering_name, **stats}


def op_kb_update_worker() -> dict:
    state = _kb_update_state()
    try:
        result = op_kb_update()
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    state.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return {"status": "worker_done"}


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
    if op == "tags":
        return op_tags()
    if op == "kb_update":
        return op_kb_update()
    if op == "_kb_update_worker":
        return op_kb_update_worker()
    if op == "_ingest_worker":
        return op_ingest_worker()
    return {"error": f"opération inconnue: {op}"}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:]), ensure_ascii=False))
    sys.exit(0)
