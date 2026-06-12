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
