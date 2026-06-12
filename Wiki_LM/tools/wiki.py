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
