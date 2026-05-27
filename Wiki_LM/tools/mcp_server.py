"""Serveur MCP Wiki_LM — 6 outils pour le pipeline documentaire."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from wiki_paths import _load_dotenv as _load_wiki_dotenv
_load_wiki_dotenv()

from fastmcp import FastMCP
from capture import _parse_hashtags, capture_urls, capture_comment
from ingest import Ingestor
from query import WikiQuery
from kb_tags import collect_tags
from kb_update import update_kb, _DEFAULT_EMBED_DIR, _DEFAULT_KB_DIR

mcp = FastMCP("wiki-lm")

_GUARD_URL = "http://localhost:8990/check"
_FETCH_TIMEOUT = 30
_kb_update_lock = threading.Lock()


def _wiki_root() -> Path:
    """Racine du projet Wiki_LM (contient wiki/ et raw/)."""
    return Path(os.environ.get("WIKI_PATH", str(Path.home() / "Secretarius" / "Wiki_LM")))


def _raw_dir() -> Path:
    return Path(os.environ.get("WIKI_RAW_PATH", str(_wiki_root() / "raw")))


@mcp.tool()
def wiki_capture(text: str) -> dict:
    """Parse URLs et hashtags dans text et écrit des fichiers dans raw/."""
    tags, remaining = _parse_hashtags(text)
    urls = re.findall(r"https?://\S+", text)
    note = re.sub(r"https?://\S+", "", remaining).strip()

    raw = _raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    if urls:
        created.extend(capture_urls(urls, raw, tags=tags or None))
    if note:
        created.append(capture_comment(note, raw, tags=tags or None))

    return {"files": [p.name for p in created if p is not None]}


if __name__ == "__main__":
    mcp.run()
