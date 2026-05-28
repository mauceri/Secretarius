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

# État de l'ingestion asynchrone
_ingest_lock = threading.Lock()
_ingest_running = False
_ingest_result: dict | None = None  # résultat du dernier run terminé


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


def _screen(html: str) -> tuple[str, str | None]:
    """Soumet le contenu HTML à injection-guard.
    Retourne (clean_text, None) si OK, ou ("", raison) si bloqué.
    """
    try:
        payload = json.dumps({"type": "html", "content": html}).encode()
        req = urllib.request.Request(
            _GUARD_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
    except Exception:
        return "", "injection-guard unavailable"

    if result.get("blocked"):
        return "", result.get("reason", "blocked")
    clean = result.get("clean_text")
    if not clean:
        return "", "clean_text absent — réponse injection-guard incomplète"
    return clean, None


_INGESTABLE_SUFFIXES = {".url", ".md", ".pdf", ".txt"}


def _run_ingest() -> None:
    """Exécuté dans un thread de fond par wiki_ingest."""
    global _ingest_running, _ingest_result
    raw = _raw_dir()
    wiki_root = _wiki_root()
    ingestor = Ingestor(wiki_root, raw_path=raw)
    manifest = ingestor._load_manifest()

    all_pending = [
        f for f in sorted(raw.iterdir())
        if f.suffix in _INGESTABLE_SUFFIXES
        and f.name not in manifest
        and f.name != ingestor._MANIFEST
    ]

    ingested = blocked = errors = 0
    blocked_details: list[dict] = []
    error_details: list[dict] = []

    for f in all_pending:
        if f.suffix == ".url":
            url = Ingestor._parse_url_file(f)
            if not url:
                f.rename(f.parent / (f.name + ".error"))
                error_details.append({"file": f.name, "reason": "fichier .url vide"})
                errors += 1
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Wiki_LM/1.0"})
                with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
                    charset = resp.headers.get_content_charset("utf-8")
                    html = resp.read().decode(charset, errors="replace")
            except Exception as exc:
                f.rename(f.parent / (f.name + ".error"))
                error_details.append({"file": f.name, "reason": str(exc)})
                errors += 1
                continue
            clean_text, block_reason = _screen(html)
            if block_reason is not None:
                f.rename(f.parent / (f.name + ".blocked"))
                blocked_details.append({"file": f.name, "reason": block_reason})
                blocked += 1
                continue
            user_tags = Ingestor._parse_raw_tags(f)
            try:
                slug = ingestor.ingest(
                    url, content=clean_text,
                    extra_tags=user_tags or None, rename_raw=False,
                )
                ingestor._mark_ingested(f.name, slug=slug)
                ingested += 1
            except Exception as exc:
                error_details.append({"file": f.name, "reason": str(exc)})
                errors += 1
        else:
            user_tags = Ingestor._parse_raw_tags(f)
            try:
                slug = ingestor.ingest(
                    str(f), extra_tags=user_tags or None, rename_raw=False,
                )
                ingestor._mark_ingested(f.name, slug=slug)
                ingested += 1
            except Exception as exc:
                error_details.append({"file": f.name, "reason": str(exc)})
                errors += 1

    with _ingest_lock:
        _ingest_result = {
            "status": "done",
            "ingested": ingested, "blocked": blocked, "errors": errors,
            "blocked_details": blocked_details, "error_details": error_details,
        }
        _ingest_running = False


@mcp.tool()
def wiki_ingest() -> dict:
    """Lance l'ingestion en tâche de fond et retourne immédiatement. Interroger wiki_ingest_status pour le résultat."""
    global _ingest_running
    with _ingest_lock:
        if _ingest_running:
            return {"status": "already_running"}
        raw = _raw_dir()
        if not raw.exists():
            return {"status": "started", "queued": 0}
        ingestor = Ingestor(_wiki_root(), raw_path=raw)
        manifest = ingestor._load_manifest()
        queued = sum(
            1 for f in raw.iterdir()
            if f.suffix in _INGESTABLE_SUFFIXES
            and f.name not in manifest
            and f.name != ingestor._MANIFEST
        )
        if queued == 0:
            return {"status": "nothing_to_do", "queued": 0}
        _ingest_running = True
        t = threading.Thread(target=_run_ingest, daemon=True)
        t.start()
    return {"status": "started", "queued": queued}


@mcp.tool()
def wiki_query(question: str, top_k: int = 5) -> dict:
    """Interroge le wiki et retourne une synthèse avec les sources."""
    try:
        q = WikiQuery(_wiki_root())
        result = q.query(question, top_k=top_k)
        if not result.text:
            return {"error": "KB vide — lancer wiki_ingest d'abord"}
        return {"synthesis": result.text, "references": result.references}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def wiki_tags() -> dict:
    """Retourne la liste des tags du wiki."""
    try:
        tags = collect_tags(_wiki_root() / "wiki")
        return {"tags": sorted(tags.keys())}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def wiki_ingest_status() -> dict:
    """Retourne l'état de l'ingestion en cours (ou du dernier run) et les fichiers en attente/bloqués."""
    with _ingest_lock:
        running = _ingest_running
        last = _ingest_result.copy() if _ingest_result else None

    raw = _raw_dir()
    if not raw.exists():
        return {"running": running, "last_run": last, "pending": 0, "blocked_files": []}
    ingestor = Ingestor(_wiki_root(), raw_path=raw)
    manifest = ingestor._load_manifest()
    pending_files = [
        f.name for f in sorted(raw.iterdir())
        if f.suffix in _INGESTABLE_SUFFIXES
        and f.name not in manifest
        and f.name != ingestor._MANIFEST
    ]
    blocked_files = sorted(
        f.name for f in raw.iterdir() if f.name.endswith(".url.blocked")
    )
    return {
        "running": running,
        "last_run": last,
        "pending": len(pending_files),
        "blocked_files": blocked_files,
    }


@mcp.tool()
def wiki_list_pending() -> dict:
    """Liste les fichiers en attente d'ingestion dans raw/ avec leurs tags."""
    raw = _raw_dir()
    if not raw.exists():
        return {"pending": [], "blocked": [], "raw_dir": str(raw)}
    ingestor = Ingestor(_wiki_root(), raw_path=raw)
    manifest = ingestor._load_manifest()
    pending = []
    for f in sorted(raw.iterdir()):
        if f.suffix in _INGESTABLE_SUFFIXES and f.name not in manifest and f.name != ingestor._MANIFEST:
            tags = Ingestor._parse_raw_tags(f)
            pending.append({"file": f.name, "tags": tags or []})
    blocked = sorted(f.name for f in raw.iterdir() if f.name.endswith(".url.blocked"))
    return {"pending": pending, "blocked": blocked, "raw_dir": str(raw)}


@mcp.tool()
def wiki_kb_update() -> dict:
    """Lance la mise à jour du KB depuis le dernier clustering disponible."""
    if not _kb_update_lock.acquire(blocking=False):
        return {"status": "already_running"}
    try:
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
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    finally:
        _kb_update_lock.release()


if __name__ == "__main__":
    mcp.run()
