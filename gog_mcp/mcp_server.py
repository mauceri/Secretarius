"""Serveur MCP gog — Gmail, Calendar, Drive pour Tiron."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("gog")

_GOG = "/home/linuxbrew/.linuxbrew/bin/gog"
_GUARD_URL = "http://localhost:8990/check"
_TIMEOUT = 30
_DOWNLOAD_DIR = Path.home() / "Downloads" / "gog"

_GOG_ACCOUNT = os.environ.get("GOG_ACCOUNT", "")


def _run_gog(*args: str) -> dict:
    """Lance gog avec les args donnés, retourne {"ok": True, "data": ...} ou {"ok": False, "error": ...}."""
    account_args = ["--account", _GOG_ACCOUNT] if _GOG_ACCOUNT else []
    cmd = [_GOG, *args, *account_args, "--json", "--no-input"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr.strip() or f"code {r.returncode}"}
        data = json.loads(r.stdout) if r.stdout.strip() else {}
        return {"ok": True, "data": data}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except json.JSONDecodeError:
        return {"ok": False, "error": "parse_error"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _screen(content: str) -> tuple[str, str | None]:
    """Soumet le contenu à injection-guard. Retourne (clean_text, None) ou ('', raison)."""
    try:
        payload = json.dumps({"type": "html", "content": content}).encode()
        req = urllib.request.Request(
            _GUARD_URL, data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
    except Exception:
        return "", "injection_guard indisponible — contenu non transmis"
    if result.get("blocked"):
        return "", result.get("reason", "blocked")
    clean = result.get("clean_text", "")
    if not clean:
        return "", "clean_text absent"
    return clean, None


@mcp.tool()
def gmail_unread(max: int = 10) -> dict:
    """Liste les messages Gmail non lus (métadonnées uniquement)."""
    return _run_gog("gmail", "search", "is:unread", "--max", str(max))


@mcp.tool()
def gmail_search(query: str, max: int = 10) -> dict:
    """Recherche des messages Gmail. Retourne métadonnées (id, sujet, expéditeur, date)."""
    return _run_gog("gmail", "search", query, "--max", str(max))


@mcp.tool()
def gmail_get(message_id: str) -> dict:
    """Récupère le contenu complet d'un message Gmail. ⚠ Contenu filtré par injection-guard."""
    result = _run_gog("gmail", "get", message_id)
    if not result["ok"]:
        return result
    content = json.dumps(result["data"])
    clean, reason = _screen(content)
    if reason:
        return {"ok": False, "blocked": True, "reason": reason}
    return {"ok": True, "data": result["data"], "clean_text": clean}


@mcp.tool()
def gmail_send(to: str, subject: str, body: str, cc: str = "") -> dict:
    """⚠ Demander confirmation avant d'exécuter. Envoie un email Gmail."""
    args = ["gmail", "send", "--to", to, "--subject", subject, "--body", body]
    if cc:
        args += ["--cc", cc]
    return _run_gog(*args)


@mcp.tool()
def gmail_reply(message_id: str, body: str) -> dict:
    """⚠ Demander confirmation avant d'exécuter. Répond à un message Gmail."""
    return _run_gog("gmail", "reply", message_id, "--body", body)


@mcp.tool()
def calendar_events(from_date: str, to_date: str, calendar_id: str = "primary") -> dict:
    """Liste les événements Calendar entre from_date et to_date (format ISO 8601).
    Exemple: from_date='2026-05-30T00:00:00Z', to_date='2026-05-31T00:00:00Z'"""
    return _run_gog("calendar", "events", calendar_id, "--from", from_date, "--to", to_date)


@mcp.tool()
def calendar_create(
    title: str, start: str, end: str,
    calendar_id: str = "primary", description: str = ""
) -> dict:
    """⚠ Demander confirmation avant d'exécuter. Crée un événement Calendar.
    start et end au format ISO 8601. Ex: '2026-05-30T14:00:00'"""
    args = ["calendar", "create", calendar_id,
            "--title", title, "--start", start, "--end", end]
    if description:
        args += ["--description", description]
    return _run_gog(*args)


@mcp.tool()
def calendar_delete(event_id: str, calendar_id: str = "primary") -> dict:
    """⚠ Demander confirmation avant d'exécuter. Supprime un événement Calendar."""
    return _run_gog("calendar", "delete", calendar_id, event_id)


@mcp.tool()
def drive_search(query: str, max: int = 10) -> dict:
    """Recherche des fichiers Drive. Retourne métadonnées (id, nom, type, date)."""
    return _run_gog("drive", "search", query, "--max", str(max))


@mcp.tool()
def drive_download(file_id: str, filename: str) -> dict:
    """⚠ Contenu filtré par injection-guard. Télécharge un fichier Drive dans ~/Downloads/gog/<filename>."""
    _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output = _DOWNLOAD_DIR / filename
    cmd = [_GOG, "drive", "download", file_id, "--out", str(output), "--no-input"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    try:
        content = output.read_text(errors="replace")
    except Exception as exc:
        return {"ok": False, "error": f"lecture fichier : {exc}"}
    clean, reason = _screen(content)
    if reason:
        return {"ok": False, "blocked": True, "reason": reason, "path": str(output)}
    return {"ok": True, "path": str(output), "clean_text": clean[:4000]}


@mcp.tool()
def drive_upload(file_path: str, folder_id: str = "") -> dict:
    """⚠ Demander confirmation avant d'exécuter. Dépose un fichier local sur Drive."""
    args = ["drive", "upload", file_path]
    if folder_id:
        args += ["--folder", folder_id]
    return _run_gog(*args)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8902)
