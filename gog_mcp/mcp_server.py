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


if __name__ == "__main__":
    mcp.run()
