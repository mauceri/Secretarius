"""Lecture du wiki pour la porte Grok (spec 2026-10-13) : via wiki-lm-server
local (/query, /run /r seulement) et lecture directe des pages par slug."""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

from porte.coffre import Refus

SLUG_RE = re.compile(r"^(src|c|e)-[a-z0-9-]+$")
# Une question à la fois : chaque /query sollicite le modèle (Modal), et
# l'ingestion comme les questions partagent ce même modèle.
_VERROU_QUESTION = threading.Lock()


class WikiIndisponible(Exception):
    """wiki-lm-server injoignable ou en erreur ; message renvoyé à l'agent."""


def _post(url: str, chemin: str, corps: dict, delai: float) -> dict:
    req = urllib.request.Request(
        url + chemin,
        data=json.dumps(corps).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=delai) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except ValueError:
            raise WikiIndisponible(f"wiki en erreur ({e.code})") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise WikiIndisponible("wiki injoignable") from e
    if isinstance(data, dict) and data.get("error"):
        raise WikiIndisponible(str(data["error"]))
    return data


def question(url: str, texte: str, delai: float = 360) -> dict:
    with _VERROU_QUESTION:
        data = _post(url, "/query", {"question": texte}, delai)
    return {"synthese": data.get("text", ""), "references": data.get("references", [])}


def recherche(url: str, mots: str, delai: float = 30) -> list[dict]:
    return _post(url, "/run", {"command": "/r", "arg": mots}, delai).get("results", [])


def page(wiki_dir: Path, slug: str) -> str:
    if not SLUG_RE.match(slug):
        raise Refus("slug invalide (attendu : src-…, c-… ou e-…)")
    for f in wiki_dir.rglob(f"{slug}.md"):
        if "poubelle" not in f.relative_to(wiki_dir).parts:
            return f.read_text(encoding="utf-8")
    raise Refus(f"page introuvable : {slug}")
