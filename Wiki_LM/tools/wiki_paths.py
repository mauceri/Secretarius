"""
Utilitaires de navigation dans la structure du wiki.

Structure attendue :
    wiki/
      sources/      ← src-*.md
      concepts/     ← c-*.md
      entités/      ← e-*.md
      clusterings/  ← clustering-*/
      index.md, log.md, tags.md  ← méta, restent à la racine
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Charge Wiki_LM/.env dans os.environ (stdlib, pas de python-dotenv requis)."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    with env_file.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

CONTENT_SUBDIRS: list[str] = ["sources", "concepts", "entités"]
CLUSTERING_SUBDIR: str = "clusterings"

_PREFIX_TO_SUBDIR: dict[str, str] = {
    "src-": "sources",
    "c-": "concepts",
    "e-": "entités",
    "cluster-": "clusterings",
}


def subdir_for_slug(slug: str) -> str:
    """Retourne le sous-répertoire attendu pour un slug donné."""
    for prefix, subdir in _PREFIX_TO_SUBDIR.items():
        if slug.startswith(prefix):
            return subdir
    return "sources"


def slug_to_path(wiki_dir: Path, slug: str) -> Path:
    """Retourne le chemin attendu pour un slug (sans vérifier l'existence)."""
    return wiki_dir / subdir_for_slug(slug) / f"{slug}.md"


def find_page(wiki_dir: Path, slug: str) -> Path | None:
    """Retourne le chemin d'une page si elle existe, None sinon."""
    path = slug_to_path(wiki_dir, slug)
    return path if path.exists() else None


def iter_pages(
    wiki_dir: Path,
    subdirs: list[str] | None = None,
    prefix: str | None = None,
):
    """
    Itère sur les pages de contenu du wiki (hors méta, hors clusterings).

    subdirs : sous-répertoires à parcourir (défaut : CONTENT_SUBDIRS)
    prefix  : filtre par préfixe de fichier, ex. "src-", "c-"
    """
    pattern = f"{prefix}*.md" if prefix else "*.md"
    for sd in (subdirs or CONTENT_SUBDIRS):
        d = wiki_dir / sd
        if not d.exists():
            continue
        yield from sorted(d.glob(pattern))
