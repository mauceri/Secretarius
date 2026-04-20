"""
Exporte les signets Brave vers raw/ pour ingestion dans Wiki_LM.

Usage :
    python tools/bookmarks_to_raw.py                        # tout sauf filtres
    python tools/bookmarks_to_raw.py --folders IA Ordinateur "GPT local"
    python tools/bookmarks_to_raw.py --folders Notable Humain
    python tools/bookmarks_to_raw.py --dry-run --folders IA  # aperçu sans écriture
    python tools/bookmarks_to_raw.py --list-folders          # liste les dossiers

Le script déduplique les URLs avant d'écrire (même URL déjà dans raw/ → ignorée).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

BOOKMARKS_DEFAULT = Path.home() / "Secretarius/Wiki_LM/bookmarks.json"
RAW_DEFAULT = Path.home() / "Secretarius/Wiki_LM/raw"

# Domaines sans valeur encyclopédique
SKIP_DOMAINS = {
    "amazon.fr", "amazon.com", "amazon.co.uk",
    "google.fr", "google.com", "accounts.google.com",
    "youtube.com", "m.youtube.com", "youtu.be",
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "linkedin.com",
    "s3.amazonaws.com",  # URLs signées temporaires, inutilisables
    "localhost", "127.0.0.1",
}

# Schémas non-HTTP
SKIP_SCHEMES = {"javascript:", "chrome:", "about:", "file:"}


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1).removeprefix("www.") if m else ""


def _normalize_url(url: str) -> str:
    """Supprime fragment et paramètres de tracking pour comparaison."""
    import urllib.parse
    _TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_content",
                 "utm_term", "triedRedirect", "inbox", "ref"}
    try:
        p = urllib.parse.urlparse(url.strip())
        qs = urllib.parse.parse_qs(p.query, keep_blank_values=True)
        qs_clean = {k: v for k, v in qs.items() if k not in _TRACKING}
        clean = p._replace(
            query=urllib.parse.urlencode(qs_clean, doseq=True),
            fragment=""
        )
        return urllib.parse.urlunparse(clean).rstrip("/")
    except Exception:
        return url.strip()


def _slugify(text: str, max_words: int = 5) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    words = [w for w in text.split() if w][:max_words]
    return "-".join(words) or "bookmark"


def _existing_urls(raw: Path) -> set[str]:
    seen = set()
    for f in raw.glob("*.url"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("url:"):
                line = line[4:].strip()
            if line.startswith("http"):
                seen.add(_normalize_url(line))
                break
    return seen


def collect_bookmarks(data: dict) -> list[tuple[str, str, str]]:
    """Retourne [(dossier, titre, url), …] pour tous les signets."""
    def _walk(node: dict, folder: str = "") -> list[tuple[str, str, str]]:
        if node.get("type") == "url":
            return [(folder, node.get("name", ""), node.get("url", ""))]
        results = []
        name = node.get("name", "")
        path = (folder + "/" + name).strip("/") if name else folder
        for child in node.get("children", []):
            results += _walk(child, path)
        return results

    items = []
    for key in ("bookmark_bar", "other", "synced"):
        items += _walk(data["roots"][key])
    return items


def filter_bookmarks(
    items: list[tuple[str, str, str]],
    folders: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Filtre par dossier(s) et domaines indésirables."""
    result = []
    for folder, title, url in items:
        # Schémas non-HTTP
        if any(url.startswith(s) for s in SKIP_SCHEMES):
            continue
        # Domaine filtré
        if _domain(url) in SKIP_DOMAINS:
            continue
        # Filtre dossier
        if folders:
            parts = folder.split("/")
            # Cherche le dossier à n'importe quel niveau
            if not any(f.strip() in parts for f in folders):
                continue
        result.append((folder, title, url))
    return result


def export_to_raw(
    items: list[tuple[str, str, str]],
    raw: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Écrit les .url dans raw/. Retourne (créés, doublons)."""
    existing = _existing_urls(raw)
    ts_base = datetime.now().strftime("%Y%m%d-%H%M%S")
    created = skipped = 0

    for i, (folder, title, url) in enumerate(items):
        norm = _normalize_url(url)
        if norm in existing:
            skipped += 1
            continue
        existing.add(norm)

        domain = _domain(url)
        domain_slug = _slugify(domain.replace(".", "-"), max_words=2)[:20]
        title_slug = _slugify(title, max_words=4)[:40]
        fname = f"{ts_base}-{i:04d}-{domain_slug}-{title_slug}.url"

        if not dry_run:
            (raw / fname).write_text(url + "\n", encoding="utf-8")
        else:
            print(f"  [dry] {fname}")
            print(f"        {url[:80]}")

        created += 1

    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporte les signets Brave vers raw/")
    parser.add_argument("--bookmarks",
                        default=os.environ.get("BRAVE_BOOKMARKS", str(BOOKMARKS_DEFAULT)))
    parser.add_argument("--raw",
                        default=os.environ.get("WIKI_RAW_PATH", str(RAW_DEFAULT)))
    parser.add_argument("--folders", nargs="+", metavar="DOSSIER",
                        help="Filtrer par dossier(s) Brave")
    parser.add_argument("--list-folders", action="store_true",
                        help="Lister les dossiers disponibles et quitter")
    parser.add_argument("--dry-run", action="store_true",
                        help="Aperçu sans écriture")
    parser.add_argument("--skip-domains", nargs="+", default=[],
                        help="Domaines supplémentaires à ignorer")
    args = parser.parse_args()

    SKIP_DOMAINS.update(args.skip_domains)

    data = json.loads(Path(args.bookmarks).read_text(encoding="utf-8"))
    items = collect_bookmarks(data)

    if args.list_folders:
        folders: Counter = Counter()
        for folder, _, _ in items:
            parts = folder.split("/")
            sub = parts[1] if len(parts) > 1 else parts[0] if parts else "(racine)"
            folders[sub] += 1
        print(f"{'Dossier':<40} {'Signets':>8}")
        print("-" * 50)
        for f, n in sorted(folders.items(), key=lambda x: -x[1]):
            print(f"{f:<40} {n:>8}")
        return

    filtered = filter_bookmarks(items, folders=args.folders)

    label = ", ".join(args.folders) if args.folders else "tous les dossiers"
    print(f"Dossier(s) : {label}")
    print(f"Signets après filtrage : {len(filtered)}")

    if args.dry_run:
        print("\nAperçu (--dry-run) :")
        for item in filtered[:20]:
            print(f"  [{item[0].split('/')[-1]}] {item[1][:60]}")
            print(f"    {item[2][:80]}")
        if len(filtered) > 20:
            print(f"  … et {len(filtered)-20} autres")
        return

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    created, skipped = export_to_raw(filtered, raw, dry_run=False)
    print(f"\nCréés  : {created} fichiers .url dans {raw}")
    print(f"Ignorés : {skipped} doublons")


if __name__ == "__main__":
    main()
