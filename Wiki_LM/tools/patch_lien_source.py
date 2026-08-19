"""
Patch rétroactif : ajoute lien_source aux pages src- qui en sont dépourvues.

Pour chaque page src- sans lien_source :
  1. Cherche le(s) fichier(s) raw correspondant(s) via le manifeste raw/.ingested
  2. Tente d'en extraire une URL (n'importe quel type de fichier raw, pas
     seulement .url — un fichier .md capturé peut aussi contenir une URL)
  3. Ajoute lien_source dans le frontmatter

Usage :
    python patch_lien_source.py [--apply] [--wiki PATH] [--raw PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import frontmatter

from ingest import _extract_url_from_file

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"
_DEFAULT_RAW = Path.home() / "Documents" / "Arbath" / "Wiki_LM" / "raw"
_MANIFEST = ".ingested"


def _load_manifest(raw_dir: Path) -> dict[str, list[Path]]:
    """Retourne {slug: [fichiers raw correspondants]}."""
    path = raw_dir / _MANIFEST
    if not path.exists():
        return {}
    result: dict[str, list[Path]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        filename = parts[0]
        slug = parts[1].strip() if len(parts) > 1 else ""
        if not slug:
            continue
        raw_file = raw_dir / filename
        result.setdefault(slug, []).append(raw_file)
    return result


def patch_pages(wiki_dir: Path, raw_dir: Path, apply: bool) -> int:
    """Applique le patch sur wiki_dir/sources/src-*.md. Retourne le nombre de pages patchées."""
    slug_to_raw = _load_manifest(raw_dir)
    patched = 0

    for page in sorted((wiki_dir / "sources").glob("src-*.md")):
        post = frontmatter.load(page)
        if post.get("lien_source"):
            continue

        slug = page.stem
        raw_files = slug_to_raw.get(slug, [])

        url = ""
        for rf in raw_files:
            if rf.exists():
                url = _extract_url_from_file(rf)
                if url:
                    break

        if not url:
            continue

        print(f"{'[dry]' if not apply else '[patch]'} {slug}")
        print(f"  → {url[:100]}")

        if apply:
            post["lien_source"] = url
            page.write_text(frontmatter.dumps(post), encoding="utf-8")

        patched += 1

    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch lien_source manquant sur les pages src-")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)),
    )
    parser.add_argument("--raw", default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)))
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    raw_dir = Path(args.raw)

    patched = patch_pages(wiki_dir, raw_dir, apply=args.apply)

    print(f"\n{'Patchées' if args.apply else 'Seraient patchées'} : {patched}")
    if not args.apply:
        print("Relancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
