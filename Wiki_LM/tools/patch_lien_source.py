"""
Patch rétroactif : ajoute lien_source aux pages src- qui en sont dépourvues.

Pour chaque page src- sans lien_source :
  1. Cherche le fichier raw correspondant via le manifeste raw/.ingested
  2. Si c'est un .url, extrait l'URL
  3. Ajoute lien_source dans le frontmatter

Usage :
    python patch_lien_source.py [--dry-run] [--wiki PATH] [--raw PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import frontmatter

from ingest import _extract_url_from_file

_DEFAULT_RAW = Path.home() / "Secretarius" / "Wiki_LM" / "raw"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch lien_source manquant sur les pages src-")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
    )
    parser.add_argument("--raw", default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)))
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans modifier")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    raw_dir = Path(args.raw)

    slug_to_raw = _load_manifest(raw_dir)

    patched = 0
    skipped_no_raw = 0
    skipped_no_url = 0

    for page in sorted(wiki_dir.glob("src-*.md")):
        try:
            post = frontmatter.load(page)
        except Exception:
            continue

        if post.get("lien_source"):
            continue

        slug = page.stem
        raw_files = slug_to_raw.get(slug, [])

        # Chercher un .url parmi les fichiers raw associés
        url = ""
        for rf in raw_files:
            if rf.suffix.lower() == ".url" and rf.exists():
                url = _extract_url_from_file(rf)
                if url:
                    break

        if not raw_files:
            skipped_no_raw += 1
            continue

        if not url:
            skipped_no_url += 1
            continue

        print(f"{'[dry]' if args.dry_run else '[patch]'} {slug}")
        print(f"  → {url[:100]}")

        if not args.dry_run:
            post["lien_source"] = url
            page.write_text(frontmatter.dumps(post), encoding="utf-8")

        patched += 1

    print(f"\n{'Seraient patchées' if args.dry_run else 'Patchées'} : {patched}")
    print(f"Sans entrée manifeste  : {skipped_no_raw}")
    print(f"Sans URL extractable   : {skipped_no_url}")


if __name__ == "__main__":
    main()
