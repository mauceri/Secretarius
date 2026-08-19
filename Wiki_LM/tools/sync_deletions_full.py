"""
Dé-ingestion complète : synchronise les suppressions de raw/ (comme
Ingestor._sync_deletions) et permet aussi un retrait manuel d'une page
précise via --remove.

Usage :
    python sync_deletions_full.py [--wiki PATH] [--raw PATH] [--apply]
    python sync_deletions_full.py --remove src-un-slug [--apply]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ingest import Ingestor

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"
_DEFAULT_RAW = Path.home() / "Documents" / "Arbath" / "Wiki_LM" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dé-ingestion complète (index/tags/log)")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument("--raw", default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)))
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    parser.add_argument("--remove", metavar="SLUG", help="Retrait manuel d'un slug précis")
    args = parser.parse_args()

    ingestor = Ingestor(args.wiki, raw_path=args.raw)

    if args.remove:
        affected = ingestor._trash_page_full(args.remove, dry_run=not args.apply)
    else:
        affected = ingestor._sync_deletions(dry_run=not args.apply)

    print(f"\n{'Affectées' if args.apply else 'Seraient affectées'} : {len(affected)}")
    for slug in affected:
        print(f"  - {slug}")
    if not args.apply:
        print("\nRelancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
