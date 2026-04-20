"""
Pré-chauffe optionnelle du cache Wikipedia local.

Lit les pages wiki existantes, extrait les entités/concepts,
et pré-remplit wiki_cache.db via l'API Wikipedia.

Usage :
    python tools/build_wiki_cache.py
    python tools/build_wiki_cache.py --langs fr en
    python tools/build_wiki_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wiki_lookup import WikiLookup

WIKI_PATH_DEFAULT = Path.home() / "Documents/Arbath/Wiki_LM"


def _collect_names(wiki_dir: Path) -> list[str]:
    """Extrait les noms depuis les slugs e-* et c-* existants."""
    names = []
    for p in wiki_dir.glob("*.md"):
        stem = p.stem
        if stem.startswith("e-") or stem.startswith("c-"):
            # Désluggifier : tirets → espaces, capitaliser
            name = stem[2:].replace("-", " ").title()
            names.append(name)
    return sorted(set(names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-chauffe du cache Wikipedia")
    parser.add_argument("--wiki",
                        default=os.environ.get("WIKI_PATH", str(WIKI_PATH_DEFAULT)))
    parser.add_argument("--langs", nargs="+", default=["fr", "en"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Afficher les noms sans appeler l'API")
    args = parser.parse_args()

    wiki_path = Path(args.wiki)
    wiki_dir = wiki_path / "wiki"
    if not wiki_dir.exists():
        print(f"Répertoire wiki introuvable : {wiki_dir}")
        return

    names = _collect_names(wiki_dir)
    print(f"{len(names)} entités/concepts détectés dans le wiki")

    if args.dry_run:
        for n in names:
            print(f"  {n}")
        return

    wl = WikiLookup(wiki_path)
    found = skipped = missing = 0

    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name}…", end=" ", flush=True)
        result = wl.lookup(name, langs=args.langs)
        if result:
            print(f"✓ ({result['lang']})")
            found += 1
        else:
            print("—")
            missing += 1
        time.sleep(0.1)  # courtoisie envers l'API

    wl.close()
    print(f"\nRésultat : {found} trouvés, {missing} absents de Wikipedia")


if __name__ == "__main__":
    main()
