"""
Migration ponctuelle : réorganise wiki/ plat en sous-répertoires.

  sources/     ← src-*.md
  concepts/    ← c-*.md
  entités/     ← e-*.md
  clusterings/ ← dossiers clustering-*/

Le script est idempotent : les fichiers déjà dans leurs sous-répertoires sont ignorés.

Usage :
    python tools/migrate_wiki_structure.py [--wiki PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def migrate(wiki_dir: Path, dry_run: bool = False) -> dict[str, int]:
    """
    Déplace les fichiers vers leurs sous-répertoires.
    Retourne un dict {subdir: nombre_de_fichiers_déplacés}.
    """
    moves: dict[str, list[tuple[Path, Path]]] = {
        "sources": [],
        "concepts": [],
        "entités": [],
        "clusterings": [],
    }

    # Pages .md à la racine de wiki/
    for page in sorted(wiki_dir.glob("*.md")):
        stem = page.stem
        if stem.startswith("src-"):
            moves["sources"].append((page, wiki_dir / "sources" / page.name))
        elif stem.startswith("c-"):
            moves["concepts"].append((page, wiki_dir / "concepts" / page.name))
        elif stem.startswith("e-"):
            moves["entités"].append((page, wiki_dir / "entités" / page.name))

    # Répertoires clustering-* à la racine de wiki/
    for d in sorted(wiki_dir.iterdir()):
        if d.is_dir() and d.name.startswith("clustering-"):
            moves["clusterings"].append((d, wiki_dir / "clusterings" / d.name))

    counts: dict[str, int] = {}
    for subdir, pairs in moves.items():
        dest_dir = wiki_dir / subdir
        if not dry_run:
            dest_dir.mkdir(exist_ok=True)
        count = 0
        for src, dst in pairs:
            if dst.exists():
                continue  # déjà migré
            tag = "[dry]" if dry_run else ""
            print(f"{tag} {src.name} → {subdir}/")
            if not dry_run:
                src.rename(dst)
            count += 1
        counts[subdir] = count

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migre wiki/ vers la structure en sous-répertoires")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
    )
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans déplacer")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    if not wiki_dir.exists():
        raise FileNotFoundError(f"Répertoire wiki introuvable : {wiki_dir}")

    print(f"[migrate] {'[DRY RUN] ' if args.dry_run else ''}wiki_dir={wiki_dir}")
    counts = migrate(wiki_dir, dry_run=args.dry_run)

    print("\n[migrate] Résumé :")
    for subdir, n in counts.items():
        print(f"  {subdir:12s} : {n} fichier(s) déplacé(s)")


if __name__ == "__main__":
    main()
