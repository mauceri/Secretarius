"""
Répare les pages src- dont les tags ont été corrompus par l'ancien bug de
_parse_raw_tags (tags: [a, b] mal découpé — voir ingest.py, corrigé
séparément pour les nouvelles ingestions).

Deux motifs réparés :
  - Motif A : un tag entre crochets dupliquant un tag déjà présent
    (ex. ["christianisme", "[christianisme]"] → ["christianisme"])
  - Motif B : liste coupée en fragments par la virgule à l'intérieur des
    crochets (ex. ["[documentation", "secretarius]"] → ["documentation", "secretarius"])

Usage :
    python fix_corrupted_tags.py [--wiki PATH] [--apply]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import frontmatter

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"


def _looks_corrupted(tags: list) -> bool:
    """Vérifie si les tags contiennent des motifs de corruption : crochets littéraux ou listes imbriquées."""
    for t in tags:
        if isinstance(t, list):
            return True  # Listes imbriquées = corruption
        if isinstance(t, str) and (t.startswith("[") or t.endswith("]")):
            return True  # Crochets littéraux = corruption
    return False


def clean_tags(tags: list) -> list[str]:
    """Retire les crochets superflus de chaque tag, déduplique en préservant l'ordre."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        # Gérer les listes imbriquées (malformation YAML)
        if isinstance(tag, list):
            for subtag in tag:
                clean = str(subtag).strip().lstrip("[").rstrip("]").strip()
                if clean and clean not in seen:
                    cleaned.append(clean)
                    seen.add(clean)
        else:
            clean = str(tag).strip().lstrip("[").rstrip("]").strip()
            if clean and clean not in seen:
                cleaned.append(clean)
                seen.add(clean)
    return cleaned


def fix_pages(wiki_dir: Path, apply: bool) -> list[str]:
    """Répare les pages src- aux tags corrompus. Retourne les slugs modifiés."""
    fixed: list[str] = []
    for page in sorted((wiki_dir / "sources").glob("src-*.md")):
        post = frontmatter.load(page)
        tags = list(post.get("tags", []))
        if not _looks_corrupted(tags):
            continue

        new_tags = clean_tags(tags)
        print(f"{'[dry]' if not apply else '[fix]'} {page.stem} : {tags} → {new_tags}")

        if apply:
            post["tags"] = new_tags
            page.write_text(frontmatter.dumps(post), encoding="utf-8")

        fixed.append(page.stem)

    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Répare les tags corrompus (tags: [a, b] mal découpé)")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    fixed = fix_pages(wiki_dir, apply=args.apply)

    print(f"\n{'Réparées' if args.apply else 'Seraient réparées'} : {len(fixed)}")
    if not args.apply:
        print("Relancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
