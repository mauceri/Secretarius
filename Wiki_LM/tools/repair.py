"""
Réparation du wiki Wiki_LM à partir des rapports de lint.py.

Deux familles, traitées indépendamment, jamais ensemble :
  - broken-link         : retire les crochets des liens cassés
                           ([[slug]] -> slug texte brut)
  - missing-frontmatter : répare le frontmatter vide/tronqué (déplacement
                           mécanique si un second bloc bien formé existe,
                           sinon régénération du titre par le LLM)

Usage CLI :
    python repair.py --broken-links              # essai à blanc
    python repair.py --broken-links --apply       # écrit réellement
    python repair.py --frontmatter [--apply]
    python repair.py --wiki /chemin --broken-links

Usage module :
    from repair import WikiRepair
    repairer = WikiRepair("/home/mauceric/Documents/Secretarius/Wiki_LM")
    report = repairer.repair_broken_links(dry_run=True)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from lint import WikiLint
from wiki_paths import slug_to_path


@dataclass
class RepairReport:
    family: str
    dry_run: bool
    changes: list[str] = field(default_factory=list)
    before_count: int = 0
    after_count: int = 0


class WikiRepair:
    def __init__(self, wiki_path: str | Path) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"

    def repair_broken_links(self, dry_run: bool = True) -> RepairReport:
        """Retire les crochets des liens cassés — [[slug]] devient slug en
        texte brut. Aucune hypothèse sur la cause du lien mort : ça
        s'applique à tout lien dont la cible n'existe pas, quelle qu'en
        soit l'origine."""
        report_before = WikiLint(self.wiki_root).run()
        before_count = sum(1 for i in report_before.issues if i.code == "broken-link")

        by_page: dict[str, list[str]] = {}
        for issue in report_before.issues:
            if issue.code == "broken-link":
                by_page.setdefault(issue.slug, []).append(issue.target)

        changes: list[str] = []
        fixed_count = 0
        for slug, targets in sorted(by_page.items()):
            path = slug_to_path(self.wiki_dir, slug)
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            updated = content
            for target in targets:
                updated = updated.replace(f"[[{target}]]", target)
            if updated != content:
                changes.append(f"{slug} : {len(targets)} lien(s) cassé(s) retiré(s)")
                fixed_count += len(targets)
                if not dry_run:
                    path.write_text(updated, encoding="utf-8")

        return RepairReport(
            family="broken-link",
            dry_run=dry_run,
            changes=changes,
            before_count=before_count,
            after_count=before_count - fixed_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Réparation du wiki Wiki_LM")
    import os
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Secretarius/Wiki_LM")),
        help="Chemin vers Wiki_LM",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--broken-links", action="store_true", help="Réparer les liens cassés")
    group.add_argument("--frontmatter", action="store_true", help="Réparer le frontmatter manquant")
    parser.add_argument("--apply", action="store_true", help="Écrire réellement (défaut : essai à blanc)")
    args = parser.parse_args()

    repairer = WikiRepair(args.wiki)
    if args.broken_links:
        report = repairer.repair_broken_links(dry_run=not args.apply)
    else:
        report = repairer.repair_frontmatter(dry_run=not args.apply)

    mode = "Essai à blanc" if report.dry_run else "Appliqué"
    print(f"{mode} — {report.family} : {report.before_count} → {report.after_count}")
    for change in report.changes:
        print(f"  {change}")


if __name__ == "__main__":
    main()
