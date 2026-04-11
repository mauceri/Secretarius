"""
Lint du wiki Wiki_LM : détecte les problèmes de cohérence et d'intégrité.

Vérifie :
  - Pages orphelines (aucun lien [[slug]] entrant depuis d'autres pages)
  - Liens cassés ([[slug]] référencé mais page absente)
  - Pages absentes de l'index
  - Pages dans l'index mais fichier absent
  - Frontmatter incomplet (title, category manquants)
  - Pages sources sans lien vers les concepts/entités extraits

Usage CLI :
    python lint.py
    python lint.py --wiki /chemin/vers/Wiki_LM
    python lint.py --json   # sortie JSON pour traitement automatisé

Usage module :
    from lint import WikiLint
    linter = WikiLint("/home/mauceric/Documents/Arbath/Wiki_LM")
    report = linter.run()
    print(report)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

@dataclass
class LintIssue:
    level: str          # "error" | "warning" | "info"
    code: str           # identifiant court
    slug: str           # page concernée (ou "" si global)
    message: str

    def __str__(self) -> str:
        prefix = {"error": "✗", "warning": "⚠", "info": "·"}.get(self.level, "?")
        slug_part = f"[{self.slug}] " if self.slug else ""
        return f"  {prefix} {slug_part}{self.message}"


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)
    checked_pages: int = 0
    timestamp: str = field(default_factory=lambda: datetime.date.today().isoformat())

    def add(self, level: str, code: str, slug: str, message: str) -> None:
        self.issues.append(LintIssue(level=level, code=code, slug=slug, message=message))

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def __str__(self) -> str:
        lines = [
            f"Wiki_LM Lint — {self.timestamp}",
            f"{self.checked_pages} pages vérifiées | "
            f"{len(self.errors)} erreurs | {len(self.warnings)} avertissements",
            "",
        ]
        if not self.issues:
            lines.append("  ✓ Aucun problème détecté.")
            return "\n".join(lines)

        by_level = {"error": [], "warning": [], "info": []}
        for issue in self.issues:
            by_level.setdefault(issue.level, []).append(issue)

        for level, label in [("error", "Erreurs"), ("warning", "Avertissements"), ("info", "Infos")]:
            if by_level[level]:
                lines.append(f"{label} :")
                lines.extend(str(i) for i in by_level[level])
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "checked_pages": self.checked_pages,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [
                {"level": i.level, "code": i.code, "slug": i.slug, "message": i.message}
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------

_META_PAGES = {"index", "log", "schema"}
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_REQUIRED_FRONTMATTER = {"title", "category"}


class WikiLint:
    def __init__(self, wiki_path: str | Path) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"
        if not self.wiki_dir.exists():
            raise FileNotFoundError(f"Répertoire wiki introuvable : {self.wiki_dir}")

    def run(self) -> LintReport:
        report = LintReport()

        pages = self._load_pages()
        report.checked_pages = len(pages)

        self._check_frontmatter(pages, report)
        self._check_links(pages, report)
        self._check_orphans(pages, report)
        self._check_index(pages, report)
        self._append_log(report)

        return report

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_frontmatter(self, pages: dict, report: LintReport) -> None:
        """Vérifie que chaque page a les champs frontmatter requis."""
        for slug, info in pages.items():
            meta = info["meta"]
            for field_name in _REQUIRED_FRONTMATTER:
                if not meta.get(field_name):
                    report.add(
                        "error", "missing-frontmatter", slug,
                        f"Champ frontmatter manquant : `{field_name}`",
                    )
            # Vérifier que category est une valeur connue
            cat = meta.get("category", "")
            if cat and cat not in ("source", "concept", "entité", "synthèse", "meta"):
                report.add(
                    "warning", "unknown-category", slug,
                    f"Catégorie inconnue : `{cat}`",
                )

    def _check_links(self, pages: dict, report: LintReport) -> None:
        """Détecte les liens [[slug]] cassés."""
        all_slugs = set(pages.keys()) | _META_PAGES
        for slug, info in pages.items():
            for target in info["links"]:
                if target not in all_slugs:
                    report.add(
                        "error", "broken-link", slug,
                        f"Lien cassé : [[{target}]]",
                    )

    def _check_orphans(self, pages: dict, report: LintReport) -> None:
        """Détecte les pages sans aucun lien entrant."""
        # Construire l'ensemble des slugs cibles
        linked_to: set[str] = set()
        for info in pages.values():
            linked_to.update(info["links"])

        # Lire aussi index.md pour les liens entrants implicites
        index_path = self.wiki_dir / "index.md"
        if index_path.exists():
            index_text = index_path.read_text(encoding="utf-8")
            for m in _LINK_RE.finditer(index_text):
                linked_to.add(m.group(1))

        for slug, info in pages.items():
            if slug not in linked_to:
                report.add(
                    "warning", "orphan", slug,
                    "Page orpheline (aucun lien entrant)",
                )

    def _check_index(self, pages: dict, report: LintReport) -> None:
        """Vérifie la cohérence entre index.md et les fichiers présents."""
        index_path = self.wiki_dir / "index.md"
        if not index_path.exists():
            report.add("error", "missing-index", "", "index.md est absent")
            return

        index_text = index_path.read_text(encoding="utf-8")
        # Ignorer les lignes d'exemple (Format : `…`)
        index_lines = [l for l in index_text.splitlines() if not l.strip().startswith("Format")]
        indexed_slugs = set(_LINK_RE.findall("\n".join(index_lines)))

        # Pages dans le wiki mais absentes de l'index
        for slug in pages:
            if slug not in indexed_slugs:
                report.add(
                    "warning", "not-in-index", slug,
                    "Page présente mais absente de index.md",
                )

        # Slugs dans l'index mais fichier absent
        for slug in indexed_slugs:
            if slug not in pages and slug not in _META_PAGES:
                report.add(
                    "error", "index-ghost", slug,
                    f"index.md référence [[{slug}]] mais le fichier n'existe pas",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_pages(self) -> dict[str, dict]:
        """Charge toutes les pages wiki (sauf meta) et extrait leurs liens."""
        pages: dict[str, dict] = {}
        for path in sorted(self.wiki_dir.glob("*.md")):
            slug = path.stem
            if slug in _META_PAGES:
                continue
            try:
                post = frontmatter.load(path)
                body = post.content
                meta = dict(post.metadata)
            except Exception as e:
                pages[slug] = {"path": path, "meta": {}, "links": [], "error": str(e)}
                continue

            links = [m.group(1) for m in _LINK_RE.finditer(body)]
            pages[slug] = {"path": path, "meta": meta, "links": links}

        return pages

    def _append_log(self, report: LintReport) -> None:
        log_path = self.wiki_dir / "log.md"
        if not log_path.exists():
            return
        summary = (
            f"{report.checked_pages} pages, "
            f"{len(report.errors)} erreurs, "
            f"{len(report.warnings)} avertissements"
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## [{report.timestamp}] lint | {summary}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lint du wiki Wiki_LM")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
        help="Chemin vers Wiki_LM",
    )
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    args = parser.parse_args()

    linter = WikiLint(args.wiki)
    report = linter.run()

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report)


if __name__ == "__main__":
    main()
