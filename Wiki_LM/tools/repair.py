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
import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lint import WikiLint
from llm import LLM
from wiki_paths import slug_to_path, subdir_for_slug


@dataclass
class RepairReport:
    family: str
    dry_run: bool
    changes: list[str] = field(default_factory=list)
    before_count: int = 0
    after_count: int = 0


_FENCE_LINE_RE = re.compile(r"^(?:```(?:yaml|markdown)?|yaml)\s*$", re.MULTILINE)
_CLOSED_BLOCK_RE = re.compile(r"^---\n(.*?)\n---\s*\n?", re.DOTALL | re.MULTILINE)
# Tentative de frontmatter jamais refermée traînant en fin de corps nettoyé
# (ex. `---\ntitle: X\nsources: [src-a`, coupée avant la fermeture) : ne doit
# jamais être écrite telle quelle dans la page finale (finding 3, revue du
# 22/09/2026) — reste néanmoins envoyée au LLM, qui peut en tirer un titre.
_UNCLOSED_BLOCK_RE = re.compile(r"\n?---\n(?:(?!\n---\n).)*$", re.DOTALL)

_SUBDIR_TO_CATEGORY = {"sources": "source", "concepts": "concept", "entités": "entité"}

_PROMPT_REGENERATE_TITLE = """\
Voici le contenu d'une page de wiki dont le titre a été perdu (bug \
d'écriture antérieur). Réponds uniquement par un titre court (une seule \
ligne, sans guillemets, sans ponctuation finale) résumant le sujet de \
cette page — n'invente rien qui ne soit pas dans le contenu.

Contenu :
---
{content}
---

Titre :"""


def _extract_closed_frontmatter_block(body: str) -> tuple[dict, str] | None:
    """Cherche un second bloc frontmatter bien formé et fermé dans le corps
    d'une page (motif du 21-22/09/2026 : ---\\n{}\\n---\\n vide en tête,
    parfois suivi de débris de balises de code, puis un second bloc
    --- ... --- qui, lui, contient les vraies métadonnées). Retourne
    (métadonnées, reste du corps après le bloc) si ce bloc contient au
    moins title et category ; None sinon — y compris si le bloc n'est
    jamais refermé (génération interrompue)."""
    m = _CLOSED_BLOCK_RE.search(body)
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1))
    except Exception:
        return None
    if not isinstance(meta, dict) or not meta.get("title") or not meta.get("category"):
        return None
    rest = body[m.end():].strip()
    return meta, rest


def _clean_body_for_regeneration(body: str) -> str:
    """Retire les débris de balises de code (```yaml, ```markdown, ```, ou
    un « yaml » seul sur sa ligne) qui traînent dans un corps de page dont
    le frontmatter n'a pas pu être promu — sans quoi ces lignes polluent
    l'entrée envoyée au LLM. Une éventuelle tentative de frontmatter
    tronquée (ex. `---\\ntitle: ...\\nsources: [src-a`, jamais refermée)
    reste dans le résultat : elle contient souvent le titre en clair, une
    bien meilleure base pour l'extraction qu'un corps vide."""
    return _FENCE_LINE_RE.sub("", body).strip()


def _slug_to_title(slug: str) -> str:
    """Repli déterministe si le corps ne contient rien d'exploitable :
    dérive un titre lisible du slug lui-même (c-mon-concept -> "mon
    concept"). N'appelle jamais le LLM."""
    base = re.sub(r"^(?:src|c|e)-", "", slug)
    return base.replace("-", " ").strip() or slug


class WikiRepair:
    def __init__(self, wiki_path: str | Path, llm: LLM | None = None) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"
        self.llm = llm or LLM()

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

    def repair_frontmatter(self, dry_run: bool = True) -> RepairReport:
        """Répare le frontmatter vide/tronqué. Deux traitements selon la
        forme : un second bloc bien formé mais mal placé se déplace
        mécaniquement (aucun appel LLM) ; sinon, le titre est régénéré par
        le LLM à partir du corps restant — category est toujours dérivé du
        sous-répertoire, jamais deviné."""
        report_before = WikiLint(self.wiki_root).run()
        before_count = sum(1 for i in report_before.issues if i.code == "missing-frontmatter")
        slugs = sorted({i.slug for i in report_before.issues if i.code == "missing-frontmatter"})

        changes: list[str] = []
        repaired_count = 0
        for slug in slugs:
            subdir = subdir_for_slug(slug)
            path = self.wiki_dir / subdir / f"{slug}.md"
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8")
            import frontmatter as fm_module
            try:
                post = fm_module.loads(raw)
            except Exception as exc:
                # YAML syntaxiquement invalide (ex. un caractère indicateur
                # réservé en tête de scalaire) : ne jamais laisser une seule
                # page planter la méthode entière et perdre le rapport de
                # toutes les pages déjà réparées (finding 1, revue du
                # 22/09/2026). On saute cette page, on ne touche pas au
                # fichier, et on continue.
                changes.append(f"{slug} : ignorée (frontmatter illisible : {exc})")
                continue
            body = post.content

            found = _extract_closed_frontmatter_block(body)
            if found:
                meta, rest = found
                rest = _clean_body_for_regeneration(rest)
                new_content = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + f"---\n\n{rest}\n"
                changes.append(f"{slug} : bloc frontmatter bien formé déplacé")
                repaired_count += 1
            else:
                cleaned = _clean_body_for_regeneration(body)
                written_body = _UNCLOSED_BLOCK_RE.sub("", cleaned).strip()
                category = _SUBDIR_TO_CATEGORY.get(subdir, "source")
                if cleaned and not dry_run:
                    title = self.llm.complete(
                        _PROMPT_REGENERATE_TITLE.format(content=cleaned[:4000]),
                        max_tokens=100,
                    ).strip().strip('"').strip("'")
                    if not title:
                        title = _slug_to_title(slug)
                else:
                    title = _slug_to_title(slug)
                meta = {
                    "title": title,
                    "category": category,
                    "tags": [],
                    "created": datetime.date.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "sources": [],
                }
                new_content = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + f"---\n\n{written_body}\n"
                if cleaned and dry_run:
                    changes.append(
                        f"{slug} : frontmatter serait régénéré par le LLM à partir du corps "
                        "(titre non déterminé en essai à blanc)"
                    )
                else:
                    changes.append(f"{slug} : frontmatter régénéré (titre : {title!r})")
                repaired_count += 1

            if not dry_run:
                path.write_text(new_content, encoding="utf-8")

        return RepairReport(
            family="missing-frontmatter",
            dry_run=dry_run,
            changes=changes,
            before_count=before_count,
            after_count=before_count - 2 * repaired_count,
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
