"""
Liste les pages src- sans lien_source récupérable automatiquement, pour
recherche manuelle par mots-clés.

Extraction déterministe (titre, tags, deux premières phrases du résumé) —
pas d'appel LLM, pas de recherche web automatique.

Usage :
    python list_unsourced_pages.py [--wiki PATH] [--out FICHIER]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import frontmatter

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_excerpt(body: str) -> str:
    """Retourne les deux premières phrases de la section ## Résumé, sinon du corps."""
    match = re.search(r"## Résumé\s*\n+(.+?)(\n##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else body.strip()
    sentences = _SENTENCE_RE.split(text)
    return " ".join(sentences[:2]).strip()


def collect_unsourced(wiki_dir: Path) -> list[dict]:
    """Retourne les pages src- sans lien_source, avec titre/tags/extrait."""
    result: list[dict] = []
    for page in sorted((wiki_dir / "sources").glob("src-*.md")):
        post = frontmatter.load(page)
        if post.get("lien_source"):
            continue
        # Flatten tags and ensure all are strings
        raw_tags = post.get("tags", [])
        tags = []
        for tag in raw_tags:
            if isinstance(tag, list):
                tags.extend(str(t) for t in tag)
            else:
                tags.append(str(tag))
        result.append({
            "slug": page.stem,
            "title": str(post.get("title", page.stem)),
            "tags": tags,
            "excerpt": _extract_excerpt(post.content),
        })
    return result


def format_report(entries: list[dict]) -> str:
    lines = [
        "# Pages sans source vérifiable — recherche manuelle",
        "",
        "Généré par `list_unsourced_pages.py`. Pour chaque page, chercher l'URL",
        "d'origine avec un moteur de recherche à partir du titre et des mots-clés.",
        "",
        "| Slug | Titre | Tags | Extrait |",
        "|---|---|---|---|",
    ]
    for e in entries:
        tags = ", ".join(e["tags"])
        excerpt = e["excerpt"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {e['slug']} | {e['title']} | {tags} | {excerpt} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Liste les pages src- sans lien_source")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "urls_a_rechercher.md"),
    )
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    entries = collect_unsourced(wiki_dir)
    report = format_report(entries)

    Path(args.out).write_text(report, encoding="utf-8")
    print(f"{len(entries)} page(s) sans source listée(s) dans {args.out}")


if __name__ == "__main__":
    main()
