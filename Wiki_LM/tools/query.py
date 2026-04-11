"""
Query du wiki Wiki_LM : recherche BM25 + lecture pages + synthèse LLM.

Usage CLI :
    python query.py "Comment fonctionne Zettelkasten ?"
    python query.py "Qui est Niklas Luhmann ?" --top 3
    python query.py "…" --save  # enregistre la synthèse comme page synth-

Usage module :
    from query import WikiQuery
    wq = WikiQuery("/home/mauceric/Documents/Arbath/Wiki_LM")
    answer = wq.query("Comment fonctionne Zettelkasten ?")
    print(answer.text)
    for ref in answer.references:
        print(ref)
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from llm import LLM
from search import WikiSearch


# ---------------------------------------------------------------------------
# Résultat de query
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    question: str
    text: str                          # synthèse en Markdown
    references: list[str] = field(default_factory=list)   # slugs utilisés
    saved_slug: str = ""               # slug de la page synth- si --save

    def __str__(self) -> str:
        refs = ", ".join(f"[[{r}]]" for r in self.references)
        header = f"Q : {self.question}\n\nSources : {refs or '(aucune)'}\n\n"
        return header + self.text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_QUERY = """\
Tu es un assistant de wiki personnel (Wiki_LM). \
Tu reçois des pages Markdown extraites du wiki et tu dois synthétiser \
une réponse à la question de l'utilisateur. \
Cite les pages avec [[slug]] dans ta réponse. \
Sois factuel, concis, et base-toi uniquement sur les pages fournies. \
Si l'information n'y est pas, dis-le clairement."""

_PROMPT_QUERY = """\
Question : {question}

Pages du wiki disponibles :
---
{pages_block}
---

Produis une synthèse en Markdown qui répond à la question. \
Utilise des citations [[slug]] pour chaque affirmation issue d'une page. \
Si la réponse est incomplète faute de pages disponibles, indique ce qui manque."""

_PROMPT_SYNTH_PAGE = """\
Transforme cette synthèse en page wiki au format Markdown strict :

Question d'origine : {question}
Synthèse :
---
{synthesis}
---

Format attendu :
```yaml
---
title: <Titre court résumant la question>
category: synthèse
tags: [<tags pertinents>]
created: {today}
sources: [<slugs utilisés, séparés par des virgules>]
---
```

# <Titre>

<Corps de la synthèse, réorganisé si nécessaire>
"""


# ---------------------------------------------------------------------------
# Moteur de query
# ---------------------------------------------------------------------------

class WikiQuery:
    def __init__(self, wiki_path: str | Path, llm: LLM | None = None) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"
        self._search = WikiSearch(wiki_path)
        self.llm = llm or LLM()

    def query(self, question: str, top_k: int = 5, save: bool = False) -> QueryResult:
        """Répond à une question en cherchant dans le wiki puis en synthétisant."""

        # 1. Recherche BM25
        results = self._search.search(question, top_k=top_k)
        if not results:
            return QueryResult(
                question=question,
                text="_Aucune page pertinente trouvée dans le wiki._",
            )

        # 2. Lire le contenu complet des pages trouvées
        pages_block = self._build_pages_block(results)
        slugs = [r.slug for r in results]

        # 3. Synthèse LLM
        prompt = _PROMPT_QUERY.format(question=question, pages_block=pages_block)
        synthesis = self.llm.complete(prompt, system=_SYSTEM_QUERY, max_tokens=2000)

        # 4. Extraire les slugs effectivement cités dans la réponse
        cited = re.findall(r"\[\[([^\]]+)\]\]", synthesis)
        references = list(dict.fromkeys(cited or slugs))  # ordre de première apparition

        result = QueryResult(question=question, text=synthesis, references=references)

        # 5. Optionnel : sauvegarder comme page synth-
        if save:
            result.saved_slug = self._save_synth(question, synthesis, references)
            self._append_log("query", question)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pages_block(self, results: list) -> str:
        """Construit le bloc de texte des pages à fournir au LLM."""
        parts = []
        for r in results:
            try:
                content = r.path.read_text(encoding="utf-8")
                post = frontmatter.loads(content)
                body = post.content
            except Exception:
                body = r.excerpt

            parts.append(f"### [[{r.slug}]] — {r.title}\n\n{body}")

        return "\n\n---\n\n".join(parts)

    def _save_synth(self, question: str, synthesis: str, references: list[str]) -> str:
        """Génère et sauvegarde une page synth- à partir de la synthèse."""
        today = datetime.date.today().isoformat()
        prompt = _PROMPT_SYNTH_PAGE.format(
            question=question,
            synthesis=synthesis,
            today=today,
        )
        page_md = self.llm.complete(prompt, system=_SYSTEM_QUERY, max_tokens=2000)

        # Nettoyer le bloc ```markdown...```
        match = re.search(r"```(?:markdown|yaml)?\n(.*?)```", page_md, re.DOTALL)
        if match:
            rest = page_md[match.end():].strip()
            page_md = match.group(1) + ("\n" + rest if rest else "")

        # Générer slug depuis la question
        slug_suffix = re.sub(r"[^a-z0-9]+", "-", question.lower())[:40].strip("-")
        slug = f"synth-{slug_suffix}"

        path = self.wiki_dir / f"{slug}.md"
        path.write_text(page_md.strip() + "\n", encoding="utf-8")

        # Mettre à jour index.md
        self._update_index(slug, question, "synthèse")
        print(f"[query] Synthèse sauvegardée → wiki/{slug}.md")
        return slug

    def _update_index(self, slug: str, title: str, category: str) -> None:
        index_path = self.wiki_dir / "index.md"
        if not index_path.exists():
            return
        text = index_path.read_text(encoding="utf-8")
        entry = f"- [[{slug}]] | {category} | {title}"
        pattern = re.compile(rf"^- \[\[{re.escape(slug)}\]\].*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(entry, text)
        else:
            text = text.rstrip() + "\n" + entry + "\n"
        index_path.write_text(text, encoding="utf-8")

    def _append_log(self, operation: str, title: str) -> None:
        log_path = self.wiki_dir / "log.md"
        if not log_path.exists():
            return
        today = datetime.date.today().isoformat()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## [{today}] {operation} | {title}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interroge le wiki Wiki_LM")
    parser.add_argument("question", help="Question en langage naturel")
    parser.add_argument("--top", type=int, default=5, help="Pages candidates (défaut : 5)")
    parser.add_argument("--save", action="store_true", help="Enregistrer la synthèse comme page synth-")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
        help="Chemin vers Wiki_LM",
    )
    parser.add_argument("--backend", default="", help="Backend LLM : claude | ollama | openai")
    parser.add_argument("--model", default="", help="Modèle LLM")
    args = parser.parse_args()

    llm = LLM(backend=args.backend, model=args.model) if (args.backend or args.model) else LLM()
    wq = WikiQuery(args.wiki, llm=llm)
    result = wq.query(args.question, top_k=args.top, save=args.save)
    print(result)


if __name__ == "__main__":
    main()
