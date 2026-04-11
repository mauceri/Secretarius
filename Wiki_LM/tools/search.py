"""
Recherche BM25 sur les pages Markdown du wiki.

Usage CLI :
    python search.py "zettelkasten mémoire" --top 5
    python search.py "zettelkasten mémoire" --wiki /chemin/vers/Wiki_LM

Usage module :
    from search import WikiSearch
    ws = WikiSearch("/home/mauceric/Documents/Arbath/Wiki_LM")
    results = ws.search("zettelkasten mémoire", top_k=5)
    for r in results:
        print(r.slug, r.score, r.excerpt)
"""

from __future__ import annotations

import argparse
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
from rank_bm25 import BM25Plus

# ---------------------------------------------------------------------------
# Stopwords français (liste embarquée — pas de dépendance nltk)
# ---------------------------------------------------------------------------
FR_STOPWORDS = {
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir",
    "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme",
    "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait",
    "doit", "donc", "dos", "droite", "début", "elle", "elles", "en", "encore",
    "essai", "est", "et", "eu", "eux", "fait", "faites", "fois", "font",
    "force", "haut", "hors", "ici", "il", "ils", "je", "juste", "la", "le",
    "les", "leur", "là", "ma", "maintenant", "mais", "mes", "moi", "moins",
    "mon", "mot", "même", "ni", "nommés", "nos", "notre", "nous", "ou",
    "où", "par", "parce", "pas", "peut", "peu", "plupart", "pour", "pourquoi",
    "qu", "quand", "que", "quel", "quelle", "quelles", "quels", "qui",
    "sa", "sans", "ses", "si", "sien", "son", "sur", "ta", "te", "tels",
    "tes", "toi", "ton", "tous", "tout", "trop", "très", "tu", "un", "une",
    "vos", "votre", "vous", "vu", "y", "été", "être", "à", "de",
}


# ---------------------------------------------------------------------------
# Résultat de recherche
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    slug: str
    path: Path
    title: str
    category: str
    score: float
    excerpt: str          # extrait du contenu autour des termes trouvés
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.slug} ({self.category}) — {self.title}\n  {self.excerpt}"


# ---------------------------------------------------------------------------
# Tokeniseur français
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Minuscule, supprime ponctuation, filtre stopwords et tokens courts."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in FR_STOPWORDS]


# ---------------------------------------------------------------------------
# Moteur de recherche
# ---------------------------------------------------------------------------
class WikiSearch:
    def __init__(self, wiki_path: str | Path):
        self.wiki_dir = Path(wiki_path) / "wiki"
        if not self.wiki_dir.exists():
            raise FileNotFoundError(f"Répertoire wiki introuvable : {self.wiki_dir}")
        self._pages: list[dict] = []
        self._bm25: BM25Plus | None = None
        self._load()

    def _load(self) -> None:
        """Charge et indexe toutes les pages Markdown du wiki."""
        self._pages = []
        corpus: list[list[str]] = []

        for path in sorted(self.wiki_dir.glob("**/*.md")):
            # Ignorer index et log (fichiers meta)
            if path.name in ("index.md", "log.md"):
                continue
            try:
                post = frontmatter.load(path)
            except Exception:
                continue

            slug = path.stem
            title = post.get("title", slug)
            category = post.get("category", "")
            body = post.content

            tokens = tokenize(f"{title} {body}")
            corpus.append(tokens)
            self._pages.append({
                "slug": slug,
                "path": path,
                "title": title,
                "category": category,
                "body": body,
                "metadata": dict(post.metadata),
            })

        if corpus:
            self._bm25 = BM25Plus(corpus)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retourne les top_k pages les plus pertinentes pour la query."""
        if not self._pages or self._bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score < 0.001:
                break
            page = self._pages[idx]
            results.append(SearchResult(
                slug=page["slug"],
                path=page["path"],
                title=page["title"],
                category=page["category"],
                score=score,
                excerpt=self._excerpt(page["body"], tokens),
                metadata=page["metadata"],
            ))
        return results

    def reload(self) -> None:
        """Recharge l'index depuis le disque (après ingestion)."""
        self._load()

    @staticmethod
    def _excerpt(body: str, tokens: list[str], window: int = 30) -> str:
        """Extrait un court passage autour du premier token trouvé."""
        words = body.split()
        body_lower = body.lower()
        for token in tokens:
            match = re.search(r"\b" + re.escape(token), body_lower)
            if match:
                start_char = match.start()
                # Convertir position char → position mot (approx)
                prefix = body[:start_char].split()
                word_pos = len(prefix)
                start = max(0, word_pos - window // 2)
                end = min(len(words), word_pos + window // 2)
                excerpt = " ".join(words[start:end])
                if start > 0:
                    excerpt = "…" + excerpt
                if end < len(words):
                    excerpt += "…"
                return excerpt
        # Fallback : début du document
        return " ".join(words[:window]) + ("…" if len(words) > window else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Recherche BM25 dans le wiki Wiki_LM")
    parser.add_argument("query", help="Question ou mots-clés")
    parser.add_argument("--top", type=int, default=5, help="Nombre de résultats (défaut : 5)")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
        help="Chemin vers Wiki_LM (défaut : $WIKI_PATH ou ~/Documents/Arbath/Wiki_LM)",
    )
    args = parser.parse_args()

    ws = WikiSearch(args.wiki)
    results = ws.search(args.query, top_k=args.top)

    if not results:
        print("Aucun résultat.")
        return

    for r in results:
        print(r)
        print()


if __name__ == "__main__":
    main()
