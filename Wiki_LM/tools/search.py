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
import json
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
import numpy as np
import pickle
from rank_bm25 import BM25Plus

_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
_CACHE_PATH = Path(__file__).resolve().parent.parent / "wiki_bm25_cache.pkl"
_CACHE_VERSION = 1

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
        self._corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None
        self._load()

    # ------------------------------------------------------------------
    # Cache BM25
    # ------------------------------------------------------------------

    def _dir_mtime(self) -> float:
        return self.wiki_dir.stat().st_mtime

    def _try_load_cache(self) -> bool:
        if not _CACHE_PATH.exists():
            return False
        try:
            with open(_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("version") != _CACHE_VERSION:
                return False
            if cached.get("wiki_dir") != str(self.wiki_dir):
                return False
            if cached.get("dir_mtime") != self._dir_mtime():
                return False
            self._pages = cached["pages"]
            self._corpus = cached["corpus"]
            self._bm25 = BM25Plus(self._corpus)
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "version": _CACHE_VERSION,
                    "wiki_dir": str(self.wiki_dir),
                    "dir_mtime": self._dir_mtime(),
                    "pages": self._pages,
                    "corpus": self._corpus,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[search] Cache BM25 non sauvegardé : {e}")

    # ------------------------------------------------------------------
    # Construction de l'index
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Charge l'index BM25 depuis le cache ou reconstruit depuis le disque."""
        if self._try_load_cache():
            return
        self._build_index()
        self._save_cache()

    def _build_index(self) -> None:
        """Reconstruit l'index BM25 complet depuis les fichiers wiki."""
        self._pages = []
        self._corpus = []

        for path in sorted(self.wiki_dir.glob("**/*.md")):
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
            self._corpus.append(tokens)
            # body exclu du cache — rechargé à la demande dans search()
            self._pages.append({
                "slug": slug,
                "path": path,
                "title": title,
                "category": category,
                "metadata": dict(post.metadata),
            })

        if self._corpus:
            self._bm25 = BM25Plus(self._corpus)

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
            try:
                body = frontmatter.load(page["path"]).content
            except Exception:
                body = ""
            results.append(SearchResult(
                slug=page["slug"],
                path=page["path"],
                title=page["title"],
                category=page["category"],
                score=score,
                excerpt=self._excerpt(body, tokens),
                metadata=page["metadata"],
            ))
        return results

    def reload(self) -> None:
        """Reconstruit l'index depuis le disque et invalide le cache."""
        self._build_index()
        self._save_cache()

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
# Recherche sémantique BGE-M3
# ---------------------------------------------------------------------------

class WikiSemanticSearch:
    """Recherche sémantique via embeddings BGE-M3 pré-calculés."""

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, wiki_dir: Path, embed_dir: Path | None = None):
        self.wiki_dir = wiki_dir
        self._embed_dir = embed_dir or _EMBED_DIR
        self._matrix: np.ndarray | None = None
        self._slugs: list[str] = []
        self._model = None
        self._load_index()

    def _load_index(self) -> None:
        index_path = self._embed_dir / "embeddings_index.json"
        matrix_path = self._embed_dir / "embeddings.npy"
        if not index_path.exists() or not matrix_path.exists():
            return
        self._matrix = np.load(matrix_path)
        with open(index_path, encoding="utf-8") as f:
            self._slugs = json.load(f)["slugs"]

    @property
    def available(self) -> bool:
        return self._matrix is not None and bool(self._slugs)

    def _encode(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def search(self, query: str, top_k: int = 10, min_score: float = 0.30) -> list[SearchResult]:
        if not self.available:
            return []
        q_emb = self._encode(query)
        scores = self._matrix @ q_emb
        top_indices = np.argsort(-scores)[: top_k * 3]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score or len(results) >= top_k:
                break
            slug = self._slugs[idx]
            path = self.wiki_dir / f"{slug}.md"
            if not path.exists():
                continue
            try:
                post = frontmatter.load(path)
            except Exception:
                continue
            results.append(SearchResult(
                slug=slug,
                path=path,
                title=str(post.get("title", slug)),
                category=str(post.get("category", "")),
                score=score,
                excerpt=self._excerpt(post.content, query),
                metadata=dict(post.metadata),
            ))
        return results

    @staticmethod
    def _excerpt(body: str, query: str, window: int = 30) -> str:
        words = body.split()
        for term in query.lower().split():
            for i, w in enumerate(words):
                if term in w.lower():
                    start = max(0, i - window // 2)
                    end = min(len(words), i + window // 2)
                    exc = " ".join(words[start:end])
                    if start > 0:
                        exc = "…" + exc
                    if end < len(words):
                        exc += "…"
                    return exc
        return " ".join(words[:window]) + ("…" if len(words) > window else "")


# ---------------------------------------------------------------------------
# Fusion hybride BM25 + sémantique (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def hybrid_search(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    rrf: dict[str, float] = {}
    by_slug: dict[str, SearchResult] = {}

    for rank, r in enumerate(bm25_results):
        rrf[r.slug] = rrf.get(r.slug, 0.0) + 1.0 / (rrf_k + rank + 1)
        by_slug.setdefault(r.slug, r)

    for rank, r in enumerate(semantic_results):
        rrf[r.slug] = rrf.get(r.slug, 0.0) + 1.0 / (rrf_k + rank + 1)
        by_slug.setdefault(r.slug, r)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        SearchResult(
            slug=slug,
            path=by_slug[slug].path,
            title=by_slug[slug].title,
            category=by_slug[slug].category,
            score=score,
            excerpt=by_slug[slug].excerpt,
            metadata=by_slug[slug].metadata,
        )
        for slug, score in ranked
    ]


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
