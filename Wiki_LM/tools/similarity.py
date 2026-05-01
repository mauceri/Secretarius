"""
Calcul de matrices de similarité pour les pages src- du wiki.

Signaux disponibles :
    EmbeddingSimilarity  — produit matriciel BGE-M3 (vecteurs L2-normalisés)
    CoLinkSimilarity     — Jaccard sur liens [[c-…]] / [[e-…]] partagés
    TagSimilarity        — Jaccard sur tags du frontmatter
    CombinedSimilarity   — moyenne pondérée de plusieurs signaux

Interface commune :
    signal.compute(slugs: list[str]) -> np.ndarray  # (N×N, float32), valeurs ∈ [0, 1]
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import frontmatter
import numpy as np

_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"


class BaseSimilarity(ABC):
    @abstractmethod
    def compute(self, slugs: list[str]) -> np.ndarray:
        """Retourne une matrice (N×N, float32) de similarité, valeurs ∈ [0, 1]."""
        ...


class EmbeddingSimilarity(BaseSimilarity):
    """Similarité cosinus via embeddings BGE-M3 pré-calculés."""

    def __init__(self, embed_dir: Path | None = None) -> None:
        self._embed_dir = Path(embed_dir) if embed_dir else _EMBED_DIR
        self._full_matrix: np.ndarray | None = None
        self._slug_to_idx: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        index_path = self._embed_dir / "embeddings_index.json"
        matrix_path = self._embed_dir / "embeddings.npy"
        if not index_path.exists() or not matrix_path.exists():
            return
        try:
            self._full_matrix = np.load(matrix_path)
            slugs = json.loads(index_path.read_text(encoding="utf-8"))["slugs"]
            self._slug_to_idx = {s: i for i, s in enumerate(slugs)}
        except Exception:
            self._full_matrix = None
            self._slug_to_idx = {}

    def compute(self, slugs: list[str]) -> np.ndarray:
        n = len(slugs)
        if self._full_matrix is None:
            return np.zeros((n, n), dtype=np.float32)

        rows = np.zeros((n, self._full_matrix.shape[1]), dtype=np.float32)
        for i, s in enumerate(slugs):
            if s in self._slug_to_idx:
                rows[i] = self._full_matrix[self._slug_to_idx[s]]

        sim = rows @ rows.T
        np.clip(sim, 0.0, 1.0, out=sim)
        return sim


_LINK_RE = re.compile(r"\[\[([ce]-[^\]|]+)")


class CoLinkSimilarity(BaseSimilarity):
    """Similarité Jaccard sur les liens [[c-…]] et [[e-…]] partagés."""

    def __init__(self, wiki_dir: Path) -> None:
        self._wiki_dir = Path(wiki_dir)

    def _links(self, slug: str) -> frozenset[str]:
        path = self._wiki_dir / f"{slug}.md"
        if not path.exists():
            return frozenset()
        try:
            post = frontmatter.load(path)
            return frozenset(_LINK_RE.findall(post.content))
        except Exception:
            return frozenset()

    def compute(self, slugs: list[str]) -> np.ndarray:
        n = len(slugs)
        sets = [self._links(s) for s in slugs]
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                union = len(sets[i] | sets[j])
                val = len(sets[i] & sets[j]) / union if union > 0 else 0.0
                sim[i, j] = sim[j, i] = float(val)
        return sim


class TagSimilarity(BaseSimilarity):
    """Similarité Jaccard sur les tags du frontmatter."""

    def __init__(self, wiki_dir: Path) -> None:
        self._wiki_dir = Path(wiki_dir)

    def _tags(self, slug: str) -> frozenset[str]:
        path = self._wiki_dir / f"{slug}.md"
        if not path.exists():
            return frozenset()
        try:
            post = frontmatter.load(path)
            raw = post.get("tags", [])
            if isinstance(raw, str):
                return frozenset(t.strip() for t in raw.split(",") if t.strip())
            return frozenset(str(t) for t in raw)
        except Exception:
            return frozenset()

    def compute(self, slugs: list[str]) -> np.ndarray:
        n = len(slugs)
        sets = [self._tags(s) for s in slugs]
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                union = len(sets[i] | sets[j])
                val = len(sets[i] & sets[j]) / union if union > 0 else 0.0
                sim[i, j] = sim[j, i] = float(val)
        return sim


class CombinedSimilarity(BaseSimilarity):
    """Moyenne pondérée de plusieurs signaux de similarité."""

    def __init__(
        self,
        signals: list[BaseSimilarity],
        weights: list[float] | None = None,
    ) -> None:
        self._signals = signals
        if not signals:
            self._weights = []
            return
        if weights is None:
            w = 1.0 / len(signals)
            self._weights = [w] * len(signals)
        else:
            total = sum(weights)
            self._weights = [w / total for w in weights]

    def compute(self, slugs: list[str]) -> np.ndarray:
        result: np.ndarray | None = None
        for sig, w in zip(self._signals, self._weights):
            m = sig.compute(slugs)
            result = m * w if result is None else result + m * w
        if result is None:
            return np.zeros((len(slugs), len(slugs)), dtype=np.float32)
        return result
