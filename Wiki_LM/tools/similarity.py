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
