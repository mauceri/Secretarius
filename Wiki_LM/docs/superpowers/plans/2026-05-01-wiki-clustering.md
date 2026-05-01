# Wiki Clustering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un système de clustering multi-granularité des pages `src-` du wiki, navigable dans Obsidian via le graphe de liens et déclenchable depuis le serveur Flask.

**Architecture:** `similarity.py` calcule des matrices de similarité (N×N) à partir de signaux pluggables (embeddings BGE-M3, co-liens, tags). `cluster.py` consume une matrice, lance HDBSCAN, génère des fichiers wiki avec `[[liens]]` Obsidian. `server.py` expose `/cluster` et `/cluster-status`.

**Tech Stack:** scikit-learn>=1.3 (HDBSCAN), numpy, python-frontmatter, pytest, Flask (déjà installés)

---

## Fichiers concernés

| Action | Chemin |
|--------|--------|
| Créer | `tools/similarity.py` |
| Créer | `tools/cluster.py` |
| Créer | `tests/test_similarity.py` |
| Créer | `tests/test_cluster.py` |
| Modifier | `tools/server.py` (ajouter `/cluster`, `/cluster-status`) |
| Modifier | `requirements.txt` (ajouter `scikit-learn>=1.3`) |

Répertoire de travail pour toutes les commandes : `~/Secretarius/Wiki_LM`

---

## Task 1 : Dépendances

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1 : Ajouter scikit-learn à requirements.txt**

```
scikit-learn>=1.3
```
Insérer après la ligne `flask>=3.0.0`.

- [ ] **Step 2 : Vérifier que HDBSCAN est importable**

```bash
.venv/bin/python -c "from sklearn.cluster import HDBSCAN; print('OK')"
```
Expected: `OK`

- [ ] **Step 3 : Commit**

```bash
git add requirements.txt
git commit -m "deps: ajouter scikit-learn>=1.3 (HDBSCAN)"
```

---

## Task 2 : similarity.py — EmbeddingSimilarity

**Files:**
- Create: `tests/test_similarity.py`
- Create: `tools/similarity.py` (partiel — BaseSimilarity + EmbeddingSimilarity)

- [ ] **Step 1 : Écrire les tests EmbeddingSimilarity**

Créer `tests/test_similarity.py` :

```python
"""Tests pour similarity.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_dir(tmp_path: Path, slugs: list[str], dim: int = 8) -> Path:
    """Crée un répertoire d'embeddings factice avec vecteurs L2-normalisés."""
    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(slugs), dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(embed_dir / "embeddings.npy", vecs)
    (embed_dir / "embeddings_index.json").write_text(
        json.dumps({"slugs": slugs}), encoding="utf-8"
    )
    return embed_dir


# ---------------------------------------------------------------------------
# EmbeddingSimilarity
# ---------------------------------------------------------------------------

def test_embedding_similarity_diagonal_is_one(tmp_path):
    slugs = ["src-a", "src-b", "src-c"]
    embed_dir = _make_embed_dir(tmp_path, slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(slugs)

    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)


def test_embedding_similarity_values_in_range(tmp_path):
    slugs = ["src-a", "src-b", "src-c"]
    embed_dir = _make_embed_dir(tmp_path, slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(slugs)

    assert float(mat.min()) >= 0.0
    assert float(mat.max()) <= 1.0 + 1e-5


def test_embedding_similarity_symmetry(tmp_path):
    slugs = ["src-a", "src-b", "src-c"]
    embed_dir = _make_embed_dir(tmp_path, slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(slugs)

    np.testing.assert_allclose(mat, mat.T, atol=1e-5)


def test_embedding_similarity_missing_dir(tmp_path):
    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(tmp_path / "nonexistent").compute(["src-a", "src-b"])

    assert mat.shape == (2, 2)
    assert float(mat.sum()) == 0.0


def test_embedding_similarity_subset(tmp_path):
    """Requête sur un sous-ensemble des slugs indexés."""
    all_slugs = ["src-a", "src-b", "src-c", "src-d"]
    embed_dir = _make_embed_dir(tmp_path, all_slugs)

    from similarity import EmbeddingSimilarity
    mat = EmbeddingSimilarity(embed_dir).compute(["src-a", "src-c"])

    assert mat.shape == (2, 2)
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)
```

- [ ] **Step 2 : Vérifier que les tests échouent (module absent)**

```bash
.venv/bin/python -m pytest tests/test_similarity.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'similarity'`

- [ ] **Step 3 : Créer tools/similarity.py avec BaseSimilarity et EmbeddingSimilarity**

```python
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
        self._full_matrix = np.load(matrix_path)
        slugs = json.loads(index_path.read_text(encoding="utf-8"))["slugs"]
        self._slug_to_idx = {s: i for i, s in enumerate(slugs)}

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
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
.venv/bin/python -m pytest tests/test_similarity.py -v -k "embedding"
```
Expected: 5 tests PASSED

- [ ] **Step 5 : Commit intermédiaire**

```bash
git add tools/similarity.py tests/test_similarity.py
git commit -m "feat: similarity.py — BaseSimilarity + EmbeddingSimilarity (TDD)"
```

---

## Task 3 : similarity.py — CoLinkSimilarity + TagSimilarity + CombinedSimilarity

**Files:**
- Modify: `tests/test_similarity.py` (ajouter tests)
- Modify: `tools/similarity.py` (compléter)

- [ ] **Step 1 : Ajouter les tests dans tests/test_similarity.py**

Ajouter à la fin du fichier :

```python
# ---------------------------------------------------------------------------
# CoLinkSimilarity
# ---------------------------------------------------------------------------

def _write_page(wiki_dir: Path, slug: str, body: str, tags: list[str] | None = None) -> None:
    tags_yaml = f"tags: {tags}\n" if tags else ""
    (wiki_dir / f"{slug}.md").write_text(
        f"---\ntitle: {slug}\n{tags_yaml}---\n\n{body}", encoding="utf-8"
    )


def test_colink_identical_pages(tmp_path):
    """Deux pages partageant exactement les mêmes liens → Jaccard = 1.0."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-foo]] [[e-bar]]")
    _write_page(wiki_dir, "src-b", "[[c-foo]] [[e-bar]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1.0)


def test_colink_no_overlap(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-foo]]")
    _write_page(wiki_dir, "src-b", "[[c-bar]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(0.0)


def test_colink_partial_overlap(tmp_path):
    """Jaccard({a,b}, {b,c}) = 1/3."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-a]] [[c-b]]")
    _write_page(wiki_dir, "src-b", "[[c-b]] [[c-c]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1 / 3, abs=1e-5)


def test_colink_symmetry(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "[[c-foo]] [[c-bar]]")
    _write_page(wiki_dir, "src-b", "[[c-bar]] [[e-baz]]")

    from similarity import CoLinkSimilarity
    mat = CoLinkSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(mat[1, 0])


# ---------------------------------------------------------------------------
# TagSimilarity
# ---------------------------------------------------------------------------

def test_tag_similarity_identical(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "", tags=["foo", "bar"])
    _write_page(wiki_dir, "src-b", "", tags=["foo", "bar"])

    from similarity import TagSimilarity
    mat = TagSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1.0)


def test_tag_similarity_partial(tmp_path):
    """Jaccard({foo,bar}, {bar,baz}) = 1/3."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "", tags=["foo", "bar"])
    _write_page(wiki_dir, "src-b", "", tags=["bar", "baz"])

    from similarity import TagSimilarity
    mat = TagSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(1 / 3, abs=1e-5)


def test_tag_similarity_no_tags(tmp_path):
    """Pages sans tags → similarité 0."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "src-a", "rien")
    _write_page(wiki_dir, "src-b", "rien")

    from similarity import TagSimilarity
    mat = TagSimilarity(wiki_dir).compute(["src-a", "src-b"])

    assert mat[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CombinedSimilarity
# ---------------------------------------------------------------------------

def test_combined_equal_weights():
    """Avec deux signaux égaux à 0.4 et 0.8 → résultat = 0.6."""
    from similarity import BaseSimilarity, CombinedSimilarity

    class _Fixed(BaseSimilarity):
        def __init__(self, val: float) -> None:
            self._val = val
        def compute(self, slugs: list[str]) -> np.ndarray:
            n = len(slugs)
            return np.full((n, n), self._val, dtype=np.float32)

    sig = CombinedSimilarity([_Fixed(0.4), _Fixed(0.8)])
    mat = sig.compute(["a", "b"])

    np.testing.assert_allclose(mat, 0.6, atol=1e-5)


def test_combined_custom_weights():
    """Poids 1:3 → résultat = 0.25*0.4 + 0.75*0.8 = 0.7."""
    from similarity import BaseSimilarity, CombinedSimilarity

    class _Fixed(BaseSimilarity):
        def __init__(self, val: float) -> None:
            self._val = val
        def compute(self, slugs: list[str]) -> np.ndarray:
            n = len(slugs)
            return np.full((n, n), self._val, dtype=np.float32)

    sig = CombinedSimilarity([_Fixed(0.4), _Fixed(0.8)], weights=[1.0, 3.0])
    mat = sig.compute(["a", "b"])

    np.testing.assert_allclose(mat, pytest.approx(0.7, abs=1e-5))
```

- [ ] **Step 2 : Vérifier que les nouveaux tests échouent**

```bash
.venv/bin/python -m pytest tests/test_similarity.py -v -k "colink or tag or combined"
```
Expected: `ImportError` ou `FAILED` sur les 9 nouveaux tests

- [ ] **Step 3 : Ajouter CoLinkSimilarity, TagSimilarity, CombinedSimilarity dans tools/similarity.py**

Ajouter après la classe `EmbeddingSimilarity` :

```python
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
```

- [ ] **Step 4 : Vérifier que toute la suite similarity passe**

```bash
.venv/bin/python -m pytest tests/test_similarity.py -v
```
Expected: tous les tests PASSED

- [ ] **Step 5 : Commit**

```bash
git add tools/similarity.py tests/test_similarity.py
git commit -m "feat: similarity.py — CoLinkSimilarity, TagSimilarity, CombinedSimilarity (TDD)"
```

---

## Task 4 : cluster.py — fonctions utilitaires

**Files:**
- Create: `tests/test_cluster.py`
- Create: `tools/cluster.py` (partiel — fonctions _find_paragon, _nearest_clusters, _describe_cluster)

- [ ] **Step 1 : Écrire les tests des fonctions utilitaires**

Créer `tests/test_cluster.py` :

```python
"""Tests pour cluster.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# ---------------------------------------------------------------------------
# _find_paragon
# ---------------------------------------------------------------------------

def test_find_paragon_selects_most_central():
    """Le parangon est le document avec la plus haute similarité moyenne."""
    from cluster import _find_paragon

    # index 1 : avg = (0.5 + 1.0 + 0.6) / 3 = 0.7 → le plus central
    sim = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.6],
        [0.2, 0.6, 1.0],
    ], dtype=np.float32)

    assert _find_paragon([0, 1, 2], sim) == 1


def test_find_paragon_single_element():
    sim = np.array([[1.0]], dtype=np.float32)
    assert _find_paragon([0], sim) == 0


def test_find_paragon_with_offset_indices():
    """Les indices passés peuvent être un sous-ensemble non continu."""
    from cluster import _find_paragon

    sim = np.array([
        [1.0, 0.1, 0.9],
        [0.1, 1.0, 0.1],
        [0.9, 0.1, 1.0],
    ], dtype=np.float32)

    # Indices [0, 2] : avg(0) = (1.0 + 0.9)/2 = 0.95, avg(2) = (0.9 + 1.0)/2 = 0.95 → premier gagne
    result = _find_paragon([0, 2], sim)
    assert result in (0, 2)


# ---------------------------------------------------------------------------
# _nearest_clusters
# ---------------------------------------------------------------------------

def test_nearest_clusters_returns_top_k():
    from cluster import _nearest_clusters

    # 4 clusters avec centroïdes simples
    centroids = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.9, 0.1], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([0.0, 0.9], dtype=np.float32),
    }

    nearest = _nearest_clusters(0, centroids, top_k=2)

    assert len(nearest) == 2
    # Le plus proche de [1,0] doit être [0.9, 0.1] (cluster 1)
    assert nearest[0][0] == 1
    # Scores décroissants
    assert nearest[0][1] >= nearest[1][1]


def test_nearest_clusters_excludes_self():
    from cluster import _nearest_clusters

    centroids = {0: np.array([1.0, 0.0]), 1: np.array([0.5, 0.5])}
    nearest = _nearest_clusters(0, centroids, top_k=3)

    assert all(cid != 0 for cid, _ in nearest)


# ---------------------------------------------------------------------------
# _describe_cluster
# ---------------------------------------------------------------------------

def test_describe_cluster_with_mock_llm():
    from cluster import _describe_cluster

    class MockLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return "TITRE: Philosophie du langage\nDESCRIPTION: Ce groupe traite du langage."

    sim = np.eye(3, dtype=np.float32)
    slugs = ["src-a", "src-b", "src-c"]
    pages = {
        "src-a": {"title": "A", "abstract": "Résumé A"},
        "src-b": {"title": "B", "abstract": "Résumé B"},
        "src-c": {"title": "C", "abstract": "Résumé C"},
    }

    title, desc = _describe_cluster(slugs, pages, sim, slugs, MockLLM())

    assert title == "Philosophie du langage"
    assert "langage" in desc


def test_describe_cluster_no_llm_returns_defaults():
    from cluster import _describe_cluster

    sim = np.eye(2, dtype=np.float32)
    slugs = ["src-a", "src-b"]
    pages = {"src-a": {"title": "A", "abstract": ""}, "src-b": {"title": "B", "abstract": ""}}

    title, desc = _describe_cluster(slugs, pages, sim, slugs, llm=None)

    assert isinstance(title, str)
    assert isinstance(desc, str)
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'cluster'`

- [ ] **Step 3 : Créer tools/cluster.py avec les fonctions utilitaires**

```python
"""
Clustering des pages src- du wiki.

Usage CLI :
    python tools/cluster.py --signal embeddings --param 30
    python tools/cluster.py --signal embeddings --param 10,30,60
    python tools/cluster.py --signal embeddings+colinks --param 30 --no-llm

Endpoint serveur : POST /cluster {"signal": "embeddings", "param": 30}
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path

import frontmatter
import numpy as np
from sklearn.cluster import HDBSCAN

from llm import LLM
from similarity import (
    BaseSimilarity,
    CoLinkSimilarity,
    CombinedSimilarity,
    EmbeddingSimilarity,
    TagSimilarity,
)

_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
_DEFAULT_WIKI = os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM"))


# ---------------------------------------------------------------------------
# Chargement des pages src-
# ---------------------------------------------------------------------------

def _load_src_pages(wiki_dir: Path) -> list[dict]:
    """Retourne la liste de dicts {slug, title, abstract} pour toutes les pages src-."""
    pages = []
    for path in sorted(wiki_dir.glob("src-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        m = re.search(r"## Résumé\n(.+?)(?=\n## |\Z)", post.content, re.DOTALL)
        abstract = m.group(1).strip() if m else ""
        pages.append({
            "slug": path.stem,
            "title": str(post.get("title", path.stem)),
            "abstract": abstract,
        })
    return pages


# ---------------------------------------------------------------------------
# Construction du signal
# ---------------------------------------------------------------------------

def _build_signal(signal_str: str, wiki_dir: Path, embed_dir: Path) -> BaseSimilarity:
    parts = [p.strip() for p in signal_str.split("+")]
    signals: list[BaseSimilarity] = []
    for part in parts:
        if part == "embeddings":
            signals.append(EmbeddingSimilarity(embed_dir))
        elif part == "colinks":
            signals.append(CoLinkSimilarity(wiki_dir))
        elif part == "tags":
            signals.append(TagSimilarity(wiki_dir))
        else:
            raise ValueError(f"Signal inconnu : {part!r}. Valeurs : embeddings, colinks, tags")
    return signals[0] if len(signals) == 1 else CombinedSimilarity(signals)


# ---------------------------------------------------------------------------
# Fonctions utilitaires clustering
# ---------------------------------------------------------------------------

def _find_paragon(indices: list[int], sim: np.ndarray) -> int:
    """Retourne l'indice (dans sim) du document le plus central du cluster."""
    sub = sim[np.ix_(indices, indices)]
    return indices[int(np.argmax(sub.mean(axis=1)))]


def _nearest_clusters(
    cluster_id: int,
    centroids: dict[int, np.ndarray],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Retourne les top_k clusters les plus proches par similarité cosinus des centroïdes."""
    ref = centroids[cluster_id]
    ref_norm = np.linalg.norm(ref)
    scores = []
    for cid, vec in centroids.items():
        if cid == cluster_id:
            continue
        denom = ref_norm * np.linalg.norm(vec)
        sim = float(ref @ vec / denom) if denom > 0 else 0.0
        scores.append((cid, sim))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


def _describe_cluster(
    cluster_slugs: list[str],
    pages_by_slug: dict[str, dict],
    sim: np.ndarray,
    all_slugs: list[str],
    llm,  # LLM | None
) -> tuple[str, str]:
    """Retourne (titre, description) via LLM, ou valeurs par défaut si llm=None."""
    if llm is None:
        return "Cluster", ""

    slug_to_idx = {s: i for i, s in enumerate(all_slugs)}
    indices = [slug_to_idx[s] for s in cluster_slugs if s in slug_to_idx]
    if not indices:
        return "Cluster", ""

    sub = sim[np.ix_(indices, indices)]
    top5_local = np.argsort(-sub.mean(axis=1))[:5]
    top_slugs = [cluster_slugs[i] for i in top5_local]

    excerpts = []
    for s in top_slugs:
        p = pages_by_slug.get(s, {})
        title = p.get("title", s)
        abstract = p.get("abstract", "")[:300]
        excerpts.append(f"- {title} : {abstract}")

    prompt = (
        "Voici les résumés des documents les plus représentatifs d'un groupe thématique :\n\n"
        + "\n".join(excerpts)
        + "\n\nDonne :\n"
        "1. Un titre court (4-6 mots) caractérisant le thème commun\n"
        "2. Une description de 2-3 phrases résumant ce que ces documents ont en commun\n\n"
        "Format de réponse strict :\nTITRE: <titre>\nDESCRIPTION: <description>"
    )
    response = llm.complete(prompt, max_tokens=300)

    title = "Cluster"
    description = ""
    for line in response.splitlines():
        if line.startswith("TITRE:"):
            title = line[6:].strip()
        elif line.startswith("DESCRIPTION:"):
            description = line[12:].strip()
    return title, description
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v
```
Expected: tous les tests PASSED

- [ ] **Step 5 : Commit**

```bash
git add tools/cluster.py tests/test_cluster.py
git commit -m "feat: cluster.py — fonctions utilitaires _find_paragon, _nearest_clusters, _describe_cluster (TDD)"
```

---

## Task 5 : cluster.py — run_clustering + écriture fichiers

**Files:**
- Modify: `tests/test_cluster.py` (ajouter tests d'intégration)
- Modify: `tools/cluster.py` (ajouter _get_embed_rows, run_clustering)

- [ ] **Step 1 : Ajouter les tests d'intégration dans tests/test_cluster.py**

Ajouter à la fin du fichier :

```python
# ---------------------------------------------------------------------------
# run_clustering — intégration
# ---------------------------------------------------------------------------

def _setup_wiki_with_embeds(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    """
    Crée n pages src- + embeddings factices formant 2 groupes bien séparés.
    Retourne (wiki_dir, embed_dir).
    """
    import json

    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    slugs = [f"src-{i:02d}" for i in range(n)]
    for i, slug in enumerate(slugs):
        (wiki_dir / f"{slug}.md").write_text(
            f"---\ntitle: Source {i}\ncategory: source\ntags: [test]\n---\n\n"
            f"## Résumé\n\nTexte de test numéro {i}.\n",
            encoding="utf-8",
        )

    # Groupes : 0..n//2-1 vs n//2..n-1, vecteurs bien séparés
    rng = np.random.default_rng(0)
    vecs = np.zeros((n, 16), dtype=np.float32)
    half = n // 2
    vecs[:half, :8] = 1.0
    vecs[half:, 8:] = 1.0
    vecs += rng.standard_normal((n, 16)).astype(np.float32) * 0.05
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    np.save(embed_dir / "embeddings.npy", vecs)
    (embed_dir / "embeddings_index.json").write_text(
        json.dumps({"slugs": slugs}), encoding="utf-8"
    )

    return wiki_dir, embed_dir


def test_run_clustering_creates_output_dir(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    out_dir = wiki_dir / "clustering-embeddings-hdbscan-2"
    assert out_dir.exists()


def test_run_clustering_creates_index_and_unclustered(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    out_dir = wiki_dir / "clustering-embeddings-hdbscan-2"
    assert (out_dir / "index.md").exists()
    assert (out_dir / "unclustered.md").exists()


def test_run_clustering_cluster_files_have_members(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    stats = run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    assert stats["clusters"] >= 1
    out_dir = wiki_dir / f"clustering-embeddings-hdbscan-2"
    cluster_files = [f for f in out_dir.glob("cluster-*.md")]
    assert len(cluster_files) == stats["clusters"]

    # Chaque fichier cluster doit contenir un lien [[src-…]]
    for f in cluster_files:
        content = f.read_text(encoding="utf-8")
        assert "[[src-" in content


def test_run_clustering_stats_dict(tmp_path):
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path)
    stats = run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    assert "clusters" in stats
    assert "noise" in stats
    assert stats["clusters"] + stats["noise"] <= 6
    assert stats["signal"] == "embeddings"
    assert stats["param"] == 2


def test_run_clustering_returns_two_clusters_for_well_separated_data(tmp_path):
    """Données bien séparées en 2 groupes → exactement 2 clusters (param=2)."""
    from cluster import run_clustering

    wiki_dir, embed_dir = _setup_wiki_with_embeds(tmp_path, n=10)
    stats = run_clustering(wiki_dir, embed_dir, "embeddings", param=2, llm=None)

    assert stats["clusters"] == 2
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v -k "run_clustering"
```
Expected: `AttributeError: module 'cluster' has no attribute 'run_clustering'`

- [ ] **Step 3 : Ajouter _get_embed_rows et run_clustering dans tools/cluster.py**

Ajouter après la fonction `_describe_cluster` :

```python
# ---------------------------------------------------------------------------
# Embeddings bruts pour calcul de centroïdes
# ---------------------------------------------------------------------------

def _get_embed_rows(slugs: list[str], embed_dir: Path) -> np.ndarray | None:
    """Retourne un tableau (N, dim) des vecteurs d'embedding pour les slugs donnés."""
    index_path = embed_dir / "embeddings_index.json"
    matrix_path = embed_dir / "embeddings.npy"
    if not index_path.exists() or not matrix_path.exists():
        return None
    full = np.load(matrix_path)
    all_slugs = json.loads(index_path.read_text(encoding="utf-8"))["slugs"]
    slug_to_idx = {s: i for i, s in enumerate(all_slugs)}
    rows = np.zeros((len(slugs), full.shape[1]), dtype=np.float32)
    for i, s in enumerate(slugs):
        if s in slug_to_idx:
            rows[i] = full[slug_to_idx[s]]
    return rows


# ---------------------------------------------------------------------------
# Clustering principal
# ---------------------------------------------------------------------------

def run_clustering(
    wiki_dir: Path,
    embed_dir: Path,
    signal_str: str,
    param: int,
    llm=None,  # LLM | None
    algo: str = "hdbscan",
) -> dict:
    """
    Lance le clustering et écrit les fichiers wiki.
    Retourne un dict de stats : clusters, noise, signal, algo, param, out_dir.
    """
    pages = _load_src_pages(wiki_dir)
    if not pages:
        return {"clusters": 0, "noise": 0, "signal": signal_str, "algo": algo, "param": param, "out_dir": ""}

    slugs = [p["slug"] for p in pages]
    pages_by_slug = {p["slug"]: p for p in pages}

    signal = _build_signal(signal_str, wiki_dir, embed_dir)
    sim = signal.compute(slugs)

    dist = (1.0 - sim).clip(0.0).astype(np.float64)
    np.fill_diagonal(dist, 0.0)

    labels = HDBSCAN(
        min_cluster_size=param,
        metric="precomputed",
        cluster_selection_method="eom",
    ).fit_predict(dist)

    clusters: dict[int, list[int]] = {}
    noise: list[int] = []
    for i, label in enumerate(labels):
        if label == -1:
            noise.append(i)
        else:
            clusters.setdefault(int(label), []).append(i)

    embed_rows = _get_embed_rows(slugs, embed_dir)

    centroids: dict[int, np.ndarray] = {}
    for cid, idx_list in clusters.items():
        if embed_rows is not None:
            centroids[cid] = embed_rows[idx_list].mean(axis=0)
        else:
            centroids[cid] = sim[idx_list].mean(axis=0)

    out_dir = wiki_dir / f"clustering-{signal_str}-{algo}-{param}"
    out_dir.mkdir(exist_ok=True)
    today = datetime.date.today().isoformat()

    cluster_slugs = {cid: f"cluster-{signal_str}-{algo}-{param}-{cid:04d}" for cid in clusters}

    for cid, idx_list in clusters.items():
        paragon_idx = _find_paragon(idx_list, sim)
        paragon_slug = slugs[paragon_idx]
        member_slugs = [slugs[i] for i in idx_list]

        near = _nearest_clusters(cid, centroids) if len(centroids) > 1 else []
        title, description = _describe_cluster(member_slugs, pages_by_slug, sim, slugs, llm)
        cluster_file_slug = cluster_slugs[cid]

        lines = [
            "---",
            "category: cluster",
            f"signal: {signal_str}",
            f"algo: {algo}",
            f"param: {param}",
            f"members: {len(idx_list)}",
            f"paragon: {paragon_slug}",
            f"created: {today}",
            "---", "",
            f"# {title}", "",
        ]
        if description:
            lines += [description, ""]
        lines += ["## Parangon", ""]
        paragon_title = pages_by_slug.get(paragon_slug, {}).get("title", paragon_slug)
        lines.append(f"[[{paragon_slug}]] — {paragon_title}")
        lines += ["", "## Documents membres", ""]
        for s in member_slugs:
            t = pages_by_slug.get(s, {}).get("title", s)
            lines.append(f"- [[{s}]] — {t}")
        lines += ["", "## Clusters proches", ""]
        for near_cid, near_score in near:
            near_slug = cluster_slugs.get(near_cid, f"cluster-{near_cid}")
            lines.append(f"- [[{near_slug}]] (similarité : {near_score:.2f})")

        (out_dir / f"{cluster_file_slug}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    # unclustered.md
    unclustered_lines = [
        f"# Sources non assignées ({len(noise)} documents)", "",
        "Ces sources sont thématiquement isolées (bruit HDBSCAN).", "",
    ]
    for i in noise:
        s = slugs[i]
        t = pages_by_slug.get(s, {}).get("title", s)
        unclustered_lines.append(f"- [[{s}]] — {t}")
    (out_dir / "unclustered.md").write_text("\n".join(unclustered_lines), encoding="utf-8")

    # index.md
    index_lines = [
        f"# Index — clustering-{signal_str}-{algo}-{param}", "",
        f"- **Signal :** {signal_str}",
        f"- **Algorithme :** {algo}, param={param}",
        f"- **Clusters :** {len(clusters)}",
        f"- **Sources non assignées :** {len(noise)}",
        f"- **Généré le :** {today}", "",
        "## Clusters", "",
    ]
    for cid in sorted(clusters):
        index_lines.append(f"- [[{cluster_slugs[cid]}]]")
    (out_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    return {
        "clusters": len(clusters),
        "noise": len(noise),
        "signal": signal_str,
        "algo": algo,
        "param": param,
        "out_dir": str(out_dir),
    }
```

- [ ] **Step 4 : Vérifier que tous les tests cluster passent**

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v
```
Expected: tous les tests PASSED

- [ ] **Step 5 : Commit**

```bash
git add tools/cluster.py tests/test_cluster.py
git commit -m "feat: cluster.py — run_clustering + écriture fichiers wiki (TDD)"
```

---

## Task 6 : cluster.py — CLI (main)

**Files:**
- Modify: `tools/cluster.py` (ajouter `main()`)

- [ ] **Step 1 : Ajouter main() à la fin de tools/cluster.py**

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering des pages src- du wiki")
    parser.add_argument(
        "--wiki",
        default=_DEFAULT_WIKI,
        help="Chemin vers le wiki (défaut : $WIKI_PATH ou ~/Documents/Arbath/Wiki_LM)",
    )
    parser.add_argument(
        "--signal",
        default="embeddings",
        help="Signal(s) : embeddings, colinks, tags, ou combinaison ex. embeddings+colinks",
    )
    parser.add_argument(
        "--param",
        default="30",
        help="min_cluster_size(s), virgule-séparés ex. 10,30,60",
    )
    parser.add_argument("--no-llm", action="store_true", help="Ne pas appeler le LLM")
    parser.add_argument("--embed-dir", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    embed_dir = Path(args.embed_dir) if args.embed_dir else _EMBED_DIR

    if args.no_llm:
        llm = None
    elif args.backend or args.model:
        llm = LLM(backend=args.backend, model=args.model)
    else:
        llm = LLM()

    params = [int(p.strip()) for p in args.param.split(",") if p.strip()]
    for param in params:
        print(f"[cluster] signal={args.signal} param={param} …")
        stats = run_clustering(wiki_dir, embed_dir, args.signal, param, llm=llm)
        print(
            f"[cluster] {stats['clusters']} clusters, {stats['noise']} non assignés"
            f" → {stats['out_dir']}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Test smoke CLI avec --no-llm**

```bash
.venv/bin/python tools/cluster.py --param 2 --no-llm --embed-dir /home/mauceric/Secretarius/Wiki_LM/embeddings 2>&1 | tail -5
```
Expected: lignes `[cluster] signal=embeddings param=2 …` puis `[cluster] N clusters, M non assignés → …`

- [ ] **Step 3 : Vérifier que la suite de tests complète passe toujours**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```
Expected: tous les tests PASSED (aucune régression)

- [ ] **Step 4 : Commit**

```bash
git add tools/cluster.py
git commit -m "feat: cluster.py — CLI main() (--signal, --param, --no-llm)"
```

---

## Task 7 : server.py — endpoints /cluster et /cluster-status

**Files:**
- Modify: `tools/server.py`
- Modify: `tests/test_cluster.py` (ajouter tests serveur)

- [ ] **Step 1 : Ajouter les tests serveur dans tests/test_cluster.py**

Ajouter à la fin du fichier :

```python
# ---------------------------------------------------------------------------
# Endpoints serveur /cluster + /cluster-status
# ---------------------------------------------------------------------------

def test_cluster_status_initial(tmp_path):
    """GET /cluster-status retourne running=False initialement."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

    # Patch les globals de server avant import
    import importlib
    import server as srv_module

    # Réinitialiser le statut
    srv_module._cluster_status["running"] = False
    srv_module._cluster_status["last"] = None
    srv_module._cluster_status["error"] = None

    client = srv_module.app.test_client()
    resp = client.get("/cluster-status")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["running"] is False
    assert data["last"] is None


def test_cluster_endpoint_missing_param(tmp_path):
    """POST /cluster sans 'param' retourne une erreur 400."""
    import server as srv_module

    client = srv_module.app.test_client()
    resp = client.post("/cluster", json={"signal": "embeddings"})

    assert resp.status_code == 400
    assert "param" in resp.get_json().get("error", "")
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v -k "cluster_status or cluster_endpoint"
```
Expected: `AttributeError` ou `AssertionError` (endpoints absents)

- [ ] **Step 3 : Modifier tools/server.py pour ajouter les imports et les deux endpoints**

En haut du fichier, après `from query import WikiQuery`, ajouter :

```python
from cluster import run_clustering
```

Après la variable `_embed_status`, ajouter :

```python
_cluster_status: dict = {"running": False, "last": None, "error": None}
```

Ajouter les deux routes avant `def main()` :

```python
@app.post("/cluster")
def start_cluster():
    """Lance cluster.py en arrière-plan."""
    data = request.get_json(silent=True) or {}
    if "param" not in data:
        return jsonify({"error": "Paramètre 'param' manquant"}), 400

    signal = str(data.get("signal", "embeddings"))
    param = int(data["param"])

    if _cluster_status["running"]:
        return jsonify({"status": "already_running"})

    def _run():
        _cluster_status["running"] = True
        _cluster_status["error"] = None
        try:
            wiki_dir = Path(_wq._search.wiki_dir)
            embed_dir = Path(__file__).resolve().parent.parent / "embeddings"
            llm = LLM()
            stats = run_clustering(wiki_dir, embed_dir, signal, param, llm=llm)
            _wq._search.reload()
            import datetime
            _cluster_status["last"] = {**stats, "timestamp": datetime.datetime.now().isoformat()}
        except Exception as e:
            _cluster_status["error"] = str(e)
        finally:
            _cluster_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.get("/cluster-status")
def cluster_status():
    return jsonify(_cluster_status)
```

- [ ] **Step 4 : Vérifier que les tests serveur passent**

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v -k "cluster_status or cluster_endpoint"
```
Expected: 2 tests PASSED

- [ ] **Step 5 : Vérifier que toute la suite passe**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```
Expected: tous les tests PASSED

- [ ] **Step 6 : Commit final**

```bash
git add tools/server.py tests/test_cluster.py
git commit -m "feat: server.py — endpoints /cluster + /cluster-status"
```

---

## Task 8 : Validation manuelle sur données réelles

- [ ] **Step 1 : Lancer un clustering de test (param=60, sans LLM)**

```bash
.venv/bin/python tools/cluster.py --signal embeddings --param 60 --no-llm
```
Expected: `[cluster] N clusters, M non assignés → …/clustering-embeddings-hdbscan-60`

- [ ] **Step 2 : Vérifier les fichiers générés**

```bash
ls ~/Documents/Arbath/Wiki_LM/wiki/clustering-embeddings-hdbscan-60/ | head -20
head -30 ~/Documents/Arbath/Wiki_LM/wiki/clustering-embeddings-hdbscan-60/index.md
```
Expected: `index.md`, `unclustered.md`, N fichiers `cluster-*.md`

- [ ] **Step 3 : Vérifier le graphe Obsidian**

Ouvrir Obsidian → Graph View → filtrer sur `category:cluster`. Les clusters doivent apparaître comme nœuds liés aux pages `src-`.

- [ ] **Step 4 : Lancer avec LLM pour annoter**

```bash
.venv/bin/python tools/cluster.py --signal embeddings --param 60
```

- [ ] **Step 5 : Commit de validation**

```bash
git add -A
git commit -m "chore: clustering de validation (embeddings, param=60)"
```
