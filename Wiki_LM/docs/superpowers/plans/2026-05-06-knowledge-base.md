# Base de connaissance compactée — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire `knowledge_base/` — référentiel de centroïdes thématiques dérivés des clusters du wiki archivé, utilisé à l'ingestion pour enrichir le contexte LLM.

**Architecture:** Trois outils indépendants (`kb_query.py`, `kb_update.py`, `kb_tags.py`) écrivant dans `~/Secretarius/Wiki_LM/knowledge_base/`. Les axes sont des fichiers Markdown (même format que les `cluster-*.md` existants) accompagnés d'une matrice numpy de centroïdes. `kb_update.py` est le seul écrivain ; les lecteurs (`kb_query.py`, ingestion) ne font que lire.

**Tech Stack:** Python 3.11, numpy, python-frontmatter, sentence-transformers (BAAI/bge-m3 déjà en place), pytest.

**Spec:** `docs/superpowers/specs/2026-05-06-knowledge-base-design.md`

---

## Fichiers

| Fichier | Action | Rôle |
|---------|--------|------|
| `tools/kb_query.py` | Créer | Requête de proximité (lecture seule) |
| `tools/kb_update.py` | Créer | Mise à jour de la base depuis un clustering archivé |
| `tools/kb_tags.py` | Créer | Construction du dictionnaire de tags normalisés |
| `tests/test_kb_query.py` | Créer | Tests kb_query |
| `tests/test_kb_update.py` | Créer | Tests kb_update |
| `tests/test_kb_tags.py` | Créer | Tests kb_tags |
| `.gitignore` | Modifier | Exclure `Wiki_LM/knowledge_base/` |

---

## Contexte codebase pour les sous-agents

### Structure du wiki

```
~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026/
├── _meta.yaml
└── wiki/
    ├── sources/src-*.md          ← pages avec frontmatter (title, tags, ...)
    └── clusterings/
        └── clustering-embeddings-transfers-0.403/
            ├── cluster-embeddings-transfers-30-0001.md
            └── ...
```

### Format d'un fichier cluster-*.md (écrit par cluster.py)

```
---
category: cluster
signal: embeddings
algo: transfers
param: 30
members: 14          ← count seulement, pas la liste
paragon: src-0042
created: 2026-05-06
---

# Philosophie du langage

Description en prose.

## Parangon

[[src-0042]] — Titre de la page

## Documents membres

- [[src-0042]] — Titre
- [[src-0117]] — Titre

## Clusters proches

- [[cluster-...]] (similarité : 0.72)
```

**Note** : pas de champ `cohesion` ni `tags` dans le frontmatter des clusters — ils sont calculés par `kb_update.py`.

### Embeddings

- Matrice : `~/Secretarius/Wiki_LM/embeddings/embeddings.npy` — shape `(N, 1024)`, vecteurs L2-normalisés
- Index : `~/Secretarius/Wiki_LM/embeddings/embeddings_index.json` — `{"slugs": ["src-0001", ...]}`
- Les slugs sont les mêmes dans le wiki archivé et dans les embeddings

### Imports disponibles

```python
import frontmatter          # frontmatter.load(path), post.content, post.get("key")
import numpy as np
from pathlib import Path
import json, re, argparse
from datetime import date
```

`sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))` dans les tests.

---

## Tâche 1 : `kb_query.py` + `tests/test_kb_query.py`

**Files:**
- Create: `tools/kb_query.py`
- Create: `tests/test_kb_query.py`

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# tests/test_kb_query.py
"""Tests pour kb_query.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_kb(tmp_path: Path, n_axes: int = 3, dim: int = 8) -> tuple[Path, np.ndarray, list[str]]:
    """Crée une kb synthétique avec n_axes axes bien séparés."""
    kb_dir = tmp_path / "kb"
    (kb_dir / "axes").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir(parents=True)

    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_axes, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(kb_dir / "embeddings" / "axes.npy", mat)

    ids = [f"axis-{i + 1:04d}" for i in range(n_axes)]
    (kb_dir / "embeddings" / "axes_index.json").write_text(
        json.dumps({"ids": ids}), encoding="utf-8"
    )
    for i, aid in enumerate(ids):
        (kb_dir / "axes" / f"{aid}.md").write_text(
            f"---\ntitle: Axe {i + 1}\ntags: [tag{i}]\n---\n\nDescription.\n",
            encoding="utf-8",
        )
    return kb_dir, mat, ids


def test_kb_query_returns_top_k():
    pass  # remplacé dans step 3


def test_kb_query_scores_decreasing():
    pass


def test_kb_query_empty_kb(tmp_path):
    from kb_query import kb_query
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    result = kb_query(np.ones(8, dtype=np.float32), kb_dir, top_k=3)
    assert result == []
```

- [ ] **Step 2 : Vérifier que l'import échoue**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_query.py -v 2>&1 | head -20
```
Attendu : `ModuleNotFoundError: No module named 'kb_query'`

- [ ] **Step 3 : Créer `tools/kb_query.py`**

```python
# tools/kb_query.py
"""Requête de proximité aux axes de la base de connaissance."""
from __future__ import annotations

import json
from pathlib import Path

import frontmatter
import numpy as np


def kb_query(
    vec: np.ndarray,
    kb_dir: Path,
    top_k: int = 3,
) -> list[dict]:
    """
    Retourne les top_k axes les plus proches du vecteur vec.

    vec doit être L2-normalisé (même convention que les embeddings BGE-M3).

    Returns:
        [{"id": "axis-0001", "title": "...", "score": 0.81, "tags": [...]}, ...]
        Liste vide si la base est inexistante ou vide.
    """
    axes_npy = kb_dir / "embeddings" / "axes.npy"
    axes_index = kb_dir / "embeddings" / "axes_index.json"

    if not axes_npy.exists() or not axes_index.exists():
        return []

    matrix = np.load(axes_npy)                                          # (K, dim)
    ids: list[str] = json.loads(axes_index.read_text(encoding="utf-8"))["ids"]

    scores = matrix @ vec                                                # (K,) cosine sim
    k = min(top_k, len(ids))
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        axis_id = ids[int(i)]
        axis_path = kb_dir / "axes" / f"{axis_id}.md"
        title = axis_id
        tags: list[str] = []
        if axis_path.exists():
            post = frontmatter.load(axis_path)
            title = str(post.get("title", axis_id))
            tags = list(post.get("tags", []))
        results.append({
            "id": axis_id,
            "title": title,
            "score": float(scores[int(i)]),
            "tags": tags,
        })
    return results
```

- [ ] **Step 4 : Écrire les vrais tests**

Remplacer le contenu de `tests/test_kb_query.py` par :

```python
# tests/test_kb_query.py
"""Tests pour kb_query.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_kb(tmp_path: Path, n_axes: int = 3, dim: int = 8) -> tuple[Path, np.ndarray, list[str]]:
    kb_dir = tmp_path / "kb"
    (kb_dir / "axes").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir(parents=True)

    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_axes, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(kb_dir / "embeddings" / "axes.npy", mat)

    ids = [f"axis-{i + 1:04d}" for i in range(n_axes)]
    (kb_dir / "embeddings" / "axes_index.json").write_text(
        json.dumps({"ids": ids}), encoding="utf-8"
    )
    for i, aid in enumerate(ids):
        (kb_dir / "axes" / f"{aid}.md").write_text(
            f"---\ntitle: Axe {i + 1}\ntags: [tag{i}]\n---\n\nDescription.\n",
            encoding="utf-8",
        )
    return kb_dir, mat, ids


def test_kb_query_returns_top_k(tmp_path):
    from kb_query import kb_query
    kb_dir, mat, ids = _make_kb(tmp_path, n_axes=3)
    vec = mat[0].copy()                        # proche de axis-0001
    results = kb_query(vec, kb_dir, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "axis-0001"     # plus proche = lui-même
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_kb_query_scores_decreasing(tmp_path):
    from kb_query import kb_query
    kb_dir, mat, _ = _make_kb(tmp_path, n_axes=3)
    vec = mat[1].copy()
    results = kb_query(vec, kb_dir, top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_kb_query_includes_title_and_tags(tmp_path):
    from kb_query import kb_query
    kb_dir, mat, _ = _make_kb(tmp_path, n_axes=3)
    results = kb_query(mat[0].copy(), kb_dir, top_k=1)
    assert results[0]["title"] == "Axe 1"
    assert results[0]["tags"] == ["tag0"]


def test_kb_query_empty_kb(tmp_path):
    from kb_query import kb_query
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    result = kb_query(np.ones(8, dtype=np.float32), kb_dir, top_k=3)
    assert result == []
```

- [ ] **Step 5 : Lancer les tests**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_query.py -v
```
Attendu : `4 passed`

- [ ] **Step 6 : Commit**

```bash
git add tools/kb_query.py tests/test_kb_query.py
git commit -m "feat: kb_query — requête de proximité aux axes de la base de connaissance"
```

---

## Tâche 2 : `kb_update.py` — chargement + sélection + centroïdes + tests

**Files:**
- Create: `tools/kb_update.py`
- Create: `tests/test_kb_update.py`

- [ ] **Step 1 : Écrire les tests de chargement et sélection**

```python
# tests/test_kb_update.py
"""Tests pour kb_update.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# ---------------------------------------------------------------------------
# Helper : wiki archivé synthétique
# ---------------------------------------------------------------------------

def _make_archived_wiki(
    tmp_path: Path,
    n_clusters: int = 2,
    members_per_cluster: int = 5,
    dim: int = 16,
) -> tuple[Path, Path, str]:
    """
    Crée un mini-wiki archivé avec n_clusters clusters et embeddings synthétiques.
    Les clusters 0..n//2-1 occupent les dim/2 premières dimensions, les autres les suivantes.
    Retourne (wiki_root, embed_dir, clustering_name).
    """
    wiki_root = tmp_path / "wiki_arch"
    wiki_root.mkdir()
    wiki_dir = wiki_root / "wiki"
    sources_dir = wiki_dir / "sources"
    sources_dir.mkdir(parents=True)

    all_slugs: list[str] = []
    for c in range(n_clusters):
        for m in range(members_per_cluster):
            slug = f"src-{c:02d}{m:02d}"
            all_slugs.append(slug)
            (sources_dir / f"{slug}.md").write_text(
                f"---\ntitle: Source {c}-{m}\ncategory: source\n"
                f"tags: [theme-{c}, test]\n---\n\n## Résumé\n\nTexte {c}-{m}.\n",
                encoding="utf-8",
            )

    clustering_name = "clustering-embeddings-transfers-0.500"
    clustering_dir = wiki_dir / "clusterings" / clustering_name
    clustering_dir.mkdir(parents=True)

    for c in range(n_clusters):
        member_slugs = [f"src-{c:02d}{m:02d}" for m in range(members_per_cluster)]
        members_lines = "\n".join(
            f"- [[{s}]] — Source {c}-{i}" for i, s in enumerate(member_slugs)
        )
        cluster_path = clustering_dir / f"cluster-embeddings-transfers-30-{c:04d}.md"
        cluster_path.write_text(
            f"---\ncategory: cluster\nsignal: embeddings\nalgo: transfers\n"
            f"param: 30\nmembers: {members_per_cluster}\n"
            f"paragon: src-{c:02d}00\ncreated: 2026-05-06\n---\n\n"
            f"# Thème {c}\n\nDescription du thème {c}.\n\n"
            f"## Parangon\n\n[[src-{c:02d}00]] — Source {c}-0\n\n"
            f"## Documents membres\n\n{members_lines}\n\n"
            f"## Clusters proches\n\n",
            encoding="utf-8",
        )

    rng = np.random.default_rng(0)
    total = n_clusters * members_per_cluster
    vecs = np.zeros((total, dim), dtype=np.float32)
    half = dim // 2
    for c in range(n_clusters):
        start, end = c * members_per_cluster, (c + 1) * members_per_cluster
        vecs[start:end, c * half : (c + 1) * half] = 1.0
    vecs += rng.standard_normal((total, dim)).astype(np.float32) * 0.05
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    np.save(embed_dir / "embeddings.npy", vecs)
    (embed_dir / "embeddings_index.json").write_text(
        json.dumps({"slugs": all_slugs}), encoding="utf-8"
    )

    return wiki_root, embed_dir, clustering_name


# ---------------------------------------------------------------------------
# Tests update_kb — premier appel (création)
# ---------------------------------------------------------------------------

def test_update_kb_creates_axes_dir(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert (kb_dir / "axes").exists()
    assert (kb_dir / "embeddings" / "axes.npy").exists()
    assert (kb_dir / "embeddings" / "axes_index.json").exists()


def test_update_kb_creates_two_axes(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    stats = update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert stats["created"] == 2
    assert stats["updated"] == 0
    axes = list((kb_dir / "axes").glob("axis-*.md"))
    assert len(axes) == 2


def test_update_kb_axis_has_correct_frontmatter(tmp_path):
    from kb_update import update_kb
    import frontmatter
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    axis_path = kb_dir / "axes" / "axis-0001.md"
    post = frontmatter.load(axis_path)
    assert post.get("title") == "Thème 0"
    assert post.get("members_count") == 5
    assert "wiki_arch" in post.get("source_wikis", [])
    assert isinstance(post.get("cohesion"), float)


def test_update_kb_axes_npy_shape(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    mat = np.load(kb_dir / "embeddings" / "axes.npy")
    assert mat.shape == (2, 16)


def test_update_kb_exclusion_too_small(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(
        tmp_path, n_clusters=2, members_per_cluster=2
    )
    kb_dir = tmp_path / "kb"
    stats = update_kb(wiki_root, clustering_name, embed_dir, kb_dir, min_size=3)
    assert stats["excluded"] == 2
    excluded = json.loads((kb_dir / "excluded.json").read_text(encoding="utf-8"))
    assert any("size" in e["reason"] for e in excluded)
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_update.py -v 2>&1 | head -20
```
Attendu : `ModuleNotFoundError: No module named 'kb_update'`

- [ ] **Step 3 : Créer `tools/kb_update.py` — fonctions utilitaires**

```python
# tools/kb_update.py
"""
Mise à jour de la base de connaissance à partir d'un wiki archivé.

Usage:
    python tools/kb_update.py \\
        --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \\
        --clustering clustering-embeddings-transfers-0.403 \\
        [--embed-dir ~/Secretarius/Wiki_LM/embeddings] \\
        [--kb-dir ~/Secretarius/Wiki_LM/knowledge_base] \\
        [--fusion-threshold 0.85] \\
        [--min-size 3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

import frontmatter
import numpy as np

_DEFAULT_EMBED_DIR = Path(__file__).resolve().parent.parent / "embeddings"
_DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
FUSION_THRESHOLD = 0.85
MIN_SIZE = 3


# ---------------------------------------------------------------------------
# Chargement des fichiers cluster
# ---------------------------------------------------------------------------

def _load_cluster_files(clustering_dir: Path) -> list[dict]:
    """
    Charge tous les cluster-*.md d'un répertoire de clustering.
    Retourne [{id, title, description, members, status}].
    """
    clusters = []
    for path in sorted(clustering_dir.glob("cluster-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue

        content = post.content

        # Titre depuis le corps (première ligne # ...)
        m_title = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = m_title.group(1).strip() if m_title else path.stem

        # Description : paragraphe après le titre
        m_desc = re.search(r"^# .+\n\n(.+?)(?=\n\n|\Z)", content, re.DOTALL | re.MULTILINE)
        description = m_desc.group(1).strip() if m_desc else ""

        # Slugs membres depuis ## Documents membres
        m_members = re.search(
            r"## Documents membres\n(.*?)(?=\n## |\Z)", content, re.DOTALL
        )
        member_slugs: list[str] = []
        if m_members:
            member_slugs = re.findall(r"\[\[([^\]]+)\]\]", m_members.group(1))

        clusters.append({
            "id": path.stem,
            "title": title,
            "description": description,
            "members": member_slugs,
            "status": str(post.get("status", "active")),
        })
    return clusters


# ---------------------------------------------------------------------------
# Tags agrégés depuis les pages sources
# ---------------------------------------------------------------------------

def _collect_tags(member_slugs: list[str], sources_dir: Path, top_n: int = 10) -> list[str]:
    """Agrège les top_n tags les plus fréquents parmi les pages membres."""
    counts: dict[str, int] = {}
    for slug in member_slugs:
        path = sources_dir / f"{slug}.md"
        if not path.exists():
            continue
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        for tag in post.get("tags", []):
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])][:top_n]


# ---------------------------------------------------------------------------
# Centroïde et cohésion depuis les embeddings
# ---------------------------------------------------------------------------

def _load_embed_matrix(embed_dir: Path) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Charge la matrice d'embeddings et l'index des slugs."""
    index_path = embed_dir / "embeddings_index.json"
    matrix_path = embed_dir / "embeddings.npy"
    if not index_path.exists() or not matrix_path.exists():
        return None, None
    matrix = np.load(matrix_path)
    slugs = json.loads(index_path.read_text(encoding="utf-8"))["slugs"]
    return matrix, slugs


def _compute_centroid_and_cohesion(
    member_slugs: list[str],
    embed_matrix: np.ndarray,
    slug_to_idx: dict[str, int],
) -> tuple[np.ndarray, float] | tuple[None, float]:
    """
    Calcule le centroïde L2-normalisé et la cohésion (sim moyenne au centroïde).
    Retourne (None, 0.0) si aucun membre n'a d'embedding.
    """
    indices = [slug_to_idx[s] for s in member_slugs if s in slug_to_idx]
    if not indices:
        return None, 0.0
    vecs = embed_matrix[indices].astype(np.float32)          # (n, dim)
    centroid = vecs.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return None, 0.0
    centroid /= norm
    cohesion = float(np.mean(vecs @ centroid))
    return centroid, cohesion


# ---------------------------------------------------------------------------
# ID du prochain axe
# ---------------------------------------------------------------------------

def _next_axis_id(axes_dir: Path) -> str:
    nums = []
    for p in axes_dir.glob("axis-*.md"):
        m = re.match(r"axis-(\d+)\.md$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return f"axis-{(max(nums) + 1 if nums else 0 + 1):04d}"
```

- [ ] **Step 4 : Ajouter `update_kb()` dans `tools/kb_update.py`**

Ajouter à la suite du fichier :

```python
# ---------------------------------------------------------------------------
# Régénération de l'index
# ---------------------------------------------------------------------------

def _regenerate_index(kb_dir: Path) -> None:
    axes = []
    for path in sorted((kb_dir / "axes").glob("axis-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        axes.append({
            "id": path.stem,
            "title": str(post.get("title", path.stem)),
            "description": str(post.get("description", "")),
            "members_count": int(post.get("members_count", 0)),
            "source_wikis": list(post.get("source_wikis", [])),
        })
    lines = ["# Base de connaissance — Index thématique\n\n"]
    for a in axes:
        lines.append(f"## [[{a['id']}]] {a['title']}\n\n")
        if a["description"]:
            lines.append(f"{a['description']}\n\n")
        lines.append(
            f"*{a['members_count']} pages — {', '.join(a['source_wikis'])}*\n\n"
        )
    (kb_dir / "index.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def update_kb(
    wiki_root: Path,
    clustering_name: str,
    embed_dir: Path,
    kb_dir: Path,
    fusion_threshold: float = FUSION_THRESHOLD,
    min_size: int = MIN_SIZE,
    min_cohesion: float | None = None,
) -> dict:
    """
    Met à jour la base de connaissance depuis un clustering archivé.
    wiki_root : répertoire racine du wiki archivé (contient wiki/).
    Retourne {"created": int, "updated": int, "excluded": int}.
    """
    clustering_dir = wiki_root / "wiki" / "clusterings" / clustering_name
    if not clustering_dir.exists():
        raise FileNotFoundError(f"Clustering introuvable : {clustering_dir}")

    sources_dir = wiki_root / "wiki" / "sources"

    # Theta pour min_cohesion auto
    m_theta = re.search(r"-(\d+\.\d+)$", clustering_name)
    auto_theta = float(m_theta.group(1)) if m_theta else 0.0
    effective_min_cohesion = min_cohesion if min_cohesion is not None else auto_theta / 2

    # Charger les embeddings une seule fois
    embed_matrix, embed_slugs = _load_embed_matrix(embed_dir)
    slug_to_idx: dict[str, int] = (
        {s: i for i, s in enumerate(embed_slugs)} if embed_slugs else {}
    )

    # Préparer la base de connaissance
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "axes").mkdir(exist_ok=True)
    (kb_dir / "embeddings").mkdir(exist_ok=True)

    axes_npy_path = kb_dir / "embeddings" / "axes.npy"
    axes_index_path = kb_dir / "embeddings" / "axes_index.json"

    if axes_npy_path.exists() and axes_index_path.exists():
        axis_matrix: list[np.ndarray] = list(
            np.load(axes_npy_path).astype(np.float32)
        )
        axis_ids: list[str] = json.loads(
            axes_index_path.read_text(encoding="utf-8")
        )["ids"]
    else:
        axis_matrix = []
        axis_ids = []

    wiki_name = wiki_root.name
    today = date.today().isoformat()
    excluded: list[dict] = []
    stats = {"created": 0, "updated": 0, "excluded": 0}

    for cluster in _load_cluster_files(clustering_dir):
        # Critères de sélection
        if cluster["status"] == "garbage":
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name, "reason": "status=garbage"})
            stats["excluded"] += 1
            continue
        if len(cluster["members"]) < min_size:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name,
                              "reason": f"size {len(cluster['members'])} < {min_size}"})
            stats["excluded"] += 1
            continue

        # Centroïde et cohésion
        if embed_matrix is None:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name, "reason": "no embeddings"})
            stats["excluded"] += 1
            continue
        centroid, cohesion = _compute_centroid_and_cohesion(
            cluster["members"], embed_matrix, slug_to_idx
        )
        if centroid is None:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name, "reason": "no embeddings for members"})
            stats["excluded"] += 1
            continue
        if cohesion < effective_min_cohesion:
            excluded.append({"cluster_id": cluster["id"], "wiki": wiki_name,
                              "reason": f"cohesion {cohesion:.3f} < {effective_min_cohesion:.3f}"})
            stats["excluded"] += 1
            continue

        # Tags
        tags = _collect_tags(cluster["members"], sources_dir)

        # Fusion ou création
        best_score = -1.0
        best_idx = -1
        if axis_matrix:
            mat = np.array(axis_matrix, dtype=np.float32)
            sims = mat @ centroid
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

        if best_score >= fusion_threshold:
            # Mise à jour de l'axe existant
            axis_id = axis_ids[best_idx]
            axis_path = kb_dir / "axes" / f"{axis_id}.md"
            post = frontmatter.load(axis_path)
            old_count = int(post.get("members_count", 1))
            new_count = old_count + len(cluster["members"])
            old_vec = axis_matrix[best_idx]
            fused = (old_vec * old_count + centroid * len(cluster["members"])) / new_count
            n = float(np.linalg.norm(fused))
            axis_matrix[best_idx] = (fused / n if n > 0 else fused).astype(np.float32)
            src_wikis = list(post.get("source_wikis", []))
            if wiki_name not in src_wikis:
                src_wikis.append(wiki_name)
            existing_tags = list(post.get("tags", []))
            for t in tags:
                if t not in existing_tags:
                    existing_tags.append(t)
            post["source_wikis"] = src_wikis
            post["members_count"] = new_count
            post["cohesion"] = round(cohesion, 4)
            post["tags"] = existing_tags[:10]
            post["updated"] = today
            axis_path.write_text(frontmatter.dumps(post), encoding="utf-8")
            stats["updated"] += 1
        else:
            # Nouvel axe
            axis_id = _next_axis_id(kb_dir / "axes")
            axis_path = kb_dir / "axes" / f"{axis_id}.md"
            rep_lines = "\n".join(f"- [[{s}]]" for s in cluster["members"][:3])
            new_post = frontmatter.Post(
                f"\n## Pages représentatives\n\n{rep_lines}\n",
                title=cluster["title"],
                description=cluster["description"],
                source_wikis=[wiki_name],
                updated=today,
                members_count=len(cluster["members"]),
                cohesion=round(cohesion, 4),
                tags=tags,
                status="active",
            )
            axis_path.write_text(frontmatter.dumps(new_post), encoding="utf-8")
            axis_ids.append(axis_id)
            axis_matrix.append(centroid.astype(np.float32))
            stats["created"] += 1

    # Sauvegarder la matrice et l'index
    if axis_matrix:
        np.save(axes_npy_path, np.array(axis_matrix, dtype=np.float32))
        axes_index_path.write_text(
            json.dumps({"ids": axis_ids}), encoding="utf-8"
        )

    # excluded.json (cumulatif)
    excluded_path = kb_dir / "excluded.json"
    existing_excl: list[dict] = (
        json.loads(excluded_path.read_text(encoding="utf-8"))
        if excluded_path.exists()
        else []
    )
    excluded_path.write_text(
        json.dumps(existing_excl + excluded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _regenerate_index(kb_dir)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Met à jour la base de connaissance")
    parser.add_argument("--wiki", required=True, help="Racine du wiki archivé")
    parser.add_argument("--clustering", required=True, help="Nom du clustering (ex: clustering-embeddings-transfers-0.403)")
    parser.add_argument("--embed-dir", default=str(_DEFAULT_EMBED_DIR))
    parser.add_argument("--kb-dir", default=str(_DEFAULT_KB_DIR))
    parser.add_argument("--fusion-threshold", type=float, default=FUSION_THRESHOLD)
    parser.add_argument("--min-size", type=int, default=MIN_SIZE)
    parser.add_argument("--min-cohesion", type=float, default=None)
    args = parser.parse_args()

    stats = update_kb(
        wiki_root=Path(args.wiki),
        clustering_name=args.clustering,
        embed_dir=Path(args.embed_dir),
        kb_dir=Path(args.kb_dir),
        fusion_threshold=args.fusion_threshold,
        min_size=args.min_size,
        min_cohesion=args.min_cohesion,
    )
    print(f"[kb_update] Créés: {stats['created']}, Mis à jour: {stats['updated']}, Exclus: {stats['excluded']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5 : Lancer les tests**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_update.py -v
```
Attendu : `5 passed`

- [ ] **Step 6 : Ajouter les tests de fusion**

Ajouter à `tests/test_kb_update.py` :

```python
def test_update_kb_fusion_on_second_call(tmp_path):
    """Deuxième appel avec même clustering → axes mis à jour, pas dupliqués."""
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    stats2 = update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert stats2["created"] == 0
    assert stats2["updated"] == 2
    axes = list((kb_dir / "axes").glob("axis-*.md"))
    assert len(axes) == 2     # pas de doublon


def test_update_kb_source_wikis_accumulated(tmp_path):
    """source_wikis s'accumule entre les appels."""
    from kb_update import update_kb
    import frontmatter
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)

    # Deuxième wiki (renommé pour avoir un nom différent)
    wiki_root2 = tmp_path / "wiki_arch2"
    import shutil
    shutil.copytree(wiki_root, wiki_root2)
    update_kb(wiki_root2, clustering_name, embed_dir, kb_dir)

    post = frontmatter.load(kb_dir / "axes" / "axis-0001.md")
    assert len(post.get("source_wikis", [])) == 2


def test_update_kb_index_md_created(tmp_path):
    from kb_update import update_kb
    wiki_root, embed_dir, clustering_name = _make_archived_wiki(tmp_path, n_clusters=2)
    kb_dir = tmp_path / "kb"
    update_kb(wiki_root, clustering_name, embed_dir, kb_dir)
    assert (kb_dir / "index.md").exists()
    content = (kb_dir / "index.md").read_text(encoding="utf-8")
    assert "Thème 0" in content
    assert "Thème 1" in content
```

- [ ] **Step 7 : Lancer tous les tests kb_update**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_update.py -v
```
Attendu : `8 passed`

- [ ] **Step 8 : Lancer la suite complète**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```
Attendu : tous les tests existants passent toujours.

- [ ] **Step 9 : Commit**

```bash
git add tools/kb_update.py tests/test_kb_update.py
git commit -m "feat: kb_update — mise à jour de la base de connaissance depuis un clustering archivé"
```

---

## Tâche 3 : `kb_tags.py` + `tests/test_kb_tags.py` + `.gitignore`

**Files:**
- Create: `tools/kb_tags.py`
- Create: `tests/test_kb_tags.py`
- Modify: `.gitignore`

- [ ] **Step 1 : Écrire les tests**

```python
# tests/test_kb_tags.py
"""Tests pour kb_tags.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _make_tag_embeddings(
    tmp_path: Path,
    tags_with_counts: dict[str, int],
    groups: list[list[str]],
    dim: int = 8,
) -> tuple[Path, dict[str, np.ndarray]]:
    """
    Crée des embeddings synthétiques où les tags d'un même groupe sont proches.
    tags_with_counts : {tag: count}
    groups : [[tag1, tag2], [tag3, tag4]] → groupe 0, groupe 1
    """
    rng = np.random.default_rng(0)
    tag_list = list(tags_with_counts.keys())
    vecs: dict[str, np.ndarray] = {}

    for g_idx, group in enumerate(groups):
        base = np.zeros(dim, dtype=np.float32)
        base[g_idx] = 1.0
        for tag in group:
            v = base + rng.standard_normal(dim).astype(np.float32) * 0.05
            v /= np.linalg.norm(v)
            vecs[tag] = v

    # Tags hors groupes (hapaxes isolés)
    for tag in tag_list:
        if tag not in vecs:
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs[tag] = v

    return vecs


def _setup_kb_tags(tmp_path: Path, tags_with_counts: dict[str, int], vecs: dict[str, np.ndarray]) -> Path:
    """Crée kb/tags/ avec tags_dict.json et tags_embeddings.npy pré-remplis pour les tests."""
    kb_dir = tmp_path / "kb"
    (kb_dir / "tags").mkdir(parents=True)
    tag_list = list(tags_with_counts.keys())
    mat = np.array([vecs[t] for t in tag_list], dtype=np.float32)
    np.save(kb_dir / "tags" / "tags_embeddings.npy", mat)
    (kb_dir / "tags" / "tags_dict.json").write_text(
        json.dumps({"tags": tag_list, "counts": tags_with_counts}), encoding="utf-8"
    )
    return kb_dir


def test_build_tag_groups_synonyms(tmp_path):
    """Tags synonymes (proches) → regroupés sous le même canonique."""
    from kb_tags import build_tag_groups
    tags = {"python": 10, "Python": 8, "programmation": 5, "cuisine": 3, "Cuisine": 2}
    groups = [["python", "Python"], ["cuisine", "Cuisine"]]
    vecs = _make_tag_embeddings(tmp_path, tags, groups)
    result = build_tag_groups(tags, vecs, threshold=0.90)
    # "python" et "Python" dans le même groupe
    canonical_map = {variant: canon for canon, variants in result.items() for variant in variants}
    assert canonical_map["python"] == canonical_map["Python"]
    assert canonical_map["cuisine"] == canonical_map["Cuisine"]
    assert canonical_map["python"] != canonical_map["cuisine"]


def test_build_tag_groups_hapax_excluded(tmp_path):
    """Hapaxes (count=1) exclus par défaut."""
    from kb_tags import build_tag_groups
    tags = {"rare": 1, "commun": 5}
    groups: list[list[str]] = []
    vecs = _make_tag_embeddings(tmp_path, tags, groups)
    result = build_tag_groups(tags, vecs, threshold=0.90, min_count=2)
    all_variants = [v for variants in result.values() for v in variants]
    assert "rare" not in all_variants


def test_build_tag_groups_keep_hapax(tmp_path):
    """min_count=1 → hapaxes conservés."""
    from kb_tags import build_tag_groups
    tags = {"rare": 1, "commun": 5}
    groups: list[list[str]] = []
    vecs = _make_tag_embeddings(tmp_path, tags, groups)
    result = build_tag_groups(tags, vecs, threshold=0.90, min_count=1)
    all_variants = [v for variants in result.values() for v in variants]
    assert "rare" in all_variants


def test_save_tag_dict(tmp_path):
    """save_tag_dict écrit tags_dict.json et tags_embeddings.npy."""
    from kb_tags import save_tag_dict
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "tags").mkdir()
    groups = {"python": ["python", "Python"], "cuisine": ["cuisine"]}
    vecs = {"python": np.array([1.0, 0.0], dtype=np.float32),
            "Python": np.array([0.99, 0.01], dtype=np.float32),
            "cuisine": np.array([0.0, 1.0], dtype=np.float32)}
    save_tag_dict(kb_dir, groups, vecs)
    assert (kb_dir / "tags" / "tags_dict.json").exists()
    assert (kb_dir / "tags" / "tags_embeddings.npy").exists()
    d = json.loads((kb_dir / "tags" / "tags_dict.json").read_text(encoding="utf-8"))
    assert "python" in d
    assert "Python" in d["python"]
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_tags.py -v 2>&1 | head -20
```
Attendu : `ModuleNotFoundError: No module named 'kb_tags'`

- [ ] **Step 3 : Créer `tools/kb_tags.py`**

```python
# tools/kb_tags.py
"""
Construction du dictionnaire de tags normalisés par similarité sémantique.

Usage:
    python tools/kb_tags.py \\
        --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \\
        [--kb-dir ~/Secretarius/Wiki_LM/knowledge_base] \\
        [--threshold 0.90] \\
        [--min-count 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontmatter
import numpy as np

_DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


# ---------------------------------------------------------------------------
# Collecte des tags depuis un wiki
# ---------------------------------------------------------------------------

def collect_tags(wiki_root: Path) -> dict[str, int]:
    """Retourne {tag: count} depuis toutes les pages sources du wiki."""
    counts: dict[str, int] = {}
    sources_dir = wiki_root / "wiki" / "sources"
    if not sources_dir.exists():
        return counts
    for path in sorted(sources_dir.glob("src-*.md")):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        for tag in post.get("tags", []):
            t = str(tag)
            counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Regroupement des tags par similarité
# ---------------------------------------------------------------------------

def build_tag_groups(
    tags: dict[str, int],
    vecs: dict[str, np.ndarray],
    threshold: float = 0.90,
    min_count: int = 2,
) -> dict[str, list[str]]:
    """
    Regroupe les tags synonymes par similarité cosinus.

    tags      : {tag: count}
    vecs      : {tag: vecteur L2-normalisé}
    threshold : similarité cosinus minimum pour fusionner deux tags
    min_count : count minimum pour conserver un tag (hapaxes si min_count > 1)

    Retourne {canonical: [variant, ...]}.
    Le canonique est le tag avec le count le plus élevé du groupe.
    """
    # Filtrer par min_count
    filtered = {t: c for t, c in tags.items() if c >= min_count and t in vecs}
    if not filtered:
        return {}

    # Tri par count décroissant pour traiter les plus fréquents en premier
    sorted_tags = sorted(filtered.keys(), key=lambda t: -filtered[t])
    assigned: dict[str, str] = {}   # tag → canonical

    for tag in sorted_tags:
        if tag in assigned:
            continue
        assigned[tag] = tag
        v = vecs[tag]
        for other in sorted_tags:
            if other in assigned:
                continue
            sim = float(np.dot(v, vecs[other]))
            if sim >= threshold:
                assigned[other] = tag

    groups: dict[str, list[str]] = {}
    for tag, canon in assigned.items():
        groups.setdefault(canon, []).append(tag)
    return groups


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_tag_dict(
    kb_dir: Path,
    groups: dict[str, list[str]],
    vecs: dict[str, np.ndarray],
) -> None:
    """
    Écrit tags_dict.json et tags_embeddings.npy dans kb_dir/tags/.
    tags_dict.json : {canonical: [variants...]}
    tags_embeddings.npy : (K, dim) — vecteurs des canoniques
    """
    (kb_dir / "tags").mkdir(parents=True, exist_ok=True)

    canonicals = sorted(groups.keys())
    mat_rows = []
    for canon in canonicals:
        if canon in vecs:
            mat_rows.append(vecs[canon].astype(np.float32))

    if mat_rows:
        np.save(kb_dir / "tags" / "tags_embeddings.npy", np.array(mat_rows))

    (kb_dir / "tags" / "tags_dict.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le dictionnaire de tags normalisés")
    parser.add_argument("--wiki", required=True)
    parser.add_argument("--kb-dir", default=str(_DEFAULT_KB_DIR))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    wiki_root = Path(args.wiki)
    kb_dir = Path(args.kb_dir)

    tags = collect_tags(wiki_root)
    print(f"[kb_tags] {len(tags)} tags trouvés")

    tag_list = [t for t, c in tags.items() if c >= args.min_count]
    print(f"[kb_tags] {len(tag_list)} tags avec count >= {args.min_count}")

    model = SentenceTransformer("BAAI/bge-m3")
    raw_vecs = model.encode(tag_list, normalize_embeddings=True, show_progress_bar=True)
    vecs = {t: raw_vecs[i].astype(np.float32) for i, t in enumerate(tag_list)}

    groups = build_tag_groups(tags, vecs, threshold=args.threshold, min_count=args.min_count)
    save_tag_dict(kb_dir, groups, vecs)
    print(f"[kb_tags] {len(groups)} groupes → {kb_dir / 'tags'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer les tests**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/test_kb_tags.py -v
```
Attendu : `4 passed`

- [ ] **Step 5 : Ajouter `knowledge_base/` au `.gitignore`**

Lire le `.gitignore` existant :

```bash
cat ~/Secretarius/.gitignore 2>/dev/null || echo "(pas de .gitignore)"
```

Ajouter la ligne `Wiki_LM/knowledge_base/` dans `~/Secretarius/.gitignore` (créer le fichier s'il n'existe pas).

- [ ] **Step 6 : Lancer la suite complète**

```bash
cd ~/Secretarius/Wiki_LM
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```
Attendu : tous les tests passent (anciens + nouveaux).

- [ ] **Step 7 : Commit**

```bash
git add tools/kb_tags.py tests/test_kb_tags.py
git add ~/Secretarius/.gitignore 2>/dev/null || git add .gitignore
git commit -m "feat: kb_tags + .gitignore knowledge_base/ — dictionnaire de tags normalisés"
```

---

## Notes d'intégration futures (hors périmètre de ce plan)

- **`ingest.py`** : appeler `kb_query(embed_vec, kb_dir, top_k=3)` lors de l'ingestion et injecter les résultats dans le prompt LLM. Faire en session séparée quand l'ingestion sera revisitée.
- **`kb_lint.py`** : détecter la dérive des centroïdes, les doublons d'axes, les axes orphelins. À spécifier séparément.
- **Premier usage réel** :
  ```bash
  python tools/kb_update.py \
    --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \
    --clustering clustering-embeddings-transfers-0.403 \
    --min-size 3
  ```
