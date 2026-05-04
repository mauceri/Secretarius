# Wiki Structure Restructuration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réorganiser `wiki/` plat en 4 sous-répertoires (`sources/`, `concepts/`, `entités/`, `clusterings/`) et mettre à jour tous les outils pour utiliser la nouvelle structure.

**Architecture:** Un nouveau module `tools/wiki_paths.py` fournit `slug_to_path()`, `iter_pages()`, et les constantes de sous-répertoires. Tous les outils importent de `wiki_paths` pour la résolution chemin → slug. Un script de migration déplace les fichiers existants. Les wikilinks Obsidian `[[slug]]` survivent sans modification car Obsidian résout par nom de fichier.

**Tech Stack:** Python 3.11, pathlib, python-frontmatter, pytest

---

## Fichiers concernés

| Action | Chemin |
|--------|--------|
| Créer | `tools/wiki_paths.py` |
| Créer | `tests/test_wiki_paths.py` |
| Créer | `tools/migrate_wiki_structure.py` |
| Modifier | `tests/conftest.py` |
| Modifier | `tests/test_ingest.py` |
| Modifier | `tests/test_similarity.py` |
| Modifier | `tests/test_cluster.py` |
| Modifier | `tools/similarity.py` |
| Modifier | `tools/cluster.py` |
| Modifier | `tools/embed.py` |
| Modifier | `tools/search.py` |
| Modifier | `tools/ingest.py` |
| Modifier | `tools/dedup.py` |

Répertoire de travail pour toutes les commandes : `~/Secretarius/Wiki_LM`

---

## Task 1 : wiki_paths.py — module utilitaire

**Files:**
- Create: `tests/test_wiki_paths.py`
- Create: `tools/wiki_paths.py`

- [ ] **Step 1 : Écrire les tests**

Créer `tests/test_wiki_paths.py` :

```python
"""Tests pour wiki_paths.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def test_subdir_for_src():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("src-foo-bar") == "sources"


def test_subdir_for_concept():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("c-zettelkasten") == "concepts"


def test_subdir_for_entity():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("e-bush") == "entités"


def test_subdir_for_cluster():
    from wiki_paths import subdir_for_slug
    assert subdir_for_slug("cluster-embeddings-0000") == "clusterings"


def test_slug_to_path_src(tmp_path):
    from wiki_paths import slug_to_path
    wiki = tmp_path / "wiki"
    assert slug_to_path(wiki, "src-foo") == wiki / "sources" / "src-foo.md"


def test_slug_to_path_concept(tmp_path):
    from wiki_paths import slug_to_path
    wiki = tmp_path / "wiki"
    assert slug_to_path(wiki, "c-zettelkasten") == wiki / "concepts" / "c-zettelkasten.md"


def test_slug_to_path_entity(tmp_path):
    from wiki_paths import slug_to_path
    wiki = tmp_path / "wiki"
    assert slug_to_path(wiki, "e-bush") == wiki / "entités" / "e-bush.md"


def test_find_page_exists(tmp_path):
    from wiki_paths import find_page
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "src-foo.md").write_text("---\ntitle: Foo\n---\n", encoding="utf-8")
    p = find_page(wiki, "src-foo")
    assert p == wiki / "sources" / "src-foo.md"


def test_find_page_not_exists(tmp_path):
    from wiki_paths import find_page
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    assert find_page(wiki, "src-nonexistent") is None


def test_iter_pages_all_subdirs(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    for sd in ("sources", "concepts", "entités"):
        (wiki / sd).mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("a", encoding="utf-8")
    (wiki / "concepts" / "c-b.md").write_text("b", encoding="utf-8")
    (wiki / "entités" / "e-c.md").write_text("c", encoding="utf-8")
    (wiki / "index.md").write_text("index", encoding="utf-8")  # ne doit PAS être inclus

    paths = list(iter_pages(wiki))
    names = {p.name for p in paths}
    assert "src-a.md" in names
    assert "c-b.md" in names
    assert "e-c.md" in names
    assert "index.md" not in names
    assert len(paths) == 3


def test_iter_pages_with_prefix(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("a", encoding="utf-8")
    (wiki / "sources" / "src-b.md").write_text("b", encoding="utf-8")
    paths = list(iter_pages(wiki, prefix="src-"))
    assert len(paths) == 2


def test_iter_pages_with_subdirs(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("a", encoding="utf-8")
    (wiki / "concepts" / "c-b.md").write_text("b", encoding="utf-8")
    paths = list(iter_pages(wiki, subdirs=["concepts"]))
    assert len(paths) == 1
    assert paths[0].name == "c-b.md"


def test_iter_pages_missing_subdir(tmp_path):
    from wiki_paths import iter_pages
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    # Aucun sous-répertoire créé
    paths = list(iter_pages(wiki))
    assert paths == []
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
.venv/bin/python -m pytest tests/test_wiki_paths.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'wiki_paths'`

- [ ] **Step 3 : Créer tools/wiki_paths.py**

```python
"""
Utilitaires de navigation dans la structure du wiki.

Structure attendue :
    wiki/
      sources/      ← src-*.md
      concepts/     ← c-*.md
      entités/      ← e-*.md
      clusterings/  ← clustering-*/
      index.md, log.md, tags.md  ← méta, restent à la racine
"""

from __future__ import annotations

from pathlib import Path

CONTENT_SUBDIRS: list[str] = ["sources", "concepts", "entités"]
CLUSTERING_SUBDIR: str = "clusterings"

_PREFIX_TO_SUBDIR: dict[str, str] = {
    "src-": "sources",
    "c-": "concepts",
    "e-": "entités",
    "cluster-": "clusterings",
}


def subdir_for_slug(slug: str) -> str:
    """Retourne le sous-répertoire attendu pour un slug donné."""
    for prefix, subdir in _PREFIX_TO_SUBDIR.items():
        if slug.startswith(prefix):
            return subdir
    return "sources"


def slug_to_path(wiki_dir: Path, slug: str) -> Path:
    """Retourne le chemin attendu pour un slug (sans vérifier l'existence)."""
    return wiki_dir / subdir_for_slug(slug) / f"{slug}.md"


def find_page(wiki_dir: Path, slug: str) -> Path | None:
    """Retourne le chemin d'une page si elle existe, None sinon."""
    path = slug_to_path(wiki_dir, slug)
    return path if path.exists() else None


def iter_pages(
    wiki_dir: Path,
    subdirs: list[str] | None = None,
    prefix: str | None = None,
):
    """
    Itère sur les pages de contenu du wiki (hors méta, hors clusterings).

    subdirs : sous-répertoires à parcourir (défaut : CONTENT_SUBDIRS)
    prefix  : filtre par préfixe de fichier, ex. "src-", "c-"
    """
    pattern = f"{prefix}*.md" if prefix else "*.md"
    for sd in (subdirs or CONTENT_SUBDIRS):
        d = wiki_dir / sd
        if not d.exists():
            continue
        yield from sorted(d.glob(pattern))
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
.venv/bin/python -m pytest tests/test_wiki_paths.py -v
```

Expected: 13 tests PASSED

- [ ] **Step 5 : Commit**

```bash
git add tools/wiki_paths.py tests/test_wiki_paths.py
git commit -m "feat: wiki_paths.py — module utilitaire slug_to_path, iter_pages (TDD)"
```

---

## Task 2 : Mettre à jour les fixtures et tests existants

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_ingest.py`
- Modify: `tests/test_similarity.py`
- Modify: `tests/test_cluster.py`

- [ ] **Step 1 : Mettre à jour conftest.py**

Dans `tests/conftest.py`, modifier la fixture `wiki_dir` pour créer les sous-répertoires :

```python
@pytest.fixture
def wiki_dir(tmp_path: Path) -> Path:
    """Répertoire wiki avec sous-répertoires et fichiers méta."""
    w = tmp_path / "wiki"
    w.mkdir()
    for subdir in ("sources", "concepts", "entités", "clusterings"):
        (w / subdir).mkdir()
    (w / "index.md").write_text("# Index\n\n", encoding="utf-8")
    (w / "log.md").write_text("", encoding="utf-8")
    return w
```

- [ ] **Step 2 : Mettre à jour test_ingest.py**

Remplacer toutes les références à des chemins plats par les sous-répertoires. Chercher et remplacer dans `tests/test_ingest.py` :

| Ancien | Nouveau |
|--------|---------|
| `wiki_dir / f"{slug}.md"` | `wiki_dir / "sources" / f"{slug}.md"` |
| `wiki_dir / "c-zettelkasten.md"` | `wiki_dir / "concepts" / "c-zettelkasten.md"` |
| `wiki_dir / "e-vannevar-bush.md"` | `wiki_dir / "entités" / "e-vannevar-bush.md"` |

Lignes précises à modifier :
- L19 : `(wiki_dir / f"{slug}.md").exists()` → `(wiki_dir / "sources" / f"{slug}.md").exists()`
- L25 : `(wiki_dir / f"{slug}.md").read_text()` → `(wiki_dir / "sources" / f"{slug}.md").read_text()`
- L48 : `(wiki_dir / "c-zettelkasten.md").exists()` → `(wiki_dir / "concepts" / "c-zettelkasten.md").exists()`
- L49 : `(wiki_dir / "e-vannevar-bush.md").exists()` → `(wiki_dir / "entités" / "e-vannevar-bush.md").exists()`
- L61 : `(wiki_dir / f"{slug}.md").read_text()` → `(wiki_dir / "sources" / f"{slug}.md").read_text()`
- L67 : `page = wiki_dir / "c-zettelkasten.md"` → `page = wiki_dir / "concepts" / "c-zettelkasten.md"`
- L78 : `page = wiki_dir / "c-zettelkasten.md"` → `page = wiki_dir / "concepts" / "c-zettelkasten.md"`

- [ ] **Step 3 : Mettre à jour test_similarity.py**

Dans `tests/test_similarity.py`, la fonction `_write_page` crée des pages dans `wiki_dir` directement. La modifier pour écrire dans le bon sous-répertoire selon le préfixe du slug :

Remplacer :
```python
def _write_page(wiki_dir: Path, slug: str, body: str, tags: list[str] | None = None) -> None:
    tags_yaml = f"tags: {tags}\n" if tags else ""
    (wiki_dir / f"{slug}.md").write_text(
        f"---\ntitle: {slug}\n{tags_yaml}---\n\n{body}", encoding="utf-8"
    )
```

Par :
```python
def _write_page(wiki_dir: Path, slug: str, body: str, tags: list[str] | None = None) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from wiki_paths import slug_to_path
    tags_yaml = f"tags: {tags}\n" if tags else ""
    path = slug_to_path(wiki_dir, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {slug}\n{tags_yaml}---\n\n{body}", encoding="utf-8")
```

- [ ] **Step 4 : Mettre à jour test_cluster.py**

Dans `tests/test_cluster.py`, la fonction `_setup_wiki_with_embeds` crée des pages `src-` directement dans `wiki_dir`. La modifier :

Remplacer :
```python
    for i, slug in enumerate(slugs):
        (wiki_dir / f"{slug}.md").write_text(
            f"---\ntitle: Source {i}\ncategory: source\ntags: [test]\n---\n\n"
            f"## Résumé\n\nTexte de test numéro {i}.\n",
            encoding="utf-8",
        )
```

Par :
```python
    sources_dir = wiki_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    for i, slug in enumerate(slugs):
        (sources_dir / f"{slug}.md").write_text(
            f"---\ntitle: Source {i}\ncategory: source\ntags: [test]\n---\n\n"
            f"## Résumé\n\nTexte de test numéro {i}.\n",
            encoding="utf-8",
        )
```

- [ ] **Step 5 : Vérifier que les tests échouent (outils pas encore mis à jour)**

```bash
.venv/bin/python -m pytest tests/test_ingest.py tests/test_similarity.py tests/test_cluster.py -v --tb=short -q 2>&1 | tail -15
```

Expected: plusieurs FAILED (les outils cherchent encore les fichiers au mauvais endroit)

- [ ] **Step 6 : Commit des fixtures**

```bash
git add tests/conftest.py tests/test_ingest.py tests/test_similarity.py tests/test_cluster.py
git commit -m "test: adapter fixtures et tests à la nouvelle structure wiki/"
```

---

## Task 3 : similarity.py + cluster.py — résolution des chemins

**Files:**
- Modify: `tools/similarity.py`
- Modify: `tools/cluster.py`

- [ ] **Step 1 : Mettre à jour similarity.py**

Dans `tools/similarity.py`, ajouter l'import de `slug_to_path` en tête de fichier, après les imports existants :

```python
from wiki_paths import slug_to_path
```

Dans `CoLinkSimilarity._links`, remplacer :
```python
        path = self._wiki_dir / f"{slug}.md"
```
par :
```python
        path = slug_to_path(self._wiki_dir, slug)
```

Dans `TagSimilarity._tags`, remplacer :
```python
        path = self._wiki_dir / f"{slug}.md"
```
par :
```python
        path = slug_to_path(self._wiki_dir, slug)
```

- [ ] **Step 2 : Mettre à jour cluster.py**

Dans `tools/cluster.py`, ajouter l'import :

```python
from wiki_paths import iter_pages, CLUSTERING_SUBDIR
```

Dans `_load_src_pages`, remplacer :
```python
    for path in sorted(wiki_dir.glob("src-*.md")):
```
par :
```python
    for path in sorted((wiki_dir / "sources").glob("src-*.md")):
```

Dans `run_clustering`, remplacer :
```python
    out_dir = wiki_dir / f"clustering-{signal_str}-{algo}-{param}"
```
par :
```python
    out_dir = wiki_dir / CLUSTERING_SUBDIR / f"clustering-{signal_str}-{algo}-{param}"
```

- [ ] **Step 3 : Vérifier que les tests similarity et cluster passent**

```bash
.venv/bin/python -m pytest tests/test_similarity.py tests/test_cluster.py -v --tb=short -q 2>&1 | tail -10
```

Expected: tous les tests PASSED

- [ ] **Step 4 : Commit**

```bash
git add tools/similarity.py tools/cluster.py
git commit -m "feat: similarity.py, cluster.py — chemins via wiki_paths"
```

---

## Task 4 : embed.py + search.py — résolution des chemins

**Files:**
- Modify: `tools/embed.py`
- Modify: `tools/search.py`

- [ ] **Step 1 : Mettre à jour embed.py**

Dans `tools/embed.py`, ajouter l'import :

```python
from wiki_paths import iter_pages
```

Remplacer la fonction `load_pages` :

```python
def load_pages(wiki_dir: Path) -> list[dict]:
    pages = []
    for path in iter_pages(wiki_dir):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        pages.append({"slug": path.stem, "text": _extract_text(post)})
    return pages
```

Supprimer la constante `SKIP_NAMES = {"index.md", "log.md"}` (devenue inutile).

- [ ] **Step 2 : Mettre à jour search.py**

Dans `tools/search.py`, ajouter les imports en tête :

```python
from wiki_paths import iter_pages, slug_to_path
```

Dans `WikiSearch._max_mtime`, remplacer :
```python
        return max(
            (p.stat().st_mtime for p in self.wiki_dir.glob("*.md")
             if p.name not in ("index.md", "log.md")),
            default=0.0,
        )
```
par :
```python
        return max(
            (p.stat().st_mtime for p in iter_pages(self.wiki_dir)),
            default=0.0,
        )
```

Dans `WikiSearch._build_index`, remplacer :
```python
        for path in sorted(self.wiki_dir.glob("**/*.md")):
            if path.name in ("index.md", "log.md"):
                continue
```
par :
```python
        for path in iter_pages(self.wiki_dir):
```

Dans `WikiSemanticSearch.search`, remplacer :
```python
            path = self.wiki_dir / f"{slug}.md"
```
par :
```python
            path = slug_to_path(self.wiki_dir, slug)
```

- [ ] **Step 3 : Vérifier que la suite complète passe**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short -q 2>&1 | tail -10
```

Expected: tous les tests PASSED (aucune régression)

- [ ] **Step 4 : Commit**

```bash
git add tools/embed.py tools/search.py
git commit -m "feat: embed.py, search.py — chemins via wiki_paths"
```

---

## Task 5 : ingest.py — résolution des chemins

**Files:**
- Modify: `tools/ingest.py`

- [ ] **Step 1 : Ajouter les imports dans ingest.py**

Après la ligne `from llm import LLM` (ou parmi les imports locaux existants), ajouter :

```python
from wiki_paths import CONTENT_SUBDIRS, CLUSTERING_SUBDIR, iter_pages, slug_to_path
```

- [ ] **Step 2 : Créer les sous-répertoires dans __init__**

Dans `WikiIngestor.__init__`, après le bloc `for d in (self.wiki_dir, self.raw_dir):`, ajouter :

```python
        for subdir in CONTENT_SUBDIRS + [CLUSTERING_SUBDIR]:
            (self.wiki_dir / subdir).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3 : Mettre à jour _write_wiki_page**

Remplacer (ligne ~1260) :
```python
    def _write_wiki_page(self, slug: str, content: str) -> None:
        path = self.wiki_dir / f"{slug}.md"
        # Respecter le statut immuable : ne jamais écraser une page verrouillée
        if path.exists():
            try:
                existing = frontmatter.loads(path.read_text(encoding="utf-8"))
                if existing.get("status") == "immuable":
                    print(f"[ingest] Page immuable, ignorée : {slug}.md")
                    return
            except Exception:
                pass
        content = _fix_mojibake(content)
        content = _fix_yaml_scalars(content)
        known_slugs = {p.stem for p in self.wiki_dir.glob("*.md")}
        content = _normalize_links(content, known_slugs)
        # Re-sérialiser via PyYAML pour normaliser le frontmatter
        try:
            post = frontmatter.loads(content)
            content = frontmatter.dumps(post)
        except Exception:
            pass
        path.write_text(content, encoding="utf-8")
```

Par :
```python
    def _write_wiki_page(self, slug: str, content: str) -> None:
        path = slug_to_path(self.wiki_dir, slug)
        # Respecter le statut immuable : ne jamais écraser une page verrouillée
        if path.exists():
            try:
                existing = frontmatter.loads(path.read_text(encoding="utf-8"))
                if existing.get("status") == "immuable":
                    print(f"[ingest] Page immuable, ignorée : {slug}.md")
                    return
            except Exception:
                pass
        content = _fix_mojibake(content)
        content = _fix_yaml_scalars(content)
        known_slugs = {p.stem for p in iter_pages(self.wiki_dir)}
        content = _normalize_links(content, known_slugs)
        # Re-sérialiser via PyYAML pour normaliser le frontmatter
        try:
            post = frontmatter.loads(content)
            content = frontmatter.dumps(post)
        except Exception:
            pass
        path.write_text(content, encoding="utf-8")
```

- [ ] **Step 4 : Mettre à jour _update_concept_page et _update_entity_page**

Dans `_update_concept_page` (ligne ~1194), remplacer :
```python
        page_path = self.wiki_dir / f"{concept_slug}.md"
        existing = page_path.read_text(encoding="utf-8") if page_path.exists() else ""
```
par :
```python
        page_file = slug_to_path(self.wiki_dir, concept_slug)
        existing = page_file.read_text(encoding="utf-8") if page_file.exists() else ""
```

Dans `_update_entity_page` (ligne ~1229), remplacer :
```python
        page_path = self.wiki_dir / f"{entity_slug}.md"
        existing = page_path.read_text(encoding="utf-8") if page_path.exists() else ""
```
par :
```python
        page_file = slug_to_path(self.wiki_dir, entity_slug)
        existing = page_file.read_text(encoding="utf-8") if page_file.exists() else ""
```

- [ ] **Step 5 : Mettre à jour _sync_deletions**

Deux remplacements dans `_sync_deletions` :

Remplacer (ligne ~774) :
```python
            src_page = self.wiki_dir / f"{slug}.md"
```
par :
```python
            src_page = slug_to_path(self.wiki_dir, slug)
```

Remplacer (ligne ~793) :
```python
        for page in sorted(self.wiki_dir.glob("*.md")):
            if not any(page.stem.startswith(p) for p in ("c-", "e-")):
                continue
```
par :
```python
        for page in iter_pages(self.wiki_dir, subdirs=["concepts", "entités"]):
```

- [ ] **Step 6 : Mettre à jour _reset_wiki**

Remplacer (ligne ~839) :
```python
        for page in self.wiki_dir.glob("*.md"):
            if page.name in _META:
                continue
            if not any(page.stem.startswith(p) for p in _PAGE_PREFIXES):
                continue
```
par :
```python
        for page in iter_pages(self.wiki_dir):
```

(La constante `_PAGE_PREFIXES` et le filtre deviennent inutiles car `iter_pages` ne retourne que des pages de contenu. La constante `_META` est encore utilisée pour la protection de `index.md`, donc la supprimer seulement si elle n'est plus utilisée ailleurs — vérifier avant.)

- [ ] **Step 7 : Mettre à jour _rebuild_tags**

Remplacer (ligne ~1168) :
```python
        for page in sorted(self.wiki_dir.glob("*.md")):
```
par :
```python
        for page in sorted(iter_pages(self.wiki_dir)):
```

- [ ] **Step 8 : Vérifier que les tests ingest passent**

```bash
.venv/bin/python -m pytest tests/test_ingest.py -v --tb=short 2>&1 | tail -15
```

Expected: tous les tests PASSED

- [ ] **Step 9 : Vérifier la suite complète**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: tous les tests PASSED

- [ ] **Step 10 : Commit**

```bash
git add tools/ingest.py
git commit -m "feat: ingest.py — chemins via wiki_paths"
```

---

## Task 6 : dedup.py — résolution des chemins

**Files:**
- Modify: `tools/dedup.py`

- [ ] **Step 1 : Ajouter les imports dans dedup.py**

Après les imports existants, ajouter :

```python
from wiki_paths import iter_pages, slug_to_path
```

- [ ] **Step 2 : Mettre à jour _select_canonical**

Remplacer (ligne ~89) :
```python
        path = wiki_dir / f"{slug}.md"
```
par :
```python
        path = slug_to_path(wiki_dir, slug)
```

- [ ] **Step 3 : Mettre à jour _build_sources_index**

Remplacer (ligne ~145) :
```python
    for page in sorted(wiki_dir.glob("*.md")):
        if not (page.stem.startswith("c-") or page.stem.startswith("e-")):
            continue
```
par :
```python
    for page in iter_pages(wiki_dir, subdirs=["concepts", "entités"]):
```

- [ ] **Step 4 : Mettre à jour _clean**

Dans `_clean`, chercher toute occurrence de `wiki_dir / f"{slug}.md"` et remplacer par `slug_to_path(wiki_dir, slug)`.

Chercher avec :
```bash
grep -n 'wiki_dir / f"' ~/Secretarius/Wiki_LM/tools/dedup.py
```

Remplacer chaque occurrence trouvée.

- [ ] **Step 5 : Vérifier que la suite complète passe**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: tous les tests PASSED

- [ ] **Step 6 : Commit**

```bash
git add tools/dedup.py
git commit -m "feat: dedup.py — chemins via wiki_paths"
```

---

## Task 7 : Script de migration

**Files:**
- Create: `tools/migrate_wiki_structure.py`

- [ ] **Step 1 : Créer tools/migrate_wiki_structure.py**

```python
"""
Migration ponctuelle : réorganise wiki/ plat en sous-répertoires.

  sources/     ← src-*.md
  concepts/    ← c-*.md
  entités/     ← e-*.md
  clusterings/ ← dossiers clustering-*/

Le script est idempotent : les fichiers déjà dans leurs sous-répertoires sont ignorés.

Usage :
    python tools/migrate_wiki_structure.py [--wiki PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def migrate(wiki_dir: Path, dry_run: bool = False) -> dict[str, int]:
    """
    Déplace les fichiers vers leurs sous-répertoires.
    Retourne un dict {subdir: nombre_de_fichiers_déplacés}.
    """
    moves: dict[str, list[tuple[Path, Path]]] = {
        "sources": [],
        "concepts": [],
        "entités": [],
        "clusterings": [],
    }

    # Pages .md à la racine de wiki/
    for page in sorted(wiki_dir.glob("*.md")):
        stem = page.stem
        if stem.startswith("src-"):
            moves["sources"].append((page, wiki_dir / "sources" / page.name))
        elif stem.startswith("c-"):
            moves["concepts"].append((page, wiki_dir / "concepts" / page.name))
        elif stem.startswith("e-"):
            moves["entités"].append((page, wiki_dir / "entités" / page.name))

    # Répertoires clustering-* à la racine de wiki/
    for d in sorted(wiki_dir.iterdir()):
        if d.is_dir() and d.name.startswith("clustering-"):
            moves["clusterings"].append((d, wiki_dir / "clusterings" / d.name))

    counts: dict[str, int] = {}
    for subdir, pairs in moves.items():
        dest_dir = wiki_dir / subdir
        if not dry_run:
            dest_dir.mkdir(exist_ok=True)
        count = 0
        for src, dst in pairs:
            if dst.exists():
                continue  # déjà migré
            tag = "[dry]" if dry_run else ""
            print(f"{tag} {src.name} → {subdir}/")
            if not dry_run:
                src.rename(dst)
            count += 1
        counts[subdir] = count

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migre wiki/ vers la structure en sous-répertoires")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
    )
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans déplacer")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    if not wiki_dir.exists():
        raise FileNotFoundError(f"Répertoire wiki introuvable : {wiki_dir}")

    print(f"[migrate] {'[DRY RUN] ' if args.dry_run else ''}wiki_dir={wiki_dir}")
    counts = migrate(wiki_dir, dry_run=args.dry_run)

    print("\n[migrate] Résumé :")
    for subdir, n in counts.items():
        print(f"  {subdir:12s} : {n} fichier(s) déplacé(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Test à sec (dry-run)**

```bash
.venv/bin/python tools/migrate_wiki_structure.py --dry-run 2>&1 | head -20
```

Expected: liste des fichiers qui seraient déplacés (src-*, c-*, e-*, clustering-*)

- [ ] **Step 3 : Vérifier que la suite de tests passe toujours**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: tous les tests PASSED

- [ ] **Step 4 : Commit**

```bash
git add tools/migrate_wiki_structure.py
git commit -m "feat: migrate_wiki_structure.py — migration ponctuelle wiki/ → sous-répertoires"
```

---

## Task 8 : Validation sur données réelles

- [ ] **Step 1 : Lancer la migration**

```bash
.venv/bin/python tools/migrate_wiki_structure.py
```

Expected : résumé montrant ~1892 sources, ~11492 concepts, ~7331 entités, ~5 clusterings déplacés.

- [ ] **Step 2 : Vérifier la structure**

```bash
ls ~/Documents/Arbath/Wiki_LM/wiki/
ls ~/Documents/Arbath/Wiki_LM/wiki/sources/ | wc -l
ls ~/Documents/Arbath/Wiki_LM/wiki/concepts/ | wc -l
ls ~/Documents/Arbath/Wiki_LM/wiki/entités/ | wc -l
ls ~/Documents/Arbath/Wiki_LM/wiki/clusterings/ | wc -l
```

Expected : `sources/` (1892), `concepts/` (11492), `entités/` (7331), `clusterings/` (5), plus `index.md`, `log.md`, `tags.md`, `poubelle/` à la racine.

- [ ] **Step 3 : Tester les outils**

```bash
# BM25 cache invalide → reconstruction automatique
.venv/bin/python -c "
import sys; sys.path.insert(0, 'tools')
from search import WikiSearch
ws = WikiSearch('/home/mauceric/Documents/Arbath/Wiki_LM')
print(f'Pages indexées : {len(ws._pages)}')
"
```

Expected : `Pages indexées : ~20715` (toutes les pages, hors méta)

```bash
# Test ingest sur un fichier test
echo "Ceci est un test post-migration" > /tmp/test_post_migration.txt
.venv/bin/python tools/ingest.py --file /tmp/test_post_migration.txt --no-llm 2>&1 | tail -5
```

Expected : page créée dans `wiki/sources/src-test-post-migration-*.md`

- [ ] **Step 4 : Commit de validation**

```bash
git add -A
git commit -m "chore: migration wiki/ vers sous-répertoires (sources, concepts, entités, clusterings)"
```
