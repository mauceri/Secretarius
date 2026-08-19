# Grand ménage Wiki_LM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger quatre défauts accumulés dans le pipeline Wiki_LM — traçabilité des sources manquante, dé-ingestion incomplète, fichiers de capture bloqués, front-matter corrompu — via des scripts ponctuels dans `Wiki_LM/tools/`, cohérents avec le style existant (`patch_src_slugs.py`).

**Architecture:** Quatre livrables indépendants. Composant 1 corrige un script existant cassé (`patch_lien_source.py`) plutôt que d'en écrire un nouveau. Composant 2 étend `Ingestor._sync_deletions` (refactor minimal + nouveau script CLI). Composants 3 et 4 sont des scripts autonomes. Chaque script : `--dry-run` par défaut, `--apply` pour écrire.

**Tech Stack:** Python 3.12, `python-frontmatter==1.1.0`, `pytest`, bibliothèque interne `Wiki_LM/tools/` (`ingest.py`, `wiki_paths.py`).

## Global Constraints

- Chemin wiki réel : `~/Documents/Arbath/Wiki_LM` (racine), pages sous `wiki/sources/`, `wiki/concepts/`, `wiki/entités/` — jamais `wiki/*.md` directement (bug déjà rencontré).
- Chemin `raw/` réel : `~/Documents/Arbath/Wiki_LM/raw` — **pas** `~/Secretarius/Wiki_LM/raw` (n'existe pas, c'est un défaut stale de code existant).
- Champ de traçabilité = `lien_source:` (singulier, string). `sources:` est un champ différent (liste de slugs, références croisées) — ne jamais les confondre.
- Tous les scripts : `--dry-run` par défaut, `--apply` explicite pour écrire. Aucune écriture sans `--apply`.
- Tests sous `Wiki_LM/tests/`, fixtures `wiki_dir`/`raw_dir`/`wiki_root` déjà définies dans `conftest.py` (répertoires temporaires `tmp_path`, structure `sources/concepts/entités/index.md/log.md`).
- `wiki_signets_05_2026/` (dossier legacy) : ne jamais y toucher dans ce plan.

---

## File Structure

- Modify: `Wiki_LM/tools/patch_lien_source.py` — corrige les deux bugs de chemin, assouplit le filtre d'extraction.
- Create: `Wiki_LM/tools/list_unsourced_pages.py` — liste les pages sans `lien_source:` récupérable, génère un fichier de recherche manuelle.
- Modify: `Wiki_LM/tools/ingest.py` — extrait `_cascade_deleted_slugs` depuis `_sync_deletions` (réutilisable), ajoute `_remove_from_index`, corrige `_parse_raw_tags` (cause racine tags corrompus).
- Create: `Wiki_LM/tools/sync_deletions_full.py` — étend la dé-ingestion (index.md/tags.md/log.md) + option `--remove <slug>`.
- Create: `Wiki_LM/tools/retry_blocked_urls.py` — relance les 4 URLs bloquées.
- Delete: `Wiki_LM/tools/mcp_server.py` — après validation de la tâche précédente.
- Create: `Wiki_LM/tools/fix_corrupted_tags.py` — répare les 36 pages à tags corrompus.
- Test: `Wiki_LM/tests/test_patch_lien_source.py`
- Test: `Wiki_LM/tests/test_list_unsourced_pages.py`
- Test: `Wiki_LM/tests/test_sync_deletions_full.py`
- Test: `Wiki_LM/tests/test_retry_blocked_urls.py`
- Test: `Wiki_LM/tests/test_fix_corrupted_tags.py`
- Modify: `Wiki_LM/tests/test_ingest.py` — ajoute un test pour le `_parse_raw_tags` corrigé.

---

## Task 1: Corriger `patch_lien_source.py`

**Files:**
- Modify: `Wiki_LM/tools/patch_lien_source.py`
- Test: `Wiki_LM/tests/test_patch_lien_source.py`

**Interfaces:**
- Consumes: `ingest._extract_url_from_file(path: Path) -> str` (déjà existant, inchangé).
- Produces: `patch_lien_source.main()` reste le point d'entrée CLI ; aucune fonction publique nouvelle consommée par les tâches suivantes.

- [ ] **Step 1: Write the failing tests**

```python
# Wiki_LM/tests/test_patch_lien_source.py
from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from patch_lien_source import _load_manifest, patch_pages


def _write_manifest(raw_dir: Path, entries: list[tuple[str, str]]) -> None:
    lines = [f"{filename}\t{slug}\thash0" for filename, slug in entries]
    (raw_dir / ".ingested").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_finds_pages_in_sources_subdir(wiki_dir: Path, raw_dir: Path):
    """Régression : les pages vivent dans wiki/sources/, pas wiki/*.md."""
    page = wiki_dir / "sources" / "src-test-page.md"
    page.write_text(
        "---\ntitle: Test\ncategory: source\ntags: [x]\n"
        "created: 2026-01-01\nsources: []\n---\n\n# Test\n",
        encoding="utf-8",
    )
    (raw_dir / "src-test-page.url").write_text("https://example.com/article\n", encoding="utf-8")
    _write_manifest(raw_dir, [("src-test-page.url", "src-test-page")])

    patched = patch_pages(wiki_dir, raw_dir, apply=False)

    assert patched == 1


def test_apply_writes_lien_source(wiki_dir: Path, raw_dir: Path):
    page = wiki_dir / "sources" / "src-test-page.md"
    page.write_text(
        "---\ntitle: Test\ncategory: source\ntags: [x]\n"
        "created: 2026-01-01\nsources: []\n---\n\n# Test\n",
        encoding="utf-8",
    )
    (raw_dir / "src-test-page.url").write_text("https://example.com/article\n", encoding="utf-8")
    _write_manifest(raw_dir, [("src-test-page.url", "src-test-page")])

    patch_pages(wiki_dir, raw_dir, apply=True)

    post = frontmatter.loads(page.read_text(encoding="utf-8"))
    assert post["lien_source"] == "https://example.com/article"


def test_relaxed_filter_extracts_from_non_url_suffix(wiki_dir: Path, raw_dir: Path):
    """Le filtre assoupli : tente l'extraction même si le fichier raw n'est pas .url."""
    page = wiki_dir / "sources" / "src-note-page.md"
    page.write_text(
        "---\ntitle: Note\ncategory: source\ntags: [x]\n"
        "created: 2026-01-01\nsources: []\n---\n\n# Note\n",
        encoding="utf-8",
    )
    # fichier raw en .md contenant quand même une URL en première ligne
    (raw_dir / "src-note-page.md").write_text(
        "https://example.com/note-source\ntags: x\n", encoding="utf-8"
    )
    _write_manifest(raw_dir, [("src-note-page.md", "src-note-page")])

    patched = patch_pages(wiki_dir, raw_dir, apply=False)

    assert patched == 1


def test_skips_page_already_patched(wiki_dir: Path, raw_dir: Path):
    page = wiki_dir / "sources" / "src-done.md"
    page.write_text(
        "---\ntitle: Done\ncategory: source\ntags: [x]\ncreated: 2026-01-01\n"
        "sources: []\nlien_source: https://already-set.example.com\n---\n\n# Done\n",
        encoding="utf-8",
    )
    (raw_dir / "src-done.url").write_text("https://different.example.com\n", encoding="utf-8")
    _write_manifest(raw_dir, [("src-done.url", "src-done")])

    patched = patch_pages(wiki_dir, raw_dir, apply=True)

    assert patched == 0
    post = frontmatter.loads(page.read_text(encoding="utf-8"))
    assert post["lien_source"] == "https://already-set.example.com"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_patch_lien_source.py -v`
Expected: FAIL — `ImportError: cannot import name 'patch_pages' from 'patch_lien_source'` (la fonction `main()` actuelle ne peut pas être testée directement, elle fait tout en une seule fonction non paramétrable).

- [ ] **Step 3: Refactor `patch_lien_source.py` — extraire `patch_pages()` et corriger les deux bugs**

Remplacer tout le contenu de `Wiki_LM/tools/patch_lien_source.py` par :

```python
"""
Patch rétroactif : ajoute lien_source aux pages src- qui en sont dépourvues.

Pour chaque page src- sans lien_source :
  1. Cherche le(s) fichier(s) raw correspondant(s) via le manifeste raw/.ingested
  2. Tente d'en extraire une URL (n'importe quel type de fichier raw, pas
     seulement .url — un fichier .md capturé peut aussi contenir une URL)
  3. Ajoute lien_source dans le frontmatter

Usage :
    python patch_lien_source.py [--apply] [--wiki PATH] [--raw PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import frontmatter

from ingest import _extract_url_from_file

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"
_DEFAULT_RAW = Path.home() / "Documents" / "Arbath" / "Wiki_LM" / "raw"
_MANIFEST = ".ingested"


def _load_manifest(raw_dir: Path) -> dict[str, list[Path]]:
    """Retourne {slug: [fichiers raw correspondants]}."""
    path = raw_dir / _MANIFEST
    if not path.exists():
        return {}
    result: dict[str, list[Path]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        filename = parts[0]
        slug = parts[1].strip() if len(parts) > 1 else ""
        if not slug:
            continue
        raw_file = raw_dir / filename
        result.setdefault(slug, []).append(raw_file)
    return result


def patch_pages(wiki_dir: Path, raw_dir: Path, apply: bool) -> int:
    """Applique le patch sur wiki_dir/sources/src-*.md. Retourne le nombre de pages patchées."""
    slug_to_raw = _load_manifest(raw_dir)
    patched = 0

    for page in sorted((wiki_dir / "sources").glob("src-*.md")):
        post = frontmatter.load(page)
        if post.get("lien_source"):
            continue

        slug = page.stem
        raw_files = slug_to_raw.get(slug, [])

        url = ""
        for rf in raw_files:
            if rf.exists():
                url = _extract_url_from_file(rf)
                if url:
                    break

        if not url:
            continue

        print(f"{'[dry]' if not apply else '[patch]'} {slug}")
        print(f"  → {url[:100]}")

        if apply:
            post["lien_source"] = url
            page.write_text(frontmatter.dumps(post), encoding="utf-8")

        patched += 1

    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch lien_source manquant sur les pages src-")
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)),
    )
    parser.add_argument("--raw", default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)))
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    raw_dir = Path(args.raw)

    patched = patch_pages(wiki_dir, raw_dir, apply=args.apply)

    print(f"\n{'Patchées' if args.apply else 'Seraient patchées'} : {patched}")
    if not args.apply:
        print("Relancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
```

Changements clés par rapport à l'original :
- `wiki_dir.glob("src-*.md")` → `(wiki_dir / "sources").glob("src-*.md")` (bug de chemin corrigé).
- `_DEFAULT_RAW` pointe maintenant vers `~/Documents/Arbath/Wiki_LM/raw` (chemin réel).
- Le filtre `if rf.suffix.lower() == ".url"` est retiré — `_extract_url_from_file` est tenté sur tout fichier raw associé.
- Logique extraite dans `patch_pages(wiki_dir, raw_dir, apply)`, testable indépendamment du CLI.
- `--dry-run` supprimé au profit de `--apply` explicite (dry-run devient le défaut implicite), cohérent avec les autres scripts de ce plan.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_patch_lien_source.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Dry-run sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python patch_lien_source.py`
Expected: `Seraient patchées : 3` (ou proche — le nombre exact peut légèrement varier si des pages ont changé depuis l'analyse du 2026-08-19)

- [ ] **Step 6: Appliquer sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python patch_lien_source.py --apply`
Expected: `Patchées : 3`, puis vérifier manuellement une des pages patchées dans Obsidian (le champ `lien_source:` doit apparaître correctement dans le panneau des propriétés).

- [ ] **Step 7: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/patch_lien_source.py Wiki_LM/tests/test_patch_lien_source.py
git commit -m "fix(wiki): corrige patch_lien_source.py (chemins stales post-migration)

Cherchait dans wiki/*.md au lieu de wiki/sources/*.md, et son défaut
raw/ pointait vers un dossier inexistant. Filtre d'extraction assoupli
à tout type de fichier raw, pas seulement .url."
```

---

## Task 2: `list_unsourced_pages.py`

**Files:**
- Create: `Wiki_LM/tools/list_unsourced_pages.py`
- Test: `Wiki_LM/tests/test_list_unsourced_pages.py`

**Interfaces:**
- Consumes: rien de nouveau — relit directement le frontmatter des pages `wiki/sources/*.md`.
- Produces: `list_unsourced_pages.collect_unsourced(wiki_dir: Path) -> list[dict]` — chaque dict a les clés `slug`, `title`, `tags` (list[str]), `excerpt` (str, deux premières phrases du résumé). Utilisé par `main()` pour écrire le fichier Markdown.

- [ ] **Step 1: Write the failing tests**

```python
# Wiki_LM/tests/test_list_unsourced_pages.py
from __future__ import annotations

from pathlib import Path

from list_unsourced_pages import collect_unsourced, format_report


def _write_page(wiki_dir: Path, slug: str, lien_source: str | None, resume: str) -> Path:
    front = (
        "---\n"
        f"title: {slug.replace('src-', '').replace('-', ' ').title()}\n"
        "category: source\n"
        "tags: [test, exemple]\n"
        "created: 2026-01-01\n"
        "sources: []\n"
    )
    if lien_source:
        front += f"lien_source: {lien_source}\n"
    front += "---\n\n"
    body = f"# Titre\n\n## Résumé\n\n{resume}\n\n## Points clés\n\n- a\n"
    page = wiki_dir / "sources" / f"{slug}.md"
    page.write_text(front + body, encoding="utf-8")
    return page


def test_excludes_pages_with_lien_source(wiki_dir: Path):
    _write_page(wiki_dir, "src-a", "https://example.com", "Phrase un. Phrase deux. Phrase trois.")
    _write_page(wiki_dir, "src-b", None, "Phrase un. Phrase deux. Phrase trois.")

    result = collect_unsourced(wiki_dir)

    assert [r["slug"] for r in result] == ["src-b"]


def test_extracts_title_tags_and_two_sentence_excerpt(wiki_dir: Path):
    _write_page(wiki_dir, "src-b", None, "Phrase un. Phrase deux. Phrase trois.")

    result = collect_unsourced(wiki_dir)

    assert result[0]["title"] == "B"
    assert result[0]["tags"] == ["test", "exemple"]
    assert result[0]["excerpt"] == "Phrase un. Phrase deux."


def test_format_report_produces_markdown_table(wiki_dir: Path):
    _write_page(wiki_dir, "src-b", None, "Phrase un. Phrase deux.")
    entries = collect_unsourced(wiki_dir)

    report = format_report(entries)

    assert "| Slug | Titre | Tags | Extrait |" in report
    assert "src-b" in report
    assert "test, exemple" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_list_unsourced_pages.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'list_unsourced_pages'`

- [ ] **Step 3: Write `list_unsourced_pages.py`**

```python
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
        result.append({
            "slug": page.stem,
            "title": str(post.get("title", page.stem)),
            "tags": list(post.get("tags", [])),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_list_unsourced_pages.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lancer sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python list_unsourced_pages.py`
Expected: `~50 page(s) sans source listée(s) dans .../urls_a_rechercher.md` (le nombre exact dépend du résultat de la Task 1 — devrait être 87 − 34 − 3 = 50)

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/list_unsourced_pages.py Wiki_LM/tests/test_list_unsourced_pages.py
git commit -m "feat(wiki): liste les pages sans lien_source pour recherche manuelle"
```

---

## Task 3: Dé-ingestion complète (`index.md`/`tags.md`/`log.md` + retrait manuel)

**Files:**
- Modify: `Wiki_LM/tools/ingest.py`
- Create: `Wiki_LM/tools/sync_deletions_full.py`
- Test: `Wiki_LM/tests/test_sync_deletions_full.py`

**Interfaces:**
- Consumes: `Ingestor._sync_deletions(dry_run: bool) -> list[str]` (existant, inchangé en comportement).
- Produces: `Ingestor._cascade_deleted_slugs(deleted_slugs: set[str], dry_run: bool) -> list[str]` (nouveau, extrait de `_sync_deletions`, ne touche pas index/tags/log) ; `Ingestor._remove_from_index(slug: str) -> None` (nouveau) ; `Ingestor._finalize_deletions(affected: list[str], dry_run: bool) -> None` (nouveau, seul point qui écrit index.md/tags.md/log.md pour une dé-ingestion) ; `Ingestor._trash_page_full(slug: str, dry_run: bool) -> list[str]` (nouveau, retrait manuel d'un slug précis, retourne les slugs affectés).

- [ ] **Step 1: Write the failing tests**

```python
# Wiki_LM/tests/test_sync_deletions_full.py
from __future__ import annotations

from pathlib import Path

import frontmatter


def _write_page(wiki_dir: Path, subdir: str, slug: str, sources: list[str] | None = None) -> Path:
    front = (
        "---\n"
        f"title: {slug}\n"
        f"category: {'source' if subdir == 'sources' else 'concept'}\n"
        "tags: [test]\n"
        "created: 2026-01-01\n"
    )
    if sources is not None:
        front += f"sources: {sources}\n"
    front += "---\n\n# Test\n"
    page = wiki_dir / subdir / f"{slug}.md"
    page.write_text(front, encoding="utf-8")
    return page


class TestCascadeDeletedSlugs:
    def test_returns_affected_concept_now_orphaned(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        _write_page(wiki_dir, "concepts", "c-related", sources=["src-orphan"])

        affected = ingestor._cascade_deleted_slugs({"src-orphan"}, dry_run=False)

        assert "c-related" in affected
        assert (wiki_dir / "poubelle" / "c-related.md").exists()

    def test_does_not_touch_index_tags_or_log(self, ingestor, wiki_dir: Path):
        """_cascade_deleted_slugs ne fait que la cascade c-/e- — pas de finalisation
        (c'est _finalize_deletions qui s'en charge, appelé séparément)."""
        _write_page(wiki_dir, "sources", "src-orphan")
        _write_page(wiki_dir, "concepts", "c-related", sources=["src-orphan"])
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[c-related]] | concept | Related\n", encoding="utf-8"
        )

        ingestor._cascade_deleted_slugs({"src-orphan"}, dry_run=False)

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "c-related" in index  # pas encore retiré, _finalize_deletions n'a pas été appelé
        log = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "dé-ingestion" not in log


class TestFinalizeDeletions:
    def test_removes_from_index(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[src-orphan]] | source | Orphan\n- [[src-keep]] | source | Keep\n",
            encoding="utf-8",
        )
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=False)

        ingestor._finalize_deletions(["src-orphan"], dry_run=False)

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "src-orphan" not in index
        assert "src-keep" in index

    def test_rebuilds_tags_md_excluding_trashed(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=False)

        ingestor._finalize_deletions(["src-orphan"], dry_run=False)

        tags = (wiki_dir / "tags.md").read_text(encoding="utf-8")
        assert "src-orphan" not in tags

    def test_appends_to_log(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=False)

        ingestor._finalize_deletions(["src-orphan"], dry_run=False)

        log = (wiki_dir / "log.md").read_text(encoding="utf-8")
        assert "dé-ingestion" in log
        assert "src-orphan" in log

    def test_dry_run_does_not_write(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-orphan")
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[src-orphan]] | source | Orphan\n", encoding="utf-8"
        )
        ingestor._trash_page(wiki_dir / "sources" / "src-orphan.md", wiki_dir / "poubelle", dry_run=True)

        ingestor._finalize_deletions(["src-orphan"], dry_run=True)

        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "src-orphan" in index  # inchangé


class TestManualRemove:
    def test_trash_page_full_removes_slug_everywhere(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-manual")
        (wiki_dir / "index.md").write_text(
            "# Index\n\n- [[src-manual]] | source | Manual\n", encoding="utf-8"
        )

        affected = ingestor._trash_page_full("src-manual", dry_run=False)

        assert "src-manual" in affected
        assert (wiki_dir / "poubelle" / "src-manual.md").exists()
        assert not (wiki_dir / "sources" / "src-manual.md").exists()
        index = (wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "src-manual" not in index

    def test_trash_page_full_cascades_to_referencing_concepts(self, ingestor, wiki_dir: Path):
        _write_page(wiki_dir, "sources", "src-manual")
        _write_page(wiki_dir, "concepts", "c-related", sources=["src-manual"])

        ingestor._trash_page_full("src-manual", dry_run=False)

        assert (wiki_dir / "poubelle" / "c-related.md").exists()

    def test_trash_page_full_missing_slug_raises(self, ingestor, wiki_dir: Path):
        import pytest
        with pytest.raises(FileNotFoundError):
            ingestor._trash_page_full("src-does-not-exist", dry_run=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_sync_deletions_full.py -v`
Expected: FAIL — `AttributeError: 'Ingestor' object has no attribute '_cascade_deleted_slugs'`

- [ ] **Step 3: Refactor `_sync_deletions` et ajouter les nouvelles méthodes dans `ingest.py`**

Localiser `_sync_deletions` (actuellement lignes 826-899 environ). Le remplacer par les cinq méthodes suivantes, dans cet ordre (`_sync_deletions` et `_trash_page_full` partagent `_cascade_deleted_slugs` et `_finalize_deletions` — pas de duplication de la logique index/tags/log) :

```python
    def _sync_deletions(self, dry_run: bool = False) -> list[str]:
        """Détecte les fichiers raw supprimés, met à jour les pages c-/e- dépendantes.

        Pour chaque fichier raw disparu :
        - La page src- correspondante → poubelle/
        - Les pages c-/e- qui la référencent :
            - Si sources: devient vide → poubelle/
            - Sinon → retire le slug de sources: + status: à-réviser
        - index.md, tags.md et log.md sont mis à jour en conséquence.

        Retourne la liste des slugs mis en poubelle ou marqués à-réviser.
        """
        manifest = self._load_manifest()
        actual_files = {f.name for f in self.raw_dir.iterdir() if f.is_file()}
        poubelle_dir = self.wiki_dir / "poubelle"
        affected = []

        deleted_slugs: set[str] = set()
        for filename, meta in manifest.items():
            if filename in actual_files:
                continue
            slug = meta.get("slug", "")
            if not slug:
                continue
            src_page = slug_to_path(self.wiki_dir, slug)
            if not src_page.exists():
                deleted_slugs.add(slug)
                continue
            try:
                post = frontmatter.loads(src_page.read_text(encoding="utf-8"))
                if post.get("status") == "immuable":
                    continue
            except Exception:
                pass
            deleted_slugs.add(slug)
            self._trash_page(src_page, poubelle_dir, dry_run)
            affected.append(slug)
            print(f"[ingest] {'[dry]' if dry_run else '[trash]'} {slug} → poubelle/")

        if not deleted_slugs:
            return affected

        affected.extend(self._cascade_deleted_slugs(deleted_slugs, dry_run))
        self._finalize_deletions(affected, dry_run)
        return affected

    def _cascade_deleted_slugs(self, deleted_slugs: set[str], dry_run: bool = False) -> list[str]:
        """Répercute la suppression de deleted_slugs sur les pages c-/e- qui les référencent.

        Retourne la liste des slugs de pages c-/e- affectées (poubelle ou à-réviser).
        N'écrit pas index.md/tags.md/log.md — voir _finalize_deletions.
        """
        poubelle_dir = self.wiki_dir / "poubelle"
        affected = []

        for page in iter_pages(self.wiki_dir, subdirs=["concepts", "entités"]):
            try:
                content = page.read_text(encoding="utf-8")
                post = frontmatter.loads(content)
            except Exception:
                continue
            if post.get("status") == "immuable":
                continue

            sources = post.get("sources", []) or []
            if not isinstance(sources, list):
                sources = [sources]
            remaining = [s for s in sources if s not in deleted_slugs]
            if len(remaining) == len(sources):
                continue

            if not remaining:
                self._trash_page(page, poubelle_dir, dry_run)
                affected.append(page.stem)
                print(f"[ingest] {'[dry]' if dry_run else '[trash]'} {page.stem} → poubelle/ (plus de sources)")
            else:
                if not dry_run:
                    post["sources"] = remaining
                    post["status"] = "à-réviser"
                    page.write_text(frontmatter.dumps(post), encoding="utf-8")
                affected.append(page.stem)
                removed = set(sources) - set(remaining)
                print(f"[ingest] {'[dry]' if dry_run else '[réviser]'} {page.stem} : "
                      f"retrait {removed}, status → à-réviser")

        return affected

    def _remove_from_index(self, slug: str) -> None:
        """Retire la ligne du slug dans index.md, si présente."""
        index_path = self.wiki_dir / "index.md"
        if not index_path.exists():
            return
        text = index_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"^- \[\[{re.escape(slug)}\]\].*\n?", re.MULTILINE)
        new_text = pattern.sub("", text)
        if new_text != text:
            index_path.write_text(new_text, encoding="utf-8")

    def _finalize_deletions(self, affected: list[str], dry_run: bool = False) -> None:
        """Met à jour index.md, tags.md et log.md pour les slugs affectés.

        Partagé par _sync_deletions et _trash_page_full — seul point qui écrit
        ces trois fichiers pour une opération de dé-ingestion.
        """
        if dry_run or not affected:
            return
        for slug in affected:
            self._remove_from_index(slug)
        self._rebuild_tags_index()
        for slug in affected:
            self._append_log("dé-ingestion", slug)

    def _trash_page_full(self, slug: str, dry_run: bool = False) -> list[str]:
        """Retrait manuel et délibéré d'une page précise (sans exiger que son
        fichier raw/ ait disparu). Déclenche la même cascade que _sync_deletions.

        Lève FileNotFoundError si le slug n'existe pas.
        """
        page = slug_to_path(self.wiki_dir, slug)
        if not page.exists():
            raise FileNotFoundError(f"Page introuvable pour le slug {slug!r}")

        poubelle_dir = self.wiki_dir / "poubelle"
        self._trash_page(page, poubelle_dir, dry_run)
        affected = [slug] + self._cascade_deleted_slugs({slug}, dry_run)
        self._finalize_deletions(affected, dry_run)
        return affected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_sync_deletions_full.py tests/test_ingest.py -v`
Expected: PASS — tous les tests, y compris les tests existants de `test_ingest.py` (non-régression sur `_sync_deletions`).

- [ ] **Step 5: Write `sync_deletions_full.py`**

```python
"""
Dé-ingestion complète : synchronise les suppressions de raw/ (comme
Ingestor._sync_deletions) et permet aussi un retrait manuel d'une page
précise via --remove.

Usage :
    python sync_deletions_full.py [--wiki PATH] [--raw PATH] [--apply]
    python sync_deletions_full.py --remove src-un-slug [--apply]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ingest import Ingestor

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dé-ingestion complète (index/tags/log)")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    parser.add_argument("--remove", metavar="SLUG", help="Retrait manuel d'un slug précis")
    args = parser.parse_args()

    ingestor = Ingestor(args.wiki)

    if args.remove:
        affected = ingestor._trash_page_full(args.remove, dry_run=not args.apply)
    else:
        affected = ingestor._sync_deletions(dry_run=not args.apply)

    print(f"\n{'Affectées' if args.apply else 'Seraient affectées'} : {len(affected)}")
    for slug in affected:
        print(f"  - {slug}")
    if not args.apply:
        print("\nRelancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they still pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/ -v`
Expected: PASS — suite complète, aucune régression.

- [ ] **Step 7: Dry-run sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python sync_deletions_full.py`
Expected: affiche les pages qui seraient affectées (probablement 0 ou peu, la plupart des `raw/` manquants datent d'avant le manifeste — vérifier le nombre affiché avant de conclure).

- [ ] **Step 8: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/ingest.py Wiki_LM/tools/sync_deletions_full.py Wiki_LM/tests/test_sync_deletions_full.py
git commit -m "feat(wiki): dé-ingestion complète (index/tags/log) + retrait manuel

_sync_deletions ne mettait à jour ni index.md, ni tags.md, ni log.md.
Cascade extraite dans _cascade_deleted_slugs (réutilisable), nouvelle
_trash_page_full pour un retrait délibéré via --remove <slug>."
```

---

## Task 4: `retry_blocked_urls.py`

**Files:**
- Create: `Wiki_LM/tools/retry_blocked_urls.py`
- Test: `Wiki_LM/tests/test_retry_blocked_urls.py`

**Interfaces:**
- Consumes: `Ingestor.ingest(source: str, ...) -> str` (existant), `Ingestor._parse_url_file` / `_extract_url_from_file` (existants).
- Produces: `retry_blocked_urls.retry_all(ingestor: Ingestor, raw_dir: Path, dry_run: bool) -> list[dict]` — chaque dict a `filename`, `url`, `status` (`"ingested"` ou `"failed"`), et `error` si échec.

- [ ] **Step 1: Write the failing tests**

```python
# Wiki_LM/tests/test_retry_blocked_urls.py
from __future__ import annotations

from pathlib import Path

from retry_blocked_urls import retry_all


def test_retries_error_files_and_marks_success(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\n", encoding="utf-8"
    )

    def fake_ingest(self, source, **kwargs):
        return "src-article"

    monkeypatch.setattr(type(ingestor), "ingest", fake_ingest)

    results = retry_all(ingestor, raw_dir, dry_run=False)

    assert results[0]["status"] == "ingested"
    assert not (raw_dir / "20260101-000000-example-com.url.error").exists()


def test_retries_error_files_and_keeps_failure_with_reason(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\n", encoding="utf-8"
    )

    def failing_ingest(self, source, **kwargs):
        raise ConnectionError("DNS lookup failed")

    monkeypatch.setattr(type(ingestor), "ingest", failing_ingest)

    results = retry_all(ingestor, raw_dir, dry_run=False)

    assert results[0]["status"] == "failed"
    assert "DNS lookup failed" in results[0]["error"]
    content = (raw_dir / "20260101-000000-example-com.url.error").read_text(encoding="utf-8")
    assert "DNS lookup failed" in content


def test_dry_run_does_not_call_ingest(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\n", encoding="utf-8"
    )
    called = []

    def tracking_ingest(self, source, **kwargs):
        called.append(source)
        return "src-article"

    monkeypatch.setattr(type(ingestor), "ingest", tracking_ingest)

    retry_all(ingestor, raw_dir, dry_run=True)

    assert called == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_retry_blocked_urls.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retry_blocked_urls'`

- [ ] **Step 3: Write `retry_blocked_urls.py`**

```python
"""
Relance l'ingestion des fichiers .url.error via le pipeline actif (pas
mcp_server.py, qui est mort — abandon MCP, commit 3693a57).

En cas de nouvel échec, la raison est ajoutée en commentaire dans le
fichier .url.error (jusqu'ici silencieux — seule l'URL y était visible).

Usage :
    python retry_blocked_urls.py [--wiki PATH] [--raw PATH] [--apply]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ingest import Ingestor, _extract_url_from_file

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"
_DEFAULT_RAW = Path.home() / "Documents" / "Arbath" / "Wiki_LM" / "raw"


def retry_all(ingestor: Ingestor, raw_dir: Path, dry_run: bool) -> list[dict]:
    results: list[dict] = []

    for error_file in sorted(raw_dir.glob("*.url.error")):
        url = _extract_url_from_file(error_file)
        entry = {"filename": error_file.name, "url": url}

        if dry_run:
            entry["status"] = "would-retry"
            results.append(entry)
            continue

        try:
            slug = ingestor.ingest(url)
            entry["status"] = "ingested"
            entry["slug"] = slug
            error_file.unlink()
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            content = error_file.read_text(encoding="utf-8")
            if "# échec :" not in content:
                error_file.write_text(content.rstrip("\n") + f"\n# échec : {exc}\n", encoding="utf-8")

        results.append(entry)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Relance les fichiers .url.error")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument("--raw", default=os.environ.get("WIKI_RAW_PATH", str(_DEFAULT_RAW)))
    parser.add_argument("--apply", action="store_true", help="Relancer réellement (sinon dry-run)")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    ingestor = Ingestor(args.wiki, raw_path=raw_dir)

    results = retry_all(ingestor, raw_dir, dry_run=not args.apply)

    for r in results:
        print(f"{r['filename']} → {r['status']}" + (f" ({r.get('error')})" if r.get("error") else ""))
    if not args.apply:
        print("\nRelancez avec --apply pour exécuter réellement.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_retry_blocked_urls.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Dry-run sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python retry_blocked_urls.py`
Expected: liste les 4 fichiers `.url.error` avec statut `would-retry` et leurs URLs.

- [ ] **Step 6: Appliquer sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python retry_blocked_urls.py --apply`
Expected: pour chacune des 4 URLs, soit `ingested` (le fichier `.url.error` disparaît, une nouvelle page `src-*.md` apparaît dans `wiki/sources/`), soit `failed` (le fichier reste, avec la raison ajoutée en commentaire). Vérifier le résultat avec `ls ~/Documents/Arbath/Wiki_LM/raw/*.url.error` (devrait afficher moins de 4 fichiers, idéalement 0).

- [ ] **Step 7: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/retry_blocked_urls.py Wiki_LM/tests/test_retry_blocked_urls.py
git commit -m "feat(wiki): relance les fichiers .url.error via le pipeline actif

Remplace mcp_server.py (mort depuis l'abandon MCP) comme mécanisme de
retry. Les échecs récurrents gardent désormais leur raison en commentaire
au lieu de rester silencieux."
```

---

## Task 5: Supprimer `mcp_server.py`

**Files:**
- Delete: `Wiki_LM/tools/mcp_server.py`

**Interfaces:**
- Consumes: rien.
- Produces: rien — suppression pure.

**Préalable obligatoire :** cette tâche ne s'exécute qu'après validation du résultat de la Task 4 (les 4 URLs relancées avec succès ou échec correctement journalisé). Ne pas supprimer si la Task 4 a laissé des fichiers `.url.error` inexpliqués sans commentaire de raison.

- [ ] **Step 1: Vérifier qu'aucun code actif ne référence encore mcp_server.py**

Run: `cd ~/Secretarius && grep -rln "mcp_server" --include="*.py" --include="*.sh" --include="*.json" --include="*.service" . | grep -v "Wiki_LM/tools/mcp_server.py"`
Expected: aucune sortie (déjà vérifié en amont pendant le brainstorming, à reconfirmer avant suppression).

- [ ] **Step 2: Supprimer le fichier**

```bash
cd ~/Secretarius
git rm Wiki_LM/tools/mcp_server.py
```

- [ ] **Step 3: Run full test suite**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/ -v`
Expected: PASS — aucun test ne dépendait de `mcp_server.py` (à vérifier : `grep -rl "mcp_server" tests/` doit être vide avant de commiter).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(wiki): supprime mcp_server.py (code mort, abandon MCP)

Plus référencé nulle part depuis l'abandon de l'approche MCP (commit
3693a57). Ses fichiers .url.error orphelins ont été traités par
retry_blocked_urls.py (tâche précédente)."
```

---

## Task 6: Corriger la cause racine des tags corrompus

**Files:**
- Modify: `Wiki_LM/tools/ingest.py`
- Modify: `Wiki_LM/tests/test_ingest.py`

**Interfaces:**
- Consumes: rien de nouveau.
- Produces: `Ingestor._parse_raw_tags(path: Path) -> list[str]` (signature inchangée, comportement corrigé).

**Diagnostic confirmé :** `_parse_raw_tags` (ligne ~1028) lit une ligne `tags: ...` d'un fichier raw et découpe sur la virgule sans retirer les crochets. Si la ligne est `tags: [documentation, secretarius]` (style liste), le résultat est `['[documentation', 'secretarius]']` au lieu de `['documentation', 'secretarius']`. Confirmé en rejouant la fonction sur les fichiers raw réels associés aux pages corrompues.

- [ ] **Step 1: Write the failing test**

Ajouter à `Wiki_LM/tests/test_ingest.py` :

```python
class TestParseRawTags:
    def test_plain_comma_separated(self, tmp_path: Path):
        from ingest import Ingestor
        f = tmp_path / "raw.url"
        f.write_text("https://example.com\ntags: documentation, secretarius\n", encoding="utf-8")
        assert Ingestor._parse_raw_tags(f) == ["documentation", "secretarius"]

    def test_bracket_wrapped_list_style(self, tmp_path: Path):
        """Régression : tags: [a, b] ne doit plus produire ['[a', 'b]']."""
        from ingest import Ingestor
        f = tmp_path / "raw.url"
        f.write_text("https://example.com\ntags: [documentation, secretarius]\n", encoding="utf-8")
        assert Ingestor._parse_raw_tags(f) == ["documentation", "secretarius"]

    def test_single_bracket_wrapped_tag(self, tmp_path: Path):
        """Régression : tags: [christianisme] (un seul tag, entre crochets)."""
        from ingest import Ingestor
        f = tmp_path / "raw.url"
        f.write_text("https://example.com\ntags: [christianisme]\n", encoding="utf-8")
        assert Ingestor._parse_raw_tags(f) == ["christianisme"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_ingest.py::TestParseRawTags -v`
Expected: FAIL — `test_bracket_wrapped_list_style` et `test_single_bracket_wrapped_tag` échouent (`assert ['[documentation', 'secretarius]'] == ['documentation', 'secretarius']` et équivalent).

- [ ] **Step 3: Corriger `_parse_raw_tags`**

Dans `Wiki_LM/tools/ingest.py`, remplacer :

```python
    @staticmethod
    def _parse_raw_tags(path: Path) -> list[str]:
        """Lit la ligne `tags: tag1, tag2` d'un fichier raw si présente."""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("tags:"):
                raw = line[5:].strip()
                return [t.strip() for t in raw.split(",") if t.strip()]
        return []
```

par :

```python
    @staticmethod
    def _parse_raw_tags(path: Path) -> list[str]:
        """Lit la ligne `tags: tag1, tag2` d'un fichier raw si présente.

        Tolère aussi le style liste `tags: [tag1, tag2]`.
        """
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("tags:"):
                raw = line[5:].strip()
                raw = raw.strip("[]")
                return [t.strip() for t in raw.split(",") if t.strip()]
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_ingest.py -v`
Expected: PASS — suite `test_ingest.py` complète, y compris les 3 nouveaux tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/ingest.py Wiki_LM/tests/test_ingest.py
git commit -m "fix(wiki): _parse_raw_tags tolère le style liste tags: [a, b]

Sans strip des crochets, tags: [documentation, secretarius] devenait
['[documentation', 'secretarius]'] — 36 pages affectées, réparées dans
une tâche séparée (fix_corrupted_tags.py)."
```

---

## Task 7: `fix_corrupted_tags.py` — réparer les 36 pages existantes

**Files:**
- Create: `Wiki_LM/tools/fix_corrupted_tags.py`
- Test: `Wiki_LM/tests/test_fix_corrupted_tags.py`

**Interfaces:**
- Consumes: rien de nouveau.
- Produces: `fix_corrupted_tags.clean_tags(tags: list[str]) -> list[str]` — retire les crochets superflus de chaque tag, déduplique en préservant l'ordre. `fix_corrupted_tags.fix_pages(wiki_dir: Path, apply: bool) -> list[str]` — retourne les slugs des pages modifiées.

- [ ] **Step 1: Write the failing tests**

```python
# Wiki_LM/tests/test_fix_corrupted_tags.py
from __future__ import annotations

from pathlib import Path

import frontmatter

from fix_corrupted_tags import clean_tags, fix_pages


class TestCleanTags:
    def test_removes_duplicate_bracket_wrapped_tag(self):
        """Motif A : tag entre crochets dupliquant un tag déjà présent."""
        result = clean_tags(["christianisme", "art", "[christianisme]"])
        assert result == ["christianisme", "art"]

    def test_rejoins_split_fragments(self):
        """Motif B : liste coupée en fragments par la virgule à l'intérieur des crochets."""
        result = clean_tags(["[documentation", "secretarius]"])
        assert result == ["documentation", "secretarius"]

    def test_leaves_clean_tags_unchanged(self):
        result = clean_tags(["documentation", "secretarius", "openclaw"])
        assert result == ["documentation", "secretarius", "openclaw"]


class TestFixPages:
    def test_fixes_corrupted_page_on_disk(self, wiki_dir: Path):
        page = wiki_dir / "sources" / "src-corrupted.md"
        page.write_text(
            "---\ntitle: Corrupted\ncategory: source\n"
            "tags:\n- '[documentation'\n- secretarius]\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Corrupted\n",
            encoding="utf-8",
        )

        fixed = fix_pages(wiki_dir, apply=True)

        assert "src-corrupted" in fixed
        post = frontmatter.load(page)
        assert post["tags"] == ["documentation", "secretarius"]

    def test_dry_run_does_not_write(self, wiki_dir: Path):
        page = wiki_dir / "sources" / "src-corrupted.md"
        page.write_text(
            "---\ntitle: Corrupted\ncategory: source\n"
            "tags:\n- '[documentation'\n- secretarius]\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Corrupted\n",
            encoding="utf-8",
        )

        fix_pages(wiki_dir, apply=False)

        post = frontmatter.load(page)
        assert post["tags"] == ["[documentation", "secretarius]"]

    def test_skips_pages_without_bracket_pattern(self, wiki_dir: Path):
        page = wiki_dir / "sources" / "src-clean.md"
        page.write_text(
            "---\ntitle: Clean\ncategory: source\ntags: [a, b]\n"
            "created: 2026-01-01\nsources: []\n---\n\n# Clean\n",
            encoding="utf-8",
        )

        fixed = fix_pages(wiki_dir, apply=True)

        assert "src-clean" not in fixed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_fix_corrupted_tags.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fix_corrupted_tags'`

- [ ] **Step 3: Write `fix_corrupted_tags.py`**

```python
"""
Répare les pages src- dont les tags ont été corrompus par l'ancien bug de
_parse_raw_tags (tags: [a, b] mal découpé — voir ingest.py, corrigé
séparément pour les nouvelles ingestions).

Deux motifs réparés :
  - Motif A : un tag entre crochets dupliquant un tag déjà présent
    (ex. ["christianisme", "[christianisme]"] → ["christianisme"])
  - Motif B : liste coupée en fragments par la virgule à l'intérieur des
    crochets (ex. ["[documentation", "secretarius]"] → ["documentation", "secretarius"])

Usage :
    python fix_corrupted_tags.py [--wiki PATH] [--apply]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import frontmatter

_DEFAULT_WIKI = Path.home() / "Documents" / "Arbath" / "Wiki_LM"


def _looks_corrupted(tags: list[str]) -> bool:
    return any(t.startswith("[") or t.endswith("]") for t in tags)


def clean_tags(tags: list[str]) -> list[str]:
    """Retire les crochets superflus de chaque tag, déduplique en préservant l'ordre."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = tag.strip().lstrip("[").rstrip("]").strip()
        if clean and clean not in seen:
            cleaned.append(clean)
            seen.add(clean)
    return cleaned


def fix_pages(wiki_dir: Path, apply: bool) -> list[str]:
    """Répare les pages src- aux tags corrompus. Retourne les slugs modifiés."""
    fixed: list[str] = []
    for page in sorted((wiki_dir / "sources").glob("src-*.md")):
        post = frontmatter.load(page)
        tags = list(post.get("tags", []))
        if not _looks_corrupted(tags):
            continue

        new_tags = clean_tags(tags)
        print(f"{'[dry]' if not apply else '[fix]'} {page.stem} : {tags} → {new_tags}")

        if apply:
            post["tags"] = new_tags
            page.write_text(frontmatter.dumps(post), encoding="utf-8")

        fixed.append(page.stem)

    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Répare les tags corrompus (tags: [a, b] mal découpé)")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", str(_DEFAULT_WIKI)))
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    fixed = fix_pages(wiki_dir, apply=args.apply)

    print(f"\n{'Réparées' if args.apply else 'Seraient réparées'} : {len(fixed)}")
    if not args.apply:
        print("Relancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/test_fix_corrupted_tags.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Dry-run sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python fix_corrupted_tags.py`
Expected: `Seraient réparées : 36` (ou proche — peut varier légèrement si des pages ont changé depuis l'analyse).

- [ ] **Step 6: Appliquer sur les données réelles**

Run: `cd ~/Secretarius/Wiki_LM/tools && python fix_corrupted_tags.py --apply`
Expected: `Réparées : 36`. Vérifier manuellement 2-3 pages dans Obsidian (tags affichés proprement, sans crochets littéraux).

- [ ] **Step 7: Run full test suite (non-régression finale)**

Run: `cd ~/Secretarius/Wiki_LM && python -m pytest tests/ -v`
Expected: PASS — suite complète, tous composants du plan inclus.

- [ ] **Step 8: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/fix_corrupted_tags.py Wiki_LM/tests/test_fix_corrupted_tags.py
git commit -m "fix(wiki): répare les 36 pages aux tags corrompus par l'ancien bug

Deux motifs : tag entre crochets dupliqué (dédupliqué), et liste coupée
en fragments par la virgule (rejointe). Cause racine corrigée séparément
dans _parse_raw_tags."
```

---

## Résumé de vérification finale

Après les 7 tâches :
- `python -m pytest Wiki_LM/tests/ -v` passe intégralement.
- `~/Documents/Arbath/Wiki_LM/wiki/sources/*.md` : 37 pages avec `lien_source:` (34 + 3), 50 listées dans `urls_a_rechercher.md`.
- `~/Documents/Arbath/Wiki_LM/raw/*.url.error` : idéalement 0 fichier restant (ou fichiers restants avec raison d'échec en commentaire).
- Aucune page avec un tag commençant par `[` ou finissant par `]`.
- `Wiki_LM/tools/mcp_server.py` supprimé, aucune référence résiduelle.
- Échantillon de pages modifiées vérifié visuellement dans Obsidian.
