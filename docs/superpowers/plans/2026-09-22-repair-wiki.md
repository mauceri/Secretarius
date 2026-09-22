# Réparation du wiki à partir des rapports de lint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réparer deux familles d'erreurs réelles trouvées par `lint.py` (liens cassés, frontmatter manquant) via un nouveau module `repair.py`, exposé en lot Obsidian via `/lint`, `/repair?`, `/repair!`.

**Architecture:** `WikiRepair` (nouveau, sur le modèle de `WikiLint`) répare une famille à la fois, essai à blanc par défaut. Les liens cassés se réparent par retrait mécanique des crochets ; le frontmatter manquant se répare par déplacement mécanique (si un second bloc bien formé existe) ou régénération LLM du titre (sinon), `category` étant toujours dérivé du sous-répertoire, jamais deviné.

**Tech Stack:** Python 3.12 (PyYAML déjà disponible via `python-frontmatter`), Flask, pytest ; TypeScript (plugin Obsidian, vitest).

**Spec:** `docs/superpowers/specs/2026-09-22-repair-wiki-design.md`

## Global Constraints

- Essai à blanc par défaut partout ; écriture réelle seulement sur demande
  explicite (`--apply` en CLI, `!` en lot Obsidian).
- Une famille à la fois — jamais les deux dans le même appel.
- `category` est toujours dérivé du sous-répertoire du fichier
  (`sources/` → `source`, `concepts/` → `concept`, `entités/` → `entité`)
  — jamais deviné par le LLM.
- Exposition Obsidian uniquement (lot `` ```wiki ``) — pas de changement à
  `derisk-deleg` ni aux gabarits `openclaw.json`.
- Vocabulaire des familles identique partout : `broken-link` et
  `missing-frontmatter` (mêmes chaînes que les codes de `lint.py`), en CLI
  comme en argument de commande de lot.

---

### Task 1: `lint.py` — champ structuré pour la cible d'un lien cassé

**Files:**
- Modify: `Wiki_LM/tools/lint.py`
- Test: `Wiki_LM/tests/test_lint.py`

**Interfaces:**
- Produces: `LintIssue.target: str = ""` (peuplé uniquement pour `code == "broken-link"`), `LintReport.add(level, code, slug, message, target="")`.

- [ ] **Step 1: Écrire le test, en échec**

Ajouter à `Wiki_LM/tests/test_lint.py`, dans la classe `TestCheckLinks` :

```python
    def test_broken_link_target_is_structured(self, wiki_root, wiki_dir):
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]].",
        )

        report = WikiLint(wiki_root).run()

        broken = [i for i in report.errors if i.code == "broken-link"]
        assert len(broken) == 1
        assert broken[0].target == "c-inexistant"

    def test_non_broken_link_issue_has_empty_target(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", body="Sans frontmatter valide.")

        report = WikiLint(wiki_root).run()

        assert all(i.target == "" for i in report.issues if i.code != "broken-link")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_lint.py -k target -v`
Expected: FAIL — `AttributeError: 'LintIssue' object has no attribute 'target'`

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/lint.py`, la classe `LintIssue` (autour de la ligne 43) — remplacer :
```python
@dataclass
class LintIssue:
    level: str          # "error" | "warning" | "info"
    code: str           # identifiant court
    slug: str           # page concernée (ou "" si global)
    message: str
```
par :
```python
@dataclass
class LintIssue:
    level: str          # "error" | "warning" | "info"
    code: str           # identifiant court
    slug: str           # page concernée (ou "" si global)
    message: str
    target: str = ""    # slug cible pour code == "broken-link", vide sinon
                         # (structuré séparément du message texte pour que
                         # repair.py n'ait pas à le reparser — 2026-09-22)
```

`LintReport.add()` (autour de la ligne 62) — remplacer :
```python
    def add(self, level: str, code: str, slug: str, message: str) -> None:
        self.issues.append(LintIssue(level=level, code=code, slug=slug, message=message))
```
par :
```python
    def add(self, level: str, code: str, slug: str, message: str, target: str = "") -> None:
        self.issues.append(LintIssue(level=level, code=code, slug=slug, message=message, target=target))
```

`_check_links()` (autour de la ligne 161) — remplacer :
```python
    def _check_links(self, pages: dict, report: LintReport) -> None:
        """Détecte les liens [[slug]] cassés."""
        all_slugs = set(pages.keys()) | _META_PAGES
        for slug, info in pages.items():
            for target in info["links"]:
                if target not in all_slugs:
                    report.add(
                        "error", "broken-link", slug,
                        f"Lien cassé : [[{target}]]",
                    )
```
par :
```python
    def _check_links(self, pages: dict, report: LintReport) -> None:
        """Détecte les liens [[slug]] cassés."""
        all_slugs = set(pages.keys()) | _META_PAGES
        for slug, info in pages.items():
            for target in info["links"]:
                if target not in all_slugs:
                    report.add(
                        "error", "broken-link", slug,
                        f"Lien cassé : [[{target}]]",
                        target=target,
                    )
```

Ne rien changer d'autre : `to_dict()` et `LintIssue.__str__()` ne mentionnent pas `target`, donc leur sortie ne change pas — les tests existants du 21/09 doivent continuer de passer sans modification.

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_lint.py -v`
Expected: tous PASS (16 existants + 2 nouveaux)

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/lint.py Wiki_LM/tests/test_lint.py
git commit -m "feat(wiki): LintIssue.target structuré pour les liens cassés"
```

---

### Task 2: `repair.py` — module de base + réparation des liens cassés

**Files:**
- Create: `Wiki_LM/tools/repair.py`
- Test: `Wiki_LM/tests/test_repair.py`

**Interfaces:**
- Consumes: `lint.WikiLint`, `LintIssue.target` (Task 1) ; `wiki_paths.slug_to_path`.
- Produces: `RepairReport` (dataclass : `family: str`, `dry_run: bool`, `changes: list[str]`, `before_count: int`, `after_count: int`) ; `WikiRepair(wiki_path).repair_broken_links(dry_run: bool = True) -> RepairReport`.

- [ ] **Step 1: Écrire les tests, en échec**

Créer `Wiki_LM/tests/test_repair.py` :

```python
"""Tests de repair.py."""

from __future__ import annotations

from pathlib import Path

from repair import WikiRepair


def _write_page(wiki_dir: Path, subdir: str, slug: str, *, title: str = "",
                 category: str = "", body: str = "") -> Path:
    d = wiki_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{slug}.md"
    fm_lines = ["---"]
    if title:
        fm_lines.append(f"title: {title}")
    if category:
        fm_lines.append(f"category: {category}")
    fm_lines.append("---")
    path.write_text("\n".join(fm_lines) + f"\n\n{body}\n", encoding="utf-8")
    return path


class TestRepairBrokenLinks:
    def test_dry_run_does_not_modify_disk(self, wiki_root, wiki_dir):
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]] et [[c-autre-inexistant]].",
        )
        original = path.read_text(encoding="utf-8")

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=True)

        assert path.read_text(encoding="utf-8") == original
        assert report.dry_run is True
        assert report.family == "broken-link"

    def test_apply_strips_brackets_from_broken_links(self, wiki_root, wiki_dir):
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-inexistant]] et [[c-autre-inexistant]].",
        )

        WikiRepair(wiki_root).repair_broken_links(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "[[c-inexistant]]" not in content
        assert "[[c-autre-inexistant]]" not in content
        assert "c-inexistant" in content
        assert "c-autre-inexistant" in content

    def test_apply_leaves_valid_links_untouched(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "concepts", "c-b", title="B", category="concept")
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-b]] et [[c-inexistant]].",
        )

        WikiRepair(wiki_root).repair_broken_links(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "[[c-b]]" in content
        assert "[[c-inexistant]]" not in content

    def test_before_after_counts(self, wiki_root, wiki_dir):
        _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-x]] et [[c-y]].",
        )

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=False)

        assert report.before_count == 2
        assert report.after_count == 0

    def test_dry_run_reports_accurate_after_count_without_writing(self, wiki_root, wiki_dir):
        path = _write_page(
            wiki_dir, "sources", "src-a", title="A", category="source",
            body="Voir [[c-x]].",
        )
        original = path.read_text(encoding="utf-8")

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=True)

        assert report.before_count == 1
        assert report.after_count == 0
        assert path.read_text(encoding="utf-8") == original  # rien écrit malgré after_count correct

    def test_no_broken_links_reports_empty_changes(self, wiki_root, wiki_dir):
        _write_page(wiki_dir, "sources", "src-a", title="A", category="source", body="Rien à réparer.")

        report = WikiRepair(wiki_root).repair_broken_links(dry_run=True)

        assert report.changes == []
        assert report.before_count == 0
        assert report.after_count == 0
```

Réutiliser les fixtures `wiki_root`/`wiki_dir` déjà présentes dans
`Wiki_LM/tests/conftest.py` (mêmes fixtures qu'utilise `test_lint.py`).

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_repair.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'repair'`

- [ ] **Step 3: Implémenter**

Créer `Wiki_LM/tools/repair.py` :

```python
"""
Réparation du wiki Wiki_LM à partir des rapports de lint.py.

Deux familles, traitées indépendamment, jamais ensemble :
  - broken-link         : retire les crochets des liens cassés
                           ([[slug]] -> slug texte brut)
  - missing-frontmatter : répare le frontmatter vide/tronqué (déplacement
                           mécanique si un second bloc bien formé existe,
                           sinon régénération du titre par le LLM)

Usage CLI :
    python repair.py --broken-links              # essai à blanc
    python repair.py --broken-links --apply       # écrit réellement
    python repair.py --frontmatter [--apply]
    python repair.py --wiki /chemin --broken-links

Usage module :
    from repair import WikiRepair
    repairer = WikiRepair("/home/mauceric/Documents/Secretarius/Wiki_LM")
    report = repairer.repair_broken_links(dry_run=True)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from lint import WikiLint
from wiki_paths import slug_to_path


@dataclass
class RepairReport:
    family: str
    dry_run: bool
    changes: list[str] = field(default_factory=list)
    before_count: int = 0
    after_count: int = 0


class WikiRepair:
    def __init__(self, wiki_path: str | Path) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"

    def repair_broken_links(self, dry_run: bool = True) -> RepairReport:
        """Retire les crochets des liens cassés — [[slug]] devient slug en
        texte brut. Aucune hypothèse sur la cause du lien mort : ça
        s'applique à tout lien dont la cible n'existe pas, quelle qu'en
        soit l'origine."""
        report_before = WikiLint(self.wiki_root).run()
        before_count = sum(1 for i in report_before.issues if i.code == "broken-link")

        by_page: dict[str, list[str]] = {}
        for issue in report_before.issues:
            if issue.code == "broken-link":
                by_page.setdefault(issue.slug, []).append(issue.target)

        changes: list[str] = []
        fixed_count = 0
        for slug, targets in sorted(by_page.items()):
            path = slug_to_path(self.wiki_dir, slug)
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            updated = content
            for target in targets:
                updated = updated.replace(f"[[{target}]]", target)
            if updated != content:
                changes.append(f"{slug} : {len(targets)} lien(s) cassé(s) retiré(s)")
                fixed_count += len(targets)
                if not dry_run:
                    path.write_text(updated, encoding="utf-8")

        return RepairReport(
            family="broken-link",
            dry_run=dry_run,
            changes=changes,
            before_count=before_count,
            after_count=before_count - fixed_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Réparation du wiki Wiki_LM")
    import os
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Secretarius/Wiki_LM")),
        help="Chemin vers Wiki_LM",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--broken-links", action="store_true", help="Réparer les liens cassés")
    group.add_argument("--frontmatter", action="store_true", help="Réparer le frontmatter manquant")
    parser.add_argument("--apply", action="store_true", help="Écrire réellement (défaut : essai à blanc)")
    args = parser.parse_args()

    repairer = WikiRepair(args.wiki)
    if args.broken_links:
        report = repairer.repair_broken_links(dry_run=not args.apply)
    else:
        report = repairer.repair_frontmatter(dry_run=not args.apply)

    mode = "Essai à blanc" if report.dry_run else "Appliqué"
    print(f"{mode} — {report.family} : {report.before_count} → {report.after_count}")
    for change in report.changes:
        print(f"  {change}")


if __name__ == "__main__":
    main()
```

Note : `main()` appelle `repairer.repair_frontmatter(...)`, qui n'existe pas
encore — c'est attendu, cette méthode arrive à la Task 3. La CLI n'est pas
testée dans cette tâche (`--frontmatter` non exerçable avant la Task 3) ;
seule l'API Python (`repair_broken_links`) est testée ici.

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_repair.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/repair.py Wiki_LM/tests/test_repair.py
git commit -m "feat(wiki): repair.py — réparation des liens cassés (retrait des crochets)"
```

---

### Task 3: `repair.py` — réparation du frontmatter manquant

**Files:**
- Modify: `Wiki_LM/tools/repair.py`
- Test: `Wiki_LM/tests/test_repair.py`

**Interfaces:**
- Consumes: `wiki_paths.subdir_for_slug` ; `llm.LLM`.
- Produces: `WikiRepair(wiki_path, llm=None).repair_frontmatter(dry_run: bool = True) -> RepairReport` ; helpers de module `_extract_closed_frontmatter_block(body) -> tuple[dict, str] | None`, `_clean_body_for_regeneration(body) -> str`, `_slug_to_title(slug) -> str`.

- [ ] **Step 1: Écrire les tests, en échec**

Ajouter à `Wiki_LM/tests/test_repair.py` :

```python
class TestExtractClosedFrontmatterBlock:
    def test_finds_well_formed_second_block(self):
        from repair import _extract_closed_frontmatter_block
        body = (
            "\nyaml\n---\ntitle: Mon titre\ncategory: concept\n---\n```\n\n"
            "# Mon titre\n\nLe corps réel."
        )
        result = _extract_closed_frontmatter_block(body)
        assert result is not None
        meta, rest = result
        assert meta["title"] == "Mon titre"
        assert meta["category"] == "concept"
        assert "Le corps réel." in rest
        assert "yaml" not in rest.split("\n")[0] if rest else True

    def test_returns_none_without_a_closed_block(self):
        from repair import _extract_closed_frontmatter_block
        body = "## Extrait Wikipedia\n\nDu texte normal, aucun bloc frontmatter."
        assert _extract_closed_frontmatter_block(body) is None

    def test_returns_none_for_an_unclosed_block(self):
        from repair import _extract_closed_frontmatter_block
        body = "---\ntitle: Titre tronqué\ncategory: concept\nsources: [src-a"
        assert _extract_closed_frontmatter_block(body) is None

    def test_returns_none_when_block_lacks_title_or_category(self):
        from repair import _extract_closed_frontmatter_block
        body = "---\ntags: [a, b]\n---\n\nCorps."
        assert _extract_closed_frontmatter_block(body) is None


class TestSlugToTitle:
    def test_strips_prefix_and_humanizes(self):
        from repair import _slug_to_title
        assert _slug_to_title("c-mon-concept") == "mon concept"
        assert _slug_to_title("e-vannevar-bush") == "vannevar bush"
        assert _slug_to_title("src-un-article") == "un article"


class TestRepairFrontmatter:
    def test_well_formed_block_is_moved_mechanically_no_llm_call(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-x.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n{}\n---\n\nyaml\n---\ntitle: Mon concept\ncategory: concept\n"
            "---\n```\n\n# Mon concept\n\nLe corps réel.\n",
            encoding="utf-8",
        )

        class _NoCallLLM:
            def complete(self, *a, **k):
                raise AssertionError("le LLM ne doit pas être appelé pour un bloc bien formé")

        report = WikiRepair(wiki_root, llm=_NoCallLLM()).repair_frontmatter(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "title: Mon concept" in content
        assert content.startswith("---\n")
        assert "yaml" not in content.split("---")[0]
        assert "Le corps réel." in content
        assert report.family == "missing-frontmatter"

    def test_missing_block_regenerates_title_via_llm(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-y.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n{}\n---\n\n## Extrait Wikipedia\n\nDu contenu réel sur le sujet Y.\n",
            encoding="utf-8",
        )

        class _StubLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, system="", max_tokens=2048):
                self.calls.append(prompt)
                return "Titre régénéré"

        stub = _StubLLM()
        WikiRepair(wiki_root, llm=stub).repair_frontmatter(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "title: Titre régénéré" in content
        assert "category: concept" in content  # dérivé du sous-répertoire, pas du LLM
        assert "Du contenu réel sur le sujet Y." in content
        assert len(stub.calls) == 1
        assert "Du contenu réel sur le sujet Y." in stub.calls[0]

    def test_category_derived_from_subdir_for_entity(self, wiki_root, wiki_dir):
        path = wiki_dir / "entités" / "e-z.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\n{}\n---\n\nDu contenu sur Z.\n", encoding="utf-8")

        class _StubLLM:
            def complete(self, prompt, system="", max_tokens=2048):
                return "Z"

        WikiRepair(wiki_root, llm=_StubLLM()).repair_frontmatter(dry_run=False)

        assert "category: entité" in path.read_text(encoding="utf-8")

    def test_empty_body_falls_back_to_slug_title_without_llm_call(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-vide.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\n{}\n---\n", encoding="utf-8")

        class _NoCallLLM:
            def complete(self, *a, **k):
                raise AssertionError("le LLM ne doit pas être appelé sans contenu exploitable")

        WikiRepair(wiki_root, llm=_NoCallLLM()).repair_frontmatter(dry_run=False)

        content = path.read_text(encoding="utf-8")
        assert "title: vide" in content

    def test_dry_run_does_not_modify_disk_or_call_llm(self, wiki_root, wiki_dir):
        path = wiki_dir / "concepts" / "c-y.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = "---\n{}\n---\n\nDu contenu.\n"
        path.write_text(original, encoding="utf-8")

        class _NoCallLLM:
            def complete(self, *a, **k):
                raise AssertionError("essai à blanc : le LLM ne doit pas être appelé")

        report = WikiRepair(wiki_root, llm=_NoCallLLM()).repair_frontmatter(dry_run=True)

        assert path.read_text(encoding="utf-8") == original
        assert report.dry_run is True
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_repair.py -v`
Expected: FAIL — `ImportError: cannot import name '_extract_closed_frontmatter_block'` (et `WikiRepair(..., llm=...)` rejette `llm` : paramètre inexistant)

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/repair.py` :

1. Étendre les imports en tête de fichier — remplacer :
```python
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from lint import WikiLint
from wiki_paths import slug_to_path
```
par :
```python
from __future__ import annotations

import argparse
import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lint import WikiLint
from llm import LLM
from wiki_paths import slug_to_path, subdir_for_slug
```

2. Après la définition de `RepairReport`, ajouter :
```python
_FENCE_LINE_RE = re.compile(r"^(?:```(?:yaml|markdown)?|yaml)\s*$", re.MULTILINE)
_CLOSED_BLOCK_RE = re.compile(r"^---\n(.*?)\n---\s*\n?", re.DOTALL | re.MULTILINE)

_SUBDIR_TO_CATEGORY = {"sources": "source", "concepts": "concept", "entités": "entité"}

_PROMPT_REGENERATE_TITLE = """\
Voici le contenu d'une page de wiki dont le titre a été perdu (bug \
d'écriture antérieur). Réponds uniquement par un titre court (une seule \
ligne, sans guillemets, sans ponctuation finale) résumant le sujet de \
cette page — n'invente rien qui ne soit pas dans le contenu.

Contenu :
---
{content}
---

Titre :"""


def _extract_closed_frontmatter_block(body: str) -> tuple[dict, str] | None:
    """Cherche un second bloc frontmatter bien formé et fermé dans le corps
    d'une page (motif du 21-22/09/2026 : ---\\n{}\\n---\\n vide en tête,
    parfois suivi de débris de balises de code, puis un second bloc
    --- ... --- qui, lui, contient les vraies métadonnées). Retourne
    (métadonnées, reste du corps après le bloc) si ce bloc contient au
    moins title et category ; None sinon — y compris si le bloc n'est
    jamais refermé (génération interrompue)."""
    m = _CLOSED_BLOCK_RE.search(body)
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1))
    except Exception:
        return None
    if not isinstance(meta, dict) or not meta.get("title") or not meta.get("category"):
        return None
    rest = body[m.end():].strip()
    return meta, rest


def _clean_body_for_regeneration(body: str) -> str:
    """Retire les débris de balises de code (```yaml, ```markdown, ```, ou
    un « yaml » seul sur sa ligne) qui traînent dans un corps de page dont
    le frontmatter n'a pas pu être promu — sans quoi ces lignes polluent
    l'entrée envoyée au LLM. Une éventuelle tentative de frontmatter
    tronquée (ex. `---\\ntitle: ...\\nsources: [src-a`, jamais refermée)
    reste dans le résultat : elle contient souvent le titre en clair, une
    bien meilleure base pour l'extraction qu'un corps vide."""
    return _FENCE_LINE_RE.sub("", body).strip()


def _slug_to_title(slug: str) -> str:
    """Repli déterministe si le corps ne contient rien d'exploitable :
    dérive un titre lisible du slug lui-même (c-mon-concept -> "mon
    concept"). N'appelle jamais le LLM."""
    base = re.sub(r"^(?:src|c|e)-", "", slug)
    return base.replace("-", " ").strip() or slug
```

3. Modifier `WikiRepair.__init__` — remplacer :
```python
class WikiRepair:
    def __init__(self, wiki_path: str | Path) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"
```
par :
```python
class WikiRepair:
    def __init__(self, wiki_path: str | Path, llm: LLM | None = None) -> None:
        self.wiki_root = Path(wiki_path)
        self.wiki_dir = self.wiki_root / "wiki"
        self.llm = llm or LLM()
```

4. Après `repair_broken_links()` (avant `def main()`), ajouter :
```python
    def repair_frontmatter(self, dry_run: bool = True) -> RepairReport:
        """Répare le frontmatter vide/tronqué. Deux traitements selon la
        forme : un second bloc bien formé mais mal placé se déplace
        mécaniquement (aucun appel LLM) ; sinon, le titre est régénéré par
        le LLM à partir du corps restant — category est toujours dérivé du
        sous-répertoire, jamais deviné."""
        report_before = WikiLint(self.wiki_root).run()
        before_count = sum(1 for i in report_before.issues if i.code == "missing-frontmatter")
        slugs = sorted({i.slug for i in report_before.issues if i.code == "missing-frontmatter"})

        changes: list[str] = []
        for slug in slugs:
            subdir = subdir_for_slug(slug)
            path = self.wiki_dir / subdir / f"{slug}.md"
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8")
            import frontmatter as fm_module
            post = fm_module.loads(raw)
            body = post.content

            found = _extract_closed_frontmatter_block(body)
            if found:
                meta, rest = found
                new_content = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + f"---\n\n{rest}\n"
                changes.append(f"{slug} : bloc frontmatter bien formé déplacé")
            else:
                cleaned = _clean_body_for_regeneration(body)
                category = _SUBDIR_TO_CATEGORY.get(subdir, "source")
                if cleaned:
                    title = self.llm.complete(
                        _PROMPT_REGENERATE_TITLE.format(content=cleaned[:4000]),
                        max_tokens=100,
                    ).strip().strip('"').strip("'")
                    if not title:
                        title = _slug_to_title(slug)
                else:
                    title = _slug_to_title(slug)
                meta = {
                    "title": title,
                    "category": category,
                    "tags": [],
                    "created": datetime.date.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "sources": [],
                }
                new_content = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + f"---\n\n{cleaned}\n"
                changes.append(f"{slug} : frontmatter régénéré (titre : {title!r})")

            if not dry_run:
                path.write_text(new_content, encoding="utf-8")

        return RepairReport(
            family="missing-frontmatter",
            dry_run=dry_run,
            changes=changes,
            before_count=before_count,
            after_count=before_count - 2 * len(changes),
        )
```

Note sur `before_count - 2 * len(changes)` : chaque page de cette famille
manque aujourd'hui exactement deux champs (`title` et `category` — le
motif observé sur les 65 pages réelles du 22/09/2026 est systématiquement
`---\n{}\n---`, les deux vides ensemble). Si une future page ne manquait
QUE d'un champ, ce calcul sous-compterait légèrement l'écart en essai à
blanc — cas non rencontré à ce jour, non traité spécialement (YAGNI).

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_repair.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/repair.py Wiki_LM/tests/test_repair.py
git commit -m "feat(wiki): repair.py — réparation du frontmatter (déplacement ou régénération LLM)"
```

---

### Task 4: `wiki.py` — `op_lint`, `op_repair_preview`, `op_repair`

**Files:**
- Modify: `Wiki_LM/tools/wiki.py`
- Test: `Wiki_LM/tests/test_wiki_cli.py`

**Interfaces:**
- Consumes: `lint.WikiLint` (Task 1) ; `repair.WikiRepair` (Tasks 2-3).
- Produces: `op_lint() -> dict`, `op_repair_preview(family: str) -> dict`, `op_repair(family: str) -> dict`.

- [ ] **Step 1: Écrire les tests, en échec**

Ajouter à `Wiki_LM/tests/test_wiki_cli.py` (utiliser le helper `_wiki(monkeypatch, tmp_path)` déjà présent en tête de fichier, comme pour les autres tests de ce module) :

```python
class TestOpLint:
    def test_returns_counts_by_code(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)

        class _FakeIssue:
            def __init__(self, code):
                self.code = code

        class _FakeReport:
            checked_pages = 3
            issues = [_FakeIssue("broken-link"), _FakeIssue("broken-link"), _FakeIssue("missing-frontmatter")]

            @property
            def errors(self):
                return self.issues

            @property
            def warnings(self):
                return []

        class _FakeLint:
            def __init__(self, wiki_path):
                pass

            def run(self):
                return _FakeReport()

        monkeypatch.setattr(wiki, "WikiLint", _FakeLint, raising=False)
        out = wiki.op_lint()

        assert out["checked_pages"] == 3
        assert out["errors"] == 3
        assert out["by_code"] == {"broken-link": 2, "missing-frontmatter": 1}


class TestOpRepair:
    def test_preview_calls_dry_run(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)
        calls = []

        class _FakeReport:
            family = "broken-link"
            dry_run = True
            before_count = 5
            after_count = 0
            changes = ["src-a : 5 lien(s) cassé(s) retiré(s)"]

        class _FakeRepair:
            def __init__(self, wiki_path):
                pass

            def repair_broken_links(self, dry_run):
                calls.append(dry_run)
                return _FakeReport()

        monkeypatch.setattr(wiki, "WikiRepair", _FakeRepair, raising=False)
        out = wiki.op_repair_preview("broken-link")

        assert calls == [True]
        assert out["family"] == "broken-link"
        assert out["before_count"] == 5

    def test_apply_calls_real_run(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)
        calls = []

        class _FakeReport:
            family = "missing-frontmatter"
            dry_run = False
            before_count = 2
            after_count = 0
            changes = []

        class _FakeRepair:
            def __init__(self, wiki_path):
                pass

            def repair_frontmatter(self, dry_run):
                calls.append(dry_run)
                return _FakeReport()

        monkeypatch.setattr(wiki, "WikiRepair", _FakeRepair, raising=False)
        wiki.op_repair("missing-frontmatter")

        assert calls == [False]

    def test_unknown_family_returns_error(self, monkeypatch, tmp_path):
        wiki = _wiki(monkeypatch, tmp_path)
        out = wiki.op_repair_preview("famille-inconnue")
        assert "error" in out
```

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_wiki_cli.py -k "OpLint or OpRepair" -v`
Expected: FAIL — `AttributeError: module 'wiki' has no attribute 'op_lint'`

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/wiki.py` :

1. Étendre les imports — remplacer :
```python
from wiki_paths import slug_to_path
```
par :
```python
from lint import WikiLint
from repair import WikiRepair
from wiki_paths import slug_to_path
```

2. Ajouter, après `op_tags()` (autour de la ligne 293) :
```python
def op_lint() -> dict:
    report = WikiLint(_wiki_root()).run()
    from collections import Counter
    by_code = Counter(i.code for i in report.issues)
    return {
        "checked_pages": report.checked_pages,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "by_code": dict(by_code),
    }


_REPAIR_FAMILIES = {"broken-link", "missing-frontmatter"}


def _repair(family: str, dry_run: bool) -> dict:
    if family not in _REPAIR_FAMILIES:
        return {"error": f"Famille de réparation inconnue : {family!r}"}
    repairer = WikiRepair(_wiki_root())
    if family == "broken-link":
        report = repairer.repair_broken_links(dry_run=dry_run)
    else:
        report = repairer.repair_frontmatter(dry_run=dry_run)
    return {
        "family": report.family,
        "dry_run": report.dry_run,
        "before_count": report.before_count,
        "after_count": report.after_count,
        "changes": report.changes,
    }


def op_repair_preview(family: str) -> dict:
    return _repair(family, dry_run=True)


def op_repair(family: str) -> dict:
    return _repair(family, dry_run=False)
```

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_wiki_cli.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/wiki.py Wiki_LM/tests/test_wiki_cli.py
git commit -m "feat(wiki): op_lint/op_repair_preview/op_repair"
```

---

### Task 5: `server.py` — `/lint`, `/repair?`, `/repair!`

**Files:**
- Modify: `Wiki_LM/tools/server.py`
- Test: `Wiki_LM/tests/test_server.py`

**Interfaces:**
- Consumes: `op_lint`, `op_repair_preview`, `op_repair` (Task 4).

- [ ] **Step 1: Écrire les tests, en échec**

Dans `Wiki_LM/tests/test_server.py`, classe `TestHandleRun` — remplacer :
```python
    _COMMAND_TABLE = [
        ("/ingest", "op_ingest", ""),
        ("/wikistatus", "op_status", ""),
        ("/r", "op_search", "une recherche"),
        ("/tags", "op_tags", ""),
        ("/kbupdate", "op_kb_update", ""),
        ("/relire", "op_review", ""),
        ("/verifie", "op_verify", "src-test"),
        ("/supprimer?", "op_delete_preview", "src-test"),
        ("/supprimer!", "op_delete", "src-test"),
    ]
```
par :
```python
    _COMMAND_TABLE = [
        ("/ingest", "op_ingest", ""),
        ("/wikistatus", "op_status", ""),
        ("/r", "op_search", "une recherche"),
        ("/tags", "op_tags", ""),
        ("/kbupdate", "op_kb_update", ""),
        ("/relire", "op_review", ""),
        ("/verifie", "op_verify", "src-test"),
        ("/supprimer?", "op_delete_preview", "src-test"),
        ("/supprimer!", "op_delete", "src-test"),
        ("/lint", "op_lint", ""),
        ("/repair?", "op_repair_preview", "broken-link"),
        ("/repair!", "op_repair", "broken-link"),
    ]
```
`/repair?`/`/repair!` prennent un argument (`family`) mais, comme les
autres entrées de cette table (hormis `/c`/`/q`, absentes d'ici car elles
utilisent `vault_name` et ont leurs propres tests dédiés plus bas dans le
fichier), une seule chaîne d'argument suffit — même gabarit que
`/supprimer?`/`/supprimer!`.

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -k dispatches_each -v`
Expected: FAIL — `AssertionError` ou `KeyError` : `/lint`, `/repair?`, `/repair!` absents de `_RUN_OPS`

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/tools/server.py` :

1. Étendre l'import de `wiki` — remplacer :
```python
from wiki import (
    op_capture,
    op_delete,
    op_delete_preview,
    op_ingest,
    op_kb_update,
    op_query,
    op_review,
    op_search,
    op_status,
    op_tags,
    op_verify,
)
```
par :
```python
from wiki import (
    op_capture,
    op_delete,
    op_delete_preview,
    op_ingest,
    op_kb_update,
    op_lint,
    op_query,
    op_repair,
    op_repair_preview,
    op_review,
    op_search,
    op_status,
    op_tags,
    op_verify,
)
```

2. `_RUN_OPS` — remplacer :
```python
_RUN_OPS = {
    # Chaque entrée reçoit (arg, vault_name) de façon uniforme — un seul
    # protocole d'appel dans la table de dispatch. Seules /c et /q
    # utilisent vault_name ; les autres l'ignorent (sous-wikis par coffre,
    # 2026-09-22).
    "/c": lambda arg, vault: op_capture(arg, vault),
    "/q": lambda arg, vault: op_query(arg, vault),
    "/ingest": lambda arg, vault: op_ingest(),
    "/wikistatus": lambda arg, vault: op_status(),
    "/r": lambda arg, vault: op_search(arg),
    "/tags": lambda arg, vault: op_tags(),
    "/kbupdate": lambda arg, vault: op_kb_update(),
    "/relire": lambda arg, vault: op_review(),
    "/verifie": lambda arg, vault: op_verify(arg),
    # /supprimer (sans !) reste absente : jamais dispatchée, même demandée
    # explicitement — seule Telegram, avec essai à blanc puis /confirm,
    # peut supprimer sans le ! explicite ci-dessous.
    "/supprimer?": lambda arg, vault: op_delete_preview(arg),
    "/supprimer!": lambda arg, vault: op_delete(arg),
}
```
par :
```python
_RUN_OPS = {
    # Chaque entrée reçoit (arg, vault_name) de façon uniforme — un seul
    # protocole d'appel dans la table de dispatch. Seules /c et /q
    # utilisent vault_name ; les autres l'ignorent (sous-wikis par coffre,
    # 2026-09-22).
    "/c": lambda arg, vault: op_capture(arg, vault),
    "/q": lambda arg, vault: op_query(arg, vault),
    "/ingest": lambda arg, vault: op_ingest(),
    "/wikistatus": lambda arg, vault: op_status(),
    "/r": lambda arg, vault: op_search(arg),
    "/tags": lambda arg, vault: op_tags(),
    "/kbupdate": lambda arg, vault: op_kb_update(),
    "/relire": lambda arg, vault: op_review(),
    "/verifie": lambda arg, vault: op_verify(arg),
    # /supprimer (sans !) reste absente : jamais dispatchée, même demandée
    # explicitement — seule Telegram, avec essai à blanc puis /confirm,
    # peut supprimer sans le ! explicite ci-dessous.
    "/supprimer?": lambda arg, vault: op_delete_preview(arg),
    "/supprimer!": lambda arg, vault: op_delete(arg),
    "/lint": lambda arg, vault: op_lint(),
    # /repair! répare toute une famille en un seul appel (jusqu'à des
    # centaines de pages) — rayon d'action plus large qu'un /supprimer!,
    # qui ne touche qu'une page. Le ! tient lieu de la même confirmation
    # explicite (réparation du wiki, 2026-09-22).
    "/repair?": lambda arg, vault: op_repair_preview(arg),
    "/repair!": lambda arg, vault: op_repair(arg),
}
```

Aussi mettre à jour le docstring en tête de fichier (`POST /run`) pour
mentionner `/lint`, `/repair?`, `/repair!`, sur le modèle de l'entrée
`/supprimer?`/`/supprimer!` déjà présente.

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM && .venv/bin/python -m pytest tests/test_server.py -v`
Expected: tous PASS

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/tools/server.py Wiki_LM/tests/test_server.py
git commit -m "feat(wiki): /lint, /repair?, /repair! dans le dispatch de /run"
```

---

### Task 6: Plugin Obsidian — `/lint`, `/repair?`, `/repair!`

**Files:**
- Modify: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts`
- Test: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts`

**Interfaces:**
- Consumes: réponses JSON de `/run` pour `/lint` (`checked_pages`, `errors`, `warnings`, `by_code`) et `/repair?`/`/repair!` (`family`, `before_count`, `after_count`, `changes`) — Task 5.

- [ ] **Step 1: Écrire les tests, en échec**

Dans `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts`, ajouter
aux tests de `formatWikiResult` :

```typescript
  it("formats /lint as a per-code summary", () => {
    const text = formatWikiResult(
      "/lint",
      { checked_pages: 863, errors: 890, warnings: 2, by_code: { "broken-link": 875, "unknown-category": 2 } },
      true
    );
    expect(text).toContain("863");
    expect(text).toContain("890");
    expect(text).toContain("broken-link: 875");
  });

  it("formats /repair? as a dry-run report, nothing changed", () => {
    const text = formatWikiResult(
      "/repair?",
      { family: "broken-link", before_count: 890, after_count: 0, changes: ["src-a : 2 lien(s) cassé(s) retiré(s)"] },
      true
    );
    expect(text).toContain("Essai à blanc");
    expect(text).toContain("890");
    expect(text).toContain("0");
    expect(text).toContain("src-a");
    expect(text).toContain("/repair!");
  });

  it("formats /repair! as a completed repair report", () => {
    const text = formatWikiResult(
      "/repair!",
      { family: "missing-frontmatter", before_count: 130, after_count: 4, changes: ["c-x : frontmatter régénéré (titre : 'X')"] },
      true
    );
    expect(text).toContain("130");
    expect(text).toContain("4");
    expect(text).toContain("c-x");
  });
```

Dans le describe `SUPPORTED_COMMANDS` — remplacer :
```typescript
  it("contains exactly the eleven wiki commands from the spec", () => {
    expect([...SUPPORTED_COMMANDS].sort()).toEqual(
      [
        "/c",
        "/ingest",
        "/kbupdate",
        "/q",
        "/r",
        "/relire",
        "/supprimer!",
        "/supprimer?",
        "/tags",
        "/verifie",
        "/wikistatus",
      ].sort()
    );
  });
```
par :
```typescript
  it("contains exactly the fourteen wiki commands from the spec", () => {
    expect([...SUPPORTED_COMMANDS].sort()).toEqual(
      [
        "/c",
        "/ingest",
        "/kbupdate",
        "/lint",
        "/q",
        "/r",
        "/relire",
        "/repair!",
        "/repair?",
        "/supprimer!",
        "/supprimer?",
        "/tags",
        "/verifie",
        "/wikistatus",
      ].sort()
    );
  });
```
(remplace le test « contains exactly the eleven wiki commands » existant —
même nom de test à mettre à jour avec le nouveau compte et la nouvelle
liste).

- [ ] **Step 2: Vérifier l'échec**

Run: `cd Wiki_LM/obsidian-wikilm-capture && npm test -- --run`
Expected: FAIL — `/lint`/`/repair?`/`/repair!` non gérées par `formatWikiResult` (retombent sur le `default`), et le test de comptage de `SUPPORTED_COMMANDS` échoue (11 au lieu de 14)

- [ ] **Step 3: Implémenter**

Dans `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts` :

1. `SUPPORTED_COMMANDS` — remplacer :
```typescript
export const SUPPORTED_COMMANDS = [
  "/c",
  "/q",
  "/ingest",
  "/wikistatus",
  "/r",
  "/tags",
  "/kbupdate",
  "/relire",
  "/verifie",
  "/supprimer?",
  "/supprimer!",
] as const;
```
par :
```typescript
export const SUPPORTED_COMMANDS = [
  "/c",
  "/q",
  "/ingest",
  "/wikistatus",
  "/r",
  "/tags",
  "/kbupdate",
  "/relire",
  "/verifie",
  "/supprimer?",
  "/supprimer!",
  "/lint",
  "/repair?",
  "/repair!",
] as const;
```

2. `formatWikiResult()` — remplacer :
```typescript
    case "/supprimer!": {
      const affected = (data.affected as string[] | undefined) ?? [];
      return `Supprimé : ${affected.length} page(s) — ${affected.join(", ")}.`;
    }
    default:
```
par :
```typescript
    case "/supprimer!": {
      const affected = (data.affected as string[] | undefined) ?? [];
      return `Supprimé : ${affected.length} page(s) — ${affected.join(", ")}.`;
    }
    case "/lint": {
      const byCode = (data.by_code as Record<string, number> | undefined) ?? {};
      const details = Object.entries(byCode).map(([code, n]) => `${code}: ${n}`).join(", ");
      return `${data.checked_pages} page(s) vérifiée(s) — ${data.errors} erreur(s), ${data.warnings} avertissement(s).${details ? `\n${details}` : ""}`;
    }
    case "/repair?": {
      const changes = (data.changes as string[] | undefined) ?? [];
      const detail = changes.length > 0 ? `\n\n${changes.join("\n")}` : "";
      return `Essai à blanc (${data.family}) — ${data.before_count} → ${data.after_count} après réparation.${detail}\n\nRien n'a été modifié. Remplacez par /repair! pour confirmer.`;
    }
    case "/repair!": {
      const changes = (data.changes as string[] | undefined) ?? [];
      const detail = changes.length > 0 ? `\n\n${changes.join("\n")}` : "";
      return `Réparé (${data.family}) — ${data.before_count} → ${data.after_count}.${detail}`;
    }
    default:
```

- [ ] **Step 4: Vérifier le succès**

Run: `cd Wiki_LM/obsidian-wikilm-capture && npm test -- --run`
Expected: tous PASS. Puis `npm run build` pour reconstruire `main.js`.

- [ ] **Step 5: Commit**

```bash
git add Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts
git commit -m "feat(wiki): plugin — /lint, /repair?, /repair! dans le lot Obsidian"
```

(`main.js` reste hors dépôt — `.gitignore` — comme pour tous les
changements précédents du plugin.)

---

### Task 7: Vérification finale, documentation, déploiement

**Files:**
- Modify: `docs/architecture/wiki-lm-architecture.md`

- [ ] **Step 1: Suite de tests complète**

```bash
cd Wiki_LM && .venv/bin/python -m pytest tests/ -q
cd obsidian-wikilm-capture && npm test -- --run
```
Expected: tous PASS.

- [ ] **Step 2: Mettre à jour la doc d'architecture**

Dans `docs/architecture/wiki-lm-architecture.md`, section « 10. État
actuel et chantiers ouverts », remplacer la ligne :
```
- `lint.py` — glob non récursif, ne trouve quasiment rien (1458 « erreurs »
  rapportées à tort lors du dernier audit) ; jamais corrigé.
```
(déjà obsolète depuis le 21/09 — le correctif de `lint.py` n'y avait pas
été documenté) par :
```
- `lint.py` — réparé le 21/09/2026 (migré vers `iter_pages()`). Un module
  de réparation (`repair.py`) répare depuis le 22/09/2026 deux des
  familles qu'il rapporte : liens cassés (retrait des crochets au-delà de
  la limite de citations retenues) et frontmatter manquant (déplacement
  mécanique ou régénération LLM du titre selon la forme). Exposé via
  `/lint`, `/repair?`, `/repair!` en lot Obsidian. Restent non traités :
  les pages orphelines de l'index (« index-ghost », devraient se résorber
  à la reconstruction de l'index) et les catégories inconnues.
  Spec : `docs/superpowers/specs/2026-09-22-repair-wiki-design.md`.
```

- [ ] **Step 3: Redémarrer et vérifier en direct — demander confirmation avant**

Ceci touche `wiki.py`/`server.py`, servis par `wiki-lm-server` : demander
la confirmation de l'utilisateur avant `systemctl --user restart
wiki-lm-server`, puis déployer le plugin reconstruit dans les deux coffres
(`~/Documents/Secretarius/.obsidian/plugins/wikilm-capture/main.js` et
`~/Documents/Arbath/.obsidian/plugins/wikilm-capture/main.js`).

Test en direct suggéré, **d'abord en lecture seule puis en essai à blanc
avant toute application réelle** — étant donné le nombre de pages réelles
concernées (890 liens cassés, 130 champs manquants au 22/09/2026) :
```bash
curl -s -X POST http://127.0.0.1:5051/run -H "Content-Type: application/json" \
  -d '{"command":"/lint","arg":""}'
curl -s -X POST http://127.0.0.1:5051/run -H "Content-Type: application/json" \
  -d '{"command":"/repair?","arg":"broken-link"}'
```
Ne PAS enchaîner sur `/repair!` sans un nouvel accord explicite de
l'utilisateur, distinct de celui qui couvre le redémarrage/déploiement —
c'est une écriture en série sur des centaines de pages réelles et
synchronisées, la méthode héritée de la revue du 21/09 l'exige
explicitement (mesurer avant/après, une famille à la fois, accord avant
toute écriture en série).

- [ ] **Step 4: Commit de la documentation**

```bash
git add docs/architecture/wiki-lm-architecture.md
git commit -m "docs(wiki): module de réparation livré, mise à jour de l'état des chantiers"
```
