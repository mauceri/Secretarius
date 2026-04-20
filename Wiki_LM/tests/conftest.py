"""Fixtures partagées pour les tests Wiki_LM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Rendre tools/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

_SOURCE_PAGE = """\
---
title: Test Source
category: source
tags: [test, wiki]
created: 2026-01-01
sources: []
status: nouveau
---

# Test Source

## Résumé

Contenu de test pour la suite de tests automatisés.

## Points clés

- Point 1
- Point 2

## Concepts et entités mentionnés

- concept: zettelkasten
- entité: Vannevar Bush

## Liens internes suggérés

Aucun
"""

_CONCEPT_PAGE = """\
---
title: {name}
category: concept
tags: [test]
created: 2026-01-01
sources: [src-test-source]
status: nouveau
---

# {name}

Définition de test pour {name}.

## Liens

Aucun
"""

_ENTITY_PAGE = """\
---
title: {name}
category: entité
tags: [test]
created: 2026-01-01
sources: [src-test-source]
status: nouveau
---

# {name}

Page d'entité de test pour {name}.

## Liens

Aucun
"""


class MockLLM:
    """LLM factice retournant des pages valides sans appel réseau."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
        self.calls.append(prompt[:80])
        if "concept" in prompt.lower() and "enrichis" in prompt.lower():
            name = _extract_name(prompt, "concept")
            return _CONCEPT_PAGE.format(name=name)
        if "entité" in prompt.lower() and "enrichis" in prompt.lower():
            name = _extract_name(prompt, "entité")
            return _ENTITY_PAGE.format(name=name)
        return _SOURCE_PAGE


def _extract_name(prompt: str, kind: str) -> str:
    import re
    m = re.search(rf'Tu enrichis la page wiki du {kind} "([^"]+)"', prompt)
    return m.group(1) if m else kind.capitalize()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def wiki_dir(tmp_path: Path) -> Path:
    """Répertoire wiki vide avec index.md et log.md."""
    w = tmp_path / "wiki"
    w.mkdir()
    (w / "index.md").write_text("# Index\n\n", encoding="utf-8")
    (w / "log.md").write_text("", encoding="utf-8")
    return w


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Répertoire raw/ vide."""
    r = tmp_path / "raw"
    r.mkdir()
    return r


@pytest.fixture
def wiki_root(tmp_path: Path, wiki_dir: Path, raw_dir: Path) -> Path:
    """Racine complète : wiki/ + raw/ dans tmp_path."""
    return tmp_path


@pytest.fixture
def ingestor(wiki_root: Path, raw_dir: Path, mock_llm: MockLLM):
    """Ingestor configuré avec MockLLM et raw_dir temporaire."""
    from ingest import Ingestor
    from wiki_lookup import WikiLookup
    wl = WikiLookup(wiki_root, zim_dir=wiki_root / "zim")  # ZIM dir vide → pas de ZIM réel
    ing = Ingestor(wiki_root, llm=mock_llm, raw_path=raw_dir)
    ing._wiki_lookup = wl
    return ing
