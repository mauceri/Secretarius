"""Emplacement des plongements de pages : un seul chemin, résolu depuis WIKI_PATH.

Régression du 2026-09-09 : search.py résolvait `$WIKI_PATH/embeddings` tandis
que embed.py, cluster.py, similarity.py, kb_update.py et dedup.py codaient en
dur `<dépôt>/embeddings`. Les plongements étaient donc écrits à un endroit et
cherchés à un autre — la recherche sémantique retombait silencieusement en
BM25 seul, y compris dans le sandbox où le répertoire du dépôt n'est même pas
monté.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import wiki_paths


def _reload(monkeypatch, wiki_path: Path):
    monkeypatch.setenv("WIKI_PATH", str(wiki_path))
    importlib.reload(wiki_paths)
    return wiki_paths


def test_embeddings_dir_suit_wiki_path(monkeypatch, tmp_path):
    wp = _reload(monkeypatch, tmp_path)
    assert wp.embeddings_dir() == tmp_path / "embeddings"


def test_embeddings_dir_hors_du_depot(monkeypatch, tmp_path):
    # Le dépôt ne contient que du code : les données dérivées vivent dans le
    # coffre, seul répertoire monté dans le sandbox de l'agent wiki.
    wp = _reload(monkeypatch, tmp_path)
    depot = Path(wp.__file__).resolve().parent.parent
    assert depot not in wp.embeddings_dir().parents


def test_tous_les_outils_pointent_au_meme_endroit(monkeypatch, tmp_path):
    """Anti-dérive : c'est la divergence entre ces modules qui avait cassé la
    recherche sémantique sans qu'aucun test ne le voie."""
    wp = _reload(monkeypatch, tmp_path)
    attendu = wp.embeddings_dir()

    for module_name, attribut in [
        ("embed", "EMBED_DIR"),
        ("dedup", "EMBED_DIR"),
        ("search", "_EMBED_DIR"),
        ("cluster", "_EMBED_DIR"),
        ("similarity", "_EMBED_DIR"),
        ("kb_update", "_DEFAULT_EMBED_DIR"),
    ]:
        module = importlib.reload(importlib.import_module(module_name))
        assert getattr(module, attribut) == attendu, (
            f"{module_name}.{attribut} diverge de embeddings_dir()"
        )
