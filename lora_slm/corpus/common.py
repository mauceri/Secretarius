#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires partagés pour la génération de corpus synthétique.

Chargeurs de fichiers de configuration (thèmes, catégories, types, exemples)
et fonction de conversion vers le format « chunks » attendu par src/train.py.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Chargeurs de fichiers de configuration
# ---------------------------------------------------------------------------

def _extract_python_container(text: str, open_ch: str, close_ch: str) -> str:
    """Extrait le premier littéral conteneur Python équilibré (liste ou dict)."""
    start = text.find(open_ch)
    if start == -1:
        raise ValueError(f"'{open_ch}' introuvable dans le texte.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Conteneur non fermé (déséquilibré).")


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_themes(path: str) -> List[str]:
    """
    Charge une liste de thèmes depuis :
    - un fichier JSON strict  (``["thème1", "thème2", ...]``)
    - ou un fichier de type ``themes = [...]`` (littéral Python).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    # Essai JSON strict
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fallback littéral Python
    list_src = _extract_python_container(data, "[", "]")
    themes: List[str] = ast.literal_eval(list_src)
    if not isinstance(themes, list) or not all(isinstance(x, str) for x in themes):
        raise RuntimeError(f"Format de thèmes invalide dans {path}")
    return themes


def load_categories(path: str) -> List[str]:
    """
    Charge une liste de catégories depuis un fichier tolérant :
    JSON strict, ou liste « Python-like » avec virgules et retours à la ligne.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    body = raw.strip().lstrip("[").rstrip("]")
    items: List[str] = []
    for line in body.splitlines():
        token = line.strip().rstrip(",")
        if not token:
            continue
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            token = token[1:-1]
        token = re.sub(r"[^a-zA-Z0-9_\-àâäéèêëîïôöùûüç]+$", "", token.strip())
        if token:
            items.append(token)
    return items


def load_types_by_category(path: str) -> Dict[str, List[str]]:
    """
    Charge un mapping catégorie → [types de documents] depuis :
    - un fichier JSON strict
    - ou un fichier de type ``source_types = {...}`` (littéral Python).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    dict_src = _extract_python_container(data, "{", "}")
    mapping: Dict[str, List[str]] = ast.literal_eval(dict_src)
    if not isinstance(mapping, dict):
        raise RuntimeError(f"Format de mapping invalide dans {path}")
    return mapping


def load_examples(path: str) -> List[dict]:
    """
    Charge des exemples d'entraînement depuis :
    - un fichier JSON strict  (liste de dicts)
    - ou un fichier de type ``exemples = [...]`` (littéral Python).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    list_src = _extract_python_container(data, "[", "]")
    examples: List[dict] = ast.literal_eval(list_src)
    if not isinstance(examples, list) or not all(isinstance(x, dict) for x in examples):
        raise RuntimeError(f"Format d'exemples invalide dans {path}")
    return examples


# ---------------------------------------------------------------------------
# Conversion vers le format « chunks » (attendu par src/train.py)
# ---------------------------------------------------------------------------

def to_chunks_record(
    contenu: str,
    expressions: List[str],
    *,
    theme: str = "",
    categorie: str = "",
    type_du_document: Optional[str] = None,
    url: Optional[str] = None,
    date: Optional[str] = None,
    source: str = "synthetic",
) -> Dict[str, Any]:
    """
    Convertit un enregistrement généré vers le format chunks :

    .. code-block:: json

        {
          "source": "synthetic",
          "titre": null,
          "chunks": [{"chunk": "...", "expressions_caracteristiques": [...]}],
          "meta": {"theme": "...", "categorie": "...", ...}
        }

    Ce format est directement consommable par ``src/common.py::build_text_dataset``.
    """
    return {
        "source": source,
        "titre": None,
        "chunks": [
            {
                "chunk": contenu,
                "expressions_caracteristiques": expressions,
            }
        ],
        "meta": {
            "theme":           theme,
            "categorie":       categorie,
            "type_du_document": type_du_document,
            "url":             url,
            "date":            date or datetime.now().strftime("%Y-%m-%d"),
            "dataset":         source,
            "lang":            "fr",
        },
    }


def normalize_expressions(raw: Any) -> List[str]:
    """
    Normalise la valeur du champ expressions renvoyée par le LLM :
    liste de strings, string JSON, ou string unique.
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [raw] if raw else []
    return []
