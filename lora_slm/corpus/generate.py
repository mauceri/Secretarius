#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère un corpus JSONL au format « chunks » en utilisant DSPy + un LLM distant.

Le corpus produit est directement utilisable par ``src/train.py`` sans conversion.

Exemple (OpenAI) :
    python corpus/generate.py \\
        --count 100 \\
        --output data/corpus_synth.jsonl \\
        --provider openai --model gpt-4o-mini

Exemple (DeepSeek) :
    DEEPSEEK_API_KEY=sk-... \\
    python corpus/generate.py \\
        --count 200 \\
        --output data/corpus_synth.jsonl \\
        --provider deepseek --model deepseek-chat

Exemple (modèle local via serveur OpenAI-compatible) :
    python corpus/generate.py \\
        --count 50 \\
        --output data/corpus_synth.jsonl \\
        --api-base http://localhost:8080/v1 \\
        --model Phi-4-mini-instruct-Q6_K.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Assure un répertoire de cache DSPy local avant d'importer dspy
if "DSPY_CACHEDIR" not in os.environ:
    _cache = os.path.join(os.getcwd(), ".dspy_cache")
    os.makedirs(_cache, exist_ok=True)
    os.environ["DSPY_CACHEDIR"] = _cache

import dspy
try:
    from dspy.clients import configure_cache as _dspy_configure_cache
    _dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=True)
except Exception:
    pass
dspy.settings.cache = None

# Utilitaires partagés (chemin relatif au script)
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_prompt,
    load_themes,
    load_categories,
    load_types_by_category,
    to_chunks_record,
    normalize_expressions,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    count:              int   = 100
    batch_size:         int   = 20
    report_every:       int   = 10
    output:             str   = "data/corpus_synth.jsonl"
    prompt_path:        str   = "corpus/prompts/prompt_init.txt"
    themes_path:        str   = "corpus/config/themes.json"
    categories_path:    str   = "corpus/config/categories.jsonl"
    types_map_path:     str   = "corpus/config/types_by_category.json"
    provider:           str   = "openai"
    model:              str   = "gpt-4o-mini"
    api_base:           str   = ""
    temperature:        float = 0.7
    max_attempts:       int   = 3
    seed:               Optional[int] = None


# ---------------------------------------------------------------------------
# Signature DSPy
# ---------------------------------------------------------------------------

def build_predictor(prompt_text: str) -> dspy.Predict:
    class GenerateNote(dspy.Signature):
        f"""{prompt_text}"""

        theme:            str  = dspy.InputField(desc="Thème principal de la note")
        categorie:        str  = dspy.InputField(
            desc="Type de note : extrait, commentaire, synthese, texte_libre"
        )
        type_du_document: str  = dspy.InputField(
            desc="Type du document source (roman, article, tweet…) ; vide si texte_libre"
        )

        contenu:                   str  = dspy.OutputField(
            desc="Contenu de la note en français"
        )
        url:                       str  = dspy.OutputField(
            desc="URL ou référence du document (inventée)"
        )
        date:                      str  = dspy.OutputField(
            desc="Date imaginaire au format ISO YYYY-MM-DD"
        )
        expressions_caracteristiques: list = dspy.OutputField(
            desc="Expressions caractéristiques apparaissant à l'identique dans contenu"
        )

    return dspy.Predict(GenerateNote)


# ---------------------------------------------------------------------------
# Configuration du LM
# ---------------------------------------------------------------------------

def configure_lm(provider: str, model: str, temperature: float, api_base: str) -> None:
    provider = provider.lower()

    # Résolution de la clé API selon le fournisseur
    api_key = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Définissez la variable OPENAI_API_KEY")
        composed = f"openai/{model}" if "/" not in model else model
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Définissez la variable DEEPSEEK_API_KEY")
        api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        composed = f"openai/{model}" if "/" not in model else model
    else:
        # Fournisseur générique (serveur OpenAI-compatible, etc.)
        api_key = os.getenv("OPENAI_API_KEY", "not-needed")
        composed = model

    kwargs: Dict[str, Any] = dict(
        model=composed,
        temperature=temperature,
        max_tokens=2000,
        cache=False,
    )
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    lm = dspy.LM(**kwargs)
    dspy.settings.configure(lm=lm)


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------

def choose_triplet(
    themes: List[str],
    categories: List[str],
    types_map: Dict[str, List[str]],
) -> tuple[str, str, str]:
    theme     = random.choice(themes)
    categorie = random.choice(categories)
    types     = types_map.get(categorie) or [""]
    type_doc  = random.choice(types)
    return theme, categorie, type_doc


def generate_one(
    predictor: dspy.Predict,
    theme: str,
    categorie: str,
    type_du_document: str,
) -> Optional[Dict[str, Any]]:
    pred = predictor(
        theme=theme,
        categorie=categorie,
        type_du_document=type_du_document or "",
    )
    contenu     = str(getattr(pred, "contenu", "") or "").strip()
    expressions = normalize_expressions(getattr(pred, "expressions_caracteristiques", []))
    url         = str(getattr(pred, "url",  "") or "")
    date        = str(getattr(pred, "date", "") or "")

    if not contenu or not expressions:
        return None

    return to_chunks_record(
        contenu,
        expressions,
        theme=theme,
        categorie=categorie,
        type_du_document=type_du_document or None,
        url=url or None,
        date=date or None,
    )


def generate_corpus(cfg: Config) -> None:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    prompt_text = load_prompt(cfg.prompt_path)
    themes      = load_themes(cfg.themes_path)
    categories  = load_categories(cfg.categories_path)
    types_map   = load_types_by_category(cfg.types_map_path)

    configure_lm(cfg.provider, cfg.model, cfg.temperature, cfg.api_base)
    predictor = build_predictor(prompt_text)

    os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
    generated = 0
    stime     = time.time()

    warnings.filterwarnings("ignore")
    with open(cfg.output, "w", encoding="utf-8") as fout:
        while generated < cfg.count:
            theme, categorie, type_doc = choose_triplet(themes, categories, types_map)
            for attempt in range(cfg.max_attempts):
                try:
                    record = generate_one(predictor, theme, categorie, type_doc)
                except Exception as e:
                    if attempt == cfg.max_attempts - 1:
                        print(
                            f"[!] Erreur ({categorie}/{type_doc}): {e}",
                            file=sys.stderr,
                        )
                    continue

                if record is None:
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1

                if generated % cfg.report_every == 0:
                    elapsed = time.time() - stime
                    print(f"[{generated}/{cfg.count}] {elapsed:.1f}s")
                break

    print(f"Corpus : {generated} entrées → {cfg.output}  ({time.time()-stime:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Génère un corpus JSONL (format chunks) via DSPy."
    )
    p.add_argument("--count",       type=int,   default=100)
    p.add_argument("--output",      default="data/corpus_synth.jsonl")
    p.add_argument("--prompt",      default="corpus/prompts/prompt_init.txt")
    p.add_argument("--themes",      default="corpus/config/themes.json")
    p.add_argument("--categories",  default="corpus/config/categories.jsonl")
    p.add_argument("--types-map",   default="corpus/config/types_by_category.json")
    p.add_argument("--provider",    default="openai",
                   help="openai | deepseek | ou tout fournisseur litellm")
    p.add_argument("--model",       default="gpt-4o-mini")
    p.add_argument("--api-base",    default="",
                   help="Endpoint custom (serveur OpenAI-compatible local, etc.)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-attempts",type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=20,
                   help="(ignoré, conservé pour compatibilité)")
    p.add_argument("--report-every",type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=None)
    a = p.parse_args()
    return Config(
        count=a.count, output=a.output,
        prompt_path=a.prompt, themes_path=a.themes,
        categories_path=a.categories, types_map_path=a.types_map,
        provider=a.provider, model=a.model, api_base=a.api_base,
        temperature=a.temperature, max_attempts=a.max_attempts,
        report_every=a.report_every, seed=a.seed,
    )


if __name__ == "__main__":
    generate_corpus(parse_args())
