#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimise le prompt de génération via DSPy.GEPA, puis génère un corpus.

DSPy.GEPA (Generalized Prompt Approximation) fait évoluer le prompt par
réflexion : à chaque itération il compare les sorties du générateur aux
exemples de référence via une métrique sémantique (TextComparatorGlobal),
et propose un prompt amélioré.

Le meilleur prompt trouvé est sauvegardé dans ``corpus/prompts/prompt_gepa.txt``
et peut ensuite être utilisé par ``corpus/generate.py --prompt``.

Exemple (DeepSeek) :
    DEEPSEEK_API_KEY=sk-... \\
    python corpus/optimize_prompt.py \\
        --count 50 \\
        --output data/corpus_gepa.jsonl

Exemple (OpenAI) :
    OPENAI_API_KEY=sk-... \\
    python corpus/optimize_prompt.py \\
        --provider openai --model gpt-4o \\
        --count 50 --output data/corpus_gepa.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cache DSPy local
if "DSPY_CACHEDIR" not in os.environ:
    _cache = os.path.join(os.getcwd(), ".dspy_cache")
    os.makedirs(_cache, exist_ok=True)
    os.environ["DSPY_CACHEDIR"] = _cache

import dspy
try:
    from dspy.clients import configure_cache as _dspy_configure_cache
    _dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None

logging.basicConfig(
    filename="logs/gepa.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_prompt,
    load_themes,
    load_categories,
    load_types_by_category,
    load_examples,
    to_chunks_record,
    normalize_expressions,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    count:               int   = 100
    batch_size:          int   = 20
    report_every:        int   = 10
    output:              str   = "data/corpus_gepa.jsonl"
    prompt_path:         str   = "corpus/prompts/prompt_init.txt"
    gepa_prompt_path:    str   = "corpus/prompts/prompt_gepa.txt"
    themes_path:         str   = "corpus/config/themes.json"
    categories_path:     str   = "corpus/config/categories.jsonl"
    types_map_path:      str   = "corpus/config/types_by_category.json"
    examples_path:       str   = "corpus/config/examples.json"
    log_file:            str   = "logs/gepa_corpus.jsonl"
    provider:            str   = "deepseek"
    generator_model:     str   = "deepseek-chat"
    comparator_model:    str   = "deepseek-chat"
    api_base:            str   = "https://api.deepseek.com"
    temperature:         float = 1.0
    max_metric_calls:    int   = 500
    cost_in:             float = 0.27 / 1_000_000
    cost_out:            float = 1.10 / 1_000_000


# ---------------------------------------------------------------------------
# Helpers LM
# ---------------------------------------------------------------------------

def _resolve_api_key(provider: str) -> str:
    provider = provider.lower()
    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("Définissez la variable DEEPSEEK_API_KEY")
    else:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Définissez la variable OPENAI_API_KEY")
    return key


def _make_lm(
    provider: str, model: str, temperature: float, api_base: str, max_tokens: int
) -> dspy.LM:
    key      = _resolve_api_key(provider)
    base     = api_base or os.getenv("DEEPSEEK_API_BASE", "")
    composed = f"openai/{model}" if "/" not in model else model
    kwargs: Dict[str, Any] = dict(
        model=composed, api_key=key, model_type="chat",
        temperature=temperature, max_tokens=max_tokens, cache=False,
    )
    if base:
        kwargs["api_base"] = base
    return dspy.LM(**kwargs)


def configure_generator_lm(cfg: Config) -> dspy.LM:
    lm = _make_lm(cfg.provider, cfg.generator_model, cfg.temperature, cfg.api_base, 16000)
    dspy.settings.configure(lm=lm)
    return lm


def configure_comparator_lm(cfg: Config) -> dspy.LM:
    lm = _make_lm(cfg.provider, cfg.comparator_model, 0.0, cfg.api_base, 128)
    # Logging des appels comparateur
    _orig = lm.__call__
    @wraps(_orig)
    def _logged(*a, **kw):
        logging.info(f"[COMP IN] {a} {kw}")
        res = _orig(*a, **kw)
        logging.info(f"[COMP OUT] {res}")
        return res
    lm.__call__ = _logged
    return lm


# ---------------------------------------------------------------------------
# Signatures DSPy
# ---------------------------------------------------------------------------

def build_note_generator(prompt_text: str) -> dspy.Module:
    class GenerateNote(dspy.Signature):
        f"""{prompt_text}"""

        theme:            str  = dspy.InputField(desc="Thème principal de la note")
        categorie:        str  = dspy.InputField(
            desc="Type de note : extrait, commentaire, synthese, texte_libre"
        )
        type_du_document: str  = dspy.InputField(
            desc="Type du document source ; vide si texte_libre"
        )
        contenu:                      str  = dspy.OutputField(
            desc="Contenu de la note en français"
        )
        url:                          str  = dspy.OutputField(
            desc="URL ou référence inventée"
        )
        date:                         str  = dspy.OutputField(
            desc="Date imaginaire ISO YYYY-MM-DD"
        )
        expressions_caracteristiques: list = dspy.OutputField(
            desc="Expressions apparaissant à l'identique dans contenu"
        )

    class NoteGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(GenerateNote)

        def forward(self, categorie: str, theme: str, type_du_document: str) -> dict:
            r = self.generate(
                theme=theme, categorie=categorie, type_du_document=type_du_document
            )
            return {
                "contenu":                      str(getattr(r, "contenu", "") or ""),
                "url":                          str(getattr(r, "url",    "") or ""),
                "date":                         str(getattr(r, "date",   "") or ""),
                "expressions_caracteristiques": normalize_expressions(
                    getattr(r, "expressions_caracteristiques", [])
                ),
                "categorie":       categorie,
                "theme":           theme,
                "type_du_document": type_du_document,
            }

    return NoteGenerator()


class CompareTextsGlobal(dspy.Signature):
    """
    Compare deux textes et produit un score 1–5 (thème, ton, style).
    5 = très ressemblant, 1 = très différent.
    Sortie : un entier 1–5, sans commentaire.
    """
    text_a: str = dspy.InputField(desc="Premier texte")
    text_b: str = dspy.InputField(desc="Second texte")
    score:  int = dspy.OutputField(desc="Entier 1–5, score global")


class TextComparatorGlobal(dspy.Module):
    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.pred = dspy.Predict(CompareTextsGlobal)
        self.lm   = lm

    def forward(self, text_a: str, text_b: str) -> int:
        ctx = dspy.settings.context(lm=self.lm) if self.lm else None
        if ctx:
            with ctx:
                out = self.pred(text_a=text_a, text_b=text_b)
        else:
            out = self.pred(text_a=text_a, text_b=text_b)
        try:
            s = int(getattr(out, "score", 3))
        except Exception:
            s = 3
        return max(1, min(5, s))


# ---------------------------------------------------------------------------
# Métrique sémantique
# ---------------------------------------------------------------------------

def make_semantic_metric(comparator: TextComparatorGlobal):
    counter = {"n": 0}

    def metric(gold, pred, trace=None, **_) -> float:
        gold_text = str((gold or {}).get("contenu") or "")
        pred_text = str((pred or {}).get("contenu") or "")
        if not gold_text or not pred_text:
            return 0.1
        try:
            score = comparator(text_a=gold_text, text_b=pred_text) / 5.0
        except Exception:
            score = 0.6
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"  [GEPA] itération {counter['n']}  score={score:.2f}")
        return score

    return metric


# ---------------------------------------------------------------------------
# Entraînement GEPA et extraction du meilleur prompt
# ---------------------------------------------------------------------------

def build_trainset(examples: List[dict]) -> List[dspy.Example]:
    return [
        dspy.Example(
            categorie=ex.get("categorie", ""),
            theme=ex.get("theme", ""),
            type_du_document=ex.get("type_du_document", ""),
            contenu=ex.get("contenu", ""),
            url=ex.get("url", ""),
            date=ex.get("date", ""),
        ).with_inputs("categorie", "theme", "type_du_document")
        for ex in examples
    ]


def _ensure_text(val: Any) -> str:
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def _is_placeholder(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    return ("given the fields" in low and "produce the fields" in low) or len(low.strip()) < 40


def extract_best_prompt(compiled, teleprompter, initial: str) -> str:
    candidates: List[str] = []
    for attr in ("best_prompt", "best_prompt_str", "best_prompt_text"):
        v = getattr(teleprompter, attr, None)
        if v:
            candidates.append(_ensure_text(v))
    bp = getattr(teleprompter, "best_prompts", None)
    if isinstance(bp, dict):
        candidates.extend(_ensure_text(v) for v in bp.values() if v)
    for obj in (getattr(compiled, "generate", None), compiled):
        sig = getattr(obj, "signature", None) if obj else None
        if sig:
            instr = getattr(sig, "instructions", None) or getattr(sig, "__doc__", None)
            if instr:
                candidates.append(_ensure_text(instr))
    candidates.append(initial)
    for c in candidates:
        if not _is_placeholder(c):
            return c
    return initial


def train_generator(
    prompt_text: str, examples: List[dict], comparator: TextComparatorGlobal, cfg: Config
):
    generator = build_note_generator(prompt_text)
    trainset  = build_trainset(examples)
    tp = dspy.GEPA(
        metric=make_semantic_metric(comparator),
        reflection_lm=dspy.settings.lm,
        max_metric_calls=cfg.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
    )
    try:
        compiled = tp.compile(generator, trainset=trainset)
    except RuntimeError as e:
        raise RuntimeError(f"Échec GEPA : {e}") from e
    best = extract_best_prompt(compiled, tp, prompt_text)
    return compiled, best


# ---------------------------------------------------------------------------
# Génération du corpus avec le générateur compilé
# ---------------------------------------------------------------------------

def generate_corpus(compiled, cfg: Config, categories, types_map, themes) -> None:
    total_in = total_out = 0
    stime    = time.time()
    pending: List[dict] = []

    os.makedirs(os.path.dirname(cfg.output)   or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.log_file) or ".", exist_ok=True)

    with (
        open(cfg.output,   "w", encoding="utf-8") as fout,
        open(cfg.log_file, "w", encoding="utf-8") as flog,
    ):
        for i in range(cfg.count):
            cat    = random.choice(categories)
            types  = types_map.get(cat) or [""]
            type_d = random.choice(types)
            theme  = random.choice(themes)

            note = compiled(categorie=cat, theme=theme, type_du_document=type_d)
            flog.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "input":  {"categorie": cat, "theme": theme, "type_du_document": type_d},
                "output": note,
            }, ensure_ascii=False) + "\n")

            total_in  += sum(len(str(v).split()) for v in [cat, type_d, theme])
            total_out += len(str(note.get("contenu", "")).split())

            record = to_chunks_record(
                note.get("contenu", ""),
                note.get("expressions_caracteristiques", []),
                theme=theme, categorie=cat,
                type_du_document=type_d or None,
                url=note.get("url") or None,
                date=note.get("date") or None,
            )
            pending.append(record)

            if len(pending) >= cfg.batch_size:
                for r in pending:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                pending = []

            if (i + 1) % cfg.report_every == 0:
                elapsed  = time.time() - stime
                cost_est = total_in * cfg.cost_in + total_out * cfg.cost_out
                print(
                    f"[{i+1}/{cfg.count}] {elapsed:.1f}s  "
                    f"tokens_in={total_in} tokens_out={total_out}  "
                    f"coût≈{cost_est:.3f}$"
                )

        for r in pending:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Corpus → {cfg.output}  ({time.time()-stime:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Optimise le prompt via GEPA puis génère un corpus."
    )
    p.add_argument("--count",            type=int,   default=100)
    p.add_argument("--output",           default="data/corpus_gepa.jsonl")
    p.add_argument("--prompt",           default="corpus/prompts/prompt_init.txt")
    p.add_argument("--gepa-prompt",      default="corpus/prompts/prompt_gepa.txt")
    p.add_argument("--themes",           default="corpus/config/themes.json")
    p.add_argument("--categories",       default="corpus/config/categories.jsonl")
    p.add_argument("--types-map",        default="corpus/config/types_by_category.json")
    p.add_argument("--examples",         default="corpus/config/examples.json")
    p.add_argument("--log-file",         default="logs/gepa_corpus.jsonl")
    p.add_argument("--provider",         default="deepseek",
                   help="deepseek | openai | autre litellm")
    p.add_argument("--generator-model",  default="deepseek-chat")
    p.add_argument("--comparator-model", default="deepseek-chat")
    p.add_argument("--api-base",         default="https://api.deepseek.com")
    p.add_argument("--temperature",      type=float, default=1.0)
    p.add_argument("--max-metric-calls", type=int,   default=500)
    p.add_argument("--report-every",     type=int,   default=10)
    p.add_argument("--batch-size",       type=int,   default=20)
    a = p.parse_args()
    return Config(
        count=a.count, output=a.output,
        prompt_path=a.prompt, gepa_prompt_path=a.gepa_prompt,
        themes_path=a.themes, categories_path=a.categories,
        types_map_path=a.types_map, examples_path=a.examples,
        log_file=a.log_file, provider=a.provider,
        generator_model=a.generator_model, comparator_model=a.comparator_model,
        api_base=a.api_base, temperature=a.temperature,
        max_metric_calls=a.max_metric_calls, batch_size=a.batch_size,
        report_every=a.report_every,
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs("logs", exist_ok=True)

    prompt_text = load_prompt(cfg.prompt_path)
    themes      = load_themes(cfg.themes_path)
    categories  = load_categories(cfg.categories_path)
    types_map   = load_types_by_category(cfg.types_map_path)
    examples    = load_examples(cfg.examples_path)

    configure_generator_lm(cfg)
    lm_comp    = configure_comparator_lm(cfg)
    comparator = TextComparatorGlobal(lm=lm_comp)

    print("Optimisation GEPA en cours…")
    compiled, best_prompt = train_generator(prompt_text, examples, comparator, cfg)

    Path(cfg.gepa_prompt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.gepa_prompt_path).write_text(best_prompt, encoding="utf-8")
    print(f"Meilleur prompt sauvegardé → {cfg.gepa_prompt_path}")

    print("Génération du corpus…")
    generate_corpus(compiled, cfg, categories, types_map, themes)


if __name__ == "__main__":
    main()
