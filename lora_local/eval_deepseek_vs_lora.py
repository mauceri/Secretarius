#!/usr/bin/env python3
"""
Comparaison DeepSeek V3 (prompt optimisé via DSPy) vs Wiki LoRA (llama.cpp)
sur la tâche d'extraction d'expressions caractéristiques.

Usage :
    source llenv/bin/activate
    source ~/.config/secrets.env
    python eval_deepseek_vs_lora.py [--n-total 100] [--n-eval 30] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib import request as urlrequest, error as urlerror

import dspy

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LLAMA_URL   = "http://127.0.0.1:8989/v1/chat/completions"
LLAMA_MODEL = "local-llama-cpp"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"

SYSTEM_PROMPT_LORA = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
USER_PREFIX = "Quelles sont les expressions clés contenues à l'identique dans ce texte :\n"

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_examples(jsonl_path: str, n: int, seed: int = 42) -> list[dict]:
    """Charge n exemples depuis un corpus JSONL.
    Supporte deux formats :
    - corpus_lora_reference : {"text": ..., "expressions_lora": [...]}
    - corpus wiki original  : {"chunks": [{"chunk": ..., "expressions_caracteristiques": [...]}]}
    """
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            # Format corpus_lora_reference
            if "expressions_lora" in doc:
                text = doc.get("text", "").strip()
                refs = doc.get("expressions_lora", [])
                if text and refs:
                    examples.append({"text": text, "reference": refs})
            # Format corpus wiki original
            else:
                for chunk in doc.get("chunks", []):
                    text = chunk.get("chunk", "").strip()
                    refs = chunk.get("expressions_caracteristiques", [])
                    if text and refs:
                        examples.append({"text": text, "reference": refs})
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n]


# ---------------------------------------------------------------------------
# Métrique F1 verbatim
# ---------------------------------------------------------------------------

def f1_verbatim(predicted: list[str], reference: list[str], text: str) -> dict:
    """F1 en filtrant les expressions non verbatim dans le texte."""
    pred_set = {e for e in predicted if isinstance(e, str) and e in text}
    ref_set  = {e for e in reference  if isinstance(e, str) and e in text}
    if not ref_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set  - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Appel Wiki LoRA (llama.cpp)
# ---------------------------------------------------------------------------

def lora_extract(text: str, timeout: float = 120.0) -> list[str]:
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_LORA},
            {"role": "user",   "content": USER_PREFIX + text},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urlrequest.Request(
        LLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        return _parse_json_list(content)
    except Exception as exc:
        print(f"  [LoRA] erreur : {exc}")
        return []


def _parse_json_list(text: str) -> list[str]:
    # Essai direct
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [x for x in result if isinstance(x, str)]
    except Exception:
        pass
    # Extraction du premier tableau JSON
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return [x for x in result if isinstance(x, str)]
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Module DSPy pour DeepSeek
# ---------------------------------------------------------------------------

class ExpressionExtractor(dspy.Signature):
    """Extrait les expressions caractéristiques d'un texte français.
    Répond UNIQUEMENT par un tableau JSON de chaînes.
    Inclut UNIQUEMENT les expressions qui apparaissent à l'identique dans le texte.
    """
    text: str = dspy.InputField(desc="Texte français à analyser")
    expressions_json: str = dspy.OutputField(
        desc="Tableau JSON de chaînes — expressions verbatim extraites du texte"
    )


class ExtractorModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ExpressionExtractor)

    def forward(self, text: str) -> dspy.Prediction:
        return self.predict(text=text)


def deepseek_extract(module: ExtractorModule, text: str) -> list[str]:
    try:
        pred = module(text=text)
        return _parse_json_list(pred.expressions_json)
    except Exception as exc:
        print(f"  [DeepSeek] erreur : {exc}")
        return []


# ---------------------------------------------------------------------------
# Métrique DSPy (pour l'optimiseur)
# ---------------------------------------------------------------------------

def dspy_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    predicted = _parse_json_list(prediction.expressions_json)
    reference = example.reference
    text      = example.text
    return f1_verbatim(predicted, reference, text)["f1"]


# ---------------------------------------------------------------------------
# Évaluation sur un ensemble d'exemples
# ---------------------------------------------------------------------------

def evaluate_lora(examples: list[dict]) -> list[dict]:
    results = []
    for i, ex in enumerate(examples):
        print(f"  LoRA {i+1}/{len(examples)}…", end="\r")
        t0 = time.time()
        predicted = lora_extract(ex["text"])
        elapsed   = time.time() - t0
        metrics   = f1_verbatim(predicted, ex["reference"], ex["text"])
        results.append({**metrics, "predicted": predicted, "elapsed_s": elapsed})
    print()
    return results


def evaluate_deepseek(module: ExtractorModule, examples: list[dict]) -> list[dict]:
    results = []
    for i, ex in enumerate(examples):
        print(f"  DeepSeek {i+1}/{len(examples)}…", end="\r")
        t0 = time.time()
        predicted = deepseek_extract(module, ex["text"])
        elapsed   = time.time() - t0
        metrics   = f1_verbatim(predicted, ex["reference"], ex["text"])
        results.append({**metrics, "predicted": predicted, "elapsed_s": elapsed})
    print()
    return results


def aggregate(results: list[dict]) -> dict:
    keys = ["precision", "recall", "f1"]
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",  default="data/corpus_wiki40b_fr_indexed_100.jsonl")
    parser.add_argument("--n-total", type=int, default=100)
    parser.add_argument("--n-eval",  type=int, default=30)
    parser.add_argument("--out",     default="eval_results.json")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Sauter l'étape d'optimisation DSPy (prompt de base uniquement)")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY manquant — lancez : source ~/.config/secrets.env")

    # --- Données ---
    print(f"Chargement de {args.n_total} exemples depuis {args.corpus}…")
    all_examples = load_examples(args.corpus, args.n_total)
    n_train = args.n_total - args.n_eval
    train_ex = all_examples[:n_train]
    eval_ex  = all_examples[n_train:]
    print(f"  {n_train} exemples d'optimisation, {len(eval_ex)} d'évaluation")

    # --- DSPy LM ---
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_base=DEEPSEEK_BASE_URL,
        api_key=api_key,
        max_tokens=2048,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

    module = ExtractorModule()

    # --- Optimisation ---
    if not args.no_optimize:
        print(f"\nOptimisation DSPy (BootstrapFewShot) sur {n_train} exemples…")
        train_dspy = [
            dspy.Example(text=ex["text"], reference=ex["reference"],
                         expressions_json=json.dumps(ex["reference"], ensure_ascii=False))
            .with_inputs("text")
            for ex in train_ex
        ]
        optimizer = dspy.BootstrapFewShot(metric=dspy_metric, max_bootstrapped_demos=4)
        module = optimizer.compile(module, trainset=train_dspy)
        print("  Optimisation terminée.")
    else:
        print("\nOptimisation ignorée (--no-optimize).")

    # --- Évaluation DeepSeek ---
    print(f"\nÉvaluation DeepSeek V3 sur {len(eval_ex)} exemples…")
    ds_results = evaluate_deepseek(module, eval_ex)
    ds_agg     = aggregate(ds_results)

    # --- Évaluation Wiki LoRA ---
    print(f"Évaluation Wiki LoRA sur {len(eval_ex)} exemples…")
    lora_results = evaluate_lora(eval_ex)
    lora_agg     = aggregate(lora_results)

    # --- Rapport ---
    print("\n" + "="*50)
    print("RÉSULTATS — F1 verbatim (moyenne sur", len(eval_ex), "exemples)")
    print("="*50)
    print(f"{'Modèle':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*50)
    print(f"{'Wiki LoRA':<20} {lora_agg['precision']:>10.4f} {lora_agg['recall']:>10.4f} {lora_agg['f1']:>10.4f}")
    print(f"{'DeepSeek V3':<20} {ds_agg['precision']:>10.4f} {ds_agg['recall']:>10.4f} {ds_agg['f1']:>10.4f}")
    print("="*50)

    # Quelques exemples qualitatifs
    print("\n--- 3 exemples qualitatifs ---")
    for i in range(min(3, len(eval_ex))):
        ex = eval_ex[i]
        print(f"\n[{i+1}] Texte (extrait) : {ex['text'][:120]}…")
        print(f"    Référence  : {ex['reference'][:5]}")
        print(f"    LoRA       : {lora_results[i]['predicted'][:5]}")
        print(f"    DeepSeek   : {ds_results[i]['predicted'][:5]}")

    # Sauvegarde
    output = {
        "config": {"n_train": n_train, "n_eval": len(eval_ex), "corpus": args.corpus},
        "lora":     {"aggregate": lora_agg, "details": lora_results},
        "deepseek": {"aggregate": ds_agg,   "details": ds_results},
        "examples": [{"text": ex["text"], "reference": ex["reference"]} for ex in eval_ex],
    }
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats complets sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()
