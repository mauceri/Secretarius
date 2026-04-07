#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue un modèle GGUF sur un corpus JSONL en calculant le score Jaccard moyen
entre les expressions générées et les expressions de référence.

Exemple :
    python src/jaccard.py \\
        --input data/test.jsonl \\
        --model output/phi4-mini_merged/gguf/model-Q6_K.gguf
"""

import argparse
import json
import logging
import os
import sys
from typing import List

from tqdm import tqdm

# --- Chargement de llama_cpp --------------------------------------------------

_LLAMA_CPP_SEARCH_PATHS = [
    os.path.expanduser("~/llama_cpp"),
    os.path.expanduser("~/llama.cpp"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../llama_cpp")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../llama.cpp")),
]

for _p in _LLAMA_CPP_SEARCH_PATHS:
    if os.path.exists(_p):
        sys.path.insert(0, _p)
        break

try:
    from llama_cpp import Llama
except ImportError:
    print(
        "Erreur : module 'llama_cpp' introuvable.\n"
        f"Chemins testés : {_LLAMA_CPP_SEARCH_PATHS}\n"
        "Installez-le avec : pip install llama-cpp-python"
    )
    sys.exit(1)

# --- Prompts par défaut -------------------------------------------------------
# Importés ici pour cohérence avec les autres scripts, mais peuvent être
# surchargés en CLI si le modèle cible utilise un format différent.
from common import (
    SYSTEM_PROMPT_DEFAULT,
    USER_PREFIX_DEFAULT,
    clean_chunk_text,
)

ASSISTANT_TAG_DEFAULT = "<|assistant|>:"

# --- Logger -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("jaccard")


# --- Utilitaires --------------------------------------------------------------

def extract_json_list(text: str) -> List[str]:
    """Extrait le premier tableau JSON [...] trouvé dans le texte généré."""
    try:
        start = text.find("[")
        end   = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return []


def compute_jaccard(pred: List[str], ref: List[str]) -> float:
    set_pred = {p.strip().lower() for p in pred}
    set_ref  = {r.strip().lower() for r in ref}
    if not set_pred and not set_ref:
        return 1.0
    union = set_pred | set_ref
    return len(set_pred & set_ref) / len(union) if union else 0.0


def build_prompt(
    chunk_text: str, system_prompt: str, user_prefix: str, assistant_tag: str
) -> str:
    return (
        f"<|system|>: {system_prompt}\n"
        f"<|user|>: {user_prefix}{chunk_text}\n"
        f"{assistant_tag}"
    )


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Évaluation Jaccard d'un modèle GGUF sur un corpus JSONL."
    )
    p.add_argument("--input",         default="data/test.jsonl",
                   help="Corpus JSONL (format 'chunks').")
    p.add_argument("--model",         required=True,
                   help="Fichier modèle GGUF.")
    p.add_argument("--n_ctx",         type=int, default=2048)
    p.add_argument("--n_gpu_layers",  type=int, default=-1,
                   help="Couches sur GPU (-1 = tout).")
    p.add_argument("--max_tokens",    type=int, default=512)
    p.add_argument("--system_prompt", default=SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--user_prefix",   default=USER_PREFIX_DEFAULT)
    p.add_argument("--assistant_tag", default=ASSISTANT_TAG_DEFAULT)
    p.add_argument("--verbose",       action="store_true")
    return p.parse_args()


# --- Main ---------------------------------------------------------------------

def main():
    args = parse_args()
    log.info(f"Modèle : {args.model}")
    log.info(f"Input  : {args.input}")

    try:
        llm = Llama(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose,
        )
    except Exception as e:
        log.error(f"Chargement du modèle : {e}")
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
    except FileNotFoundError:
        log.error(f"Fichier introuvable : {args.input}")
        sys.exit(1)

    log.info(f"{len(lines)} lignes à évaluer")
    jaccard_scores = []

    for line in tqdm(lines, desc="Évaluation"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        items = []
        if "chunks" in row:
            titre = row.get("titre")
            for ch in row["chunks"]:
                exprs = ch.get("expressions_caracteristiques", [])
                if not exprs:
                    continue
                chunk_txt = clean_chunk_text(ch.get("chunk", ""))
                if titre:
                    chunk_txt = f"{titre}\n\n{chunk_txt}"
                items.append((chunk_txt, exprs))

        for chunk_txt, true_exprs in items:
            prompt = build_prompt(
                chunk_txt, args.system_prompt, args.user_prefix, args.assistant_tag
            )
            output = llm(
                prompt,
                max_tokens=args.max_tokens,
                stop=["<|endoftext|>", "<|end|>", "</s>"],
                echo=False,
                temperature=0.0,
            )
            generated = output["choices"][0]["text"]
            pred_list = extract_json_list(generated)
            score     = compute_jaccard(pred_list, true_exprs)
            jaccard_scores.append(score)

            if args.verbose:
                print(f"\n[Ref]  ({len(true_exprs)}) : {true_exprs}")
                print(f"[Pred] ({len(pred_list)})  : {pred_list}")
                print(f"[Jaccard] : {score:.4f}")
            else:
                tqdm.write(
                    f"Jaccard={score:.4f}  |  ref={len(true_exprs)}  pred={len(pred_list)}"
                )

    if jaccard_scores:
        avg = sum(jaccard_scores) / len(jaccard_scores)
        print(f"\n=== Résultat final ===")
        print(f"Jaccard moyen : {avg:.4f}  ({len(jaccard_scores)} chunks évalués)")
    else:
        print("Aucun score calculé.")


if __name__ == "__main__":
    main()
