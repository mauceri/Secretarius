#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue un modèle GGUF sur un corpus JSONL en calculant le score Jaccard moyen
entre les expressions générées et les expressions de référence.

Exemple :
    python jaccard.py --input data/corpus_gutenberg_indexed.jsonl --model test_wikipedia_gguf/model-Q6_K.gguf
"""

import argparse
import json
import logging
import sys
import os
from typing import List, Dict, Any, Set
from tqdm import tqdm

# Configuration des chemins possibles pour llama_cpp
potential_paths = [
    "/home/mauceric/mn_sanroque/llama_cpp",  # Chemin spécifié par l'utilisateur
    "/home/mauceric/llama_cpp",               # Autre chemin mentionné
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama_cpp")), # Relatif au script
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp"))  # Relatif (tiret)
]

llama_cpp_found = False
for path in potential_paths:
    if os.path.exists(path):
        sys.path.append(path)
        llama_cpp_found = True
        # On continue pour autoriser plusieurs ajouts si nécessaire, 
        # mais le premier valide suffit souvent. 
        # On break pas forcément pour maximiser les chances si imports croisés.
        break

try:
    from llama_cpp import Llama
except ImportError:
    print("Erreur : le module 'llama_cpp' est introuvable.\n"
          f"Chemins testés : {potential_paths}\n"
          "Installez-le avec `pip install llama-cpp-python` ou vérifiez les chemins.")
    sys.exit(1)


# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("jaccard")


def clean_chunk_text(text: str) -> str:
    """Nettoie le texte du chunk (enlève b"..." si présent)."""
    if text.startswith('b"') and text.endswith('"'):
        return text[2:-1]
    return text


def extract_json_list(text: str) -> List[str]:
    """Extrait le premier tableau JSON [...] trouvé dans le texte."""
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            data = json.loads(json_str)
            if isinstance(data, list):
                return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return []


def compute_jaccard_similarity(pred: List[str], ref: List[str]) -> float:
    set_pred = set(p.strip().lower() for p in pred)
    set_ref = set(r.strip().lower() for r in ref)
    if not set_pred and not set_ref:
        return 1.0
    intersection = set_pred.intersection(set_ref)
    union = set_pred.union(set_ref)
    return len(intersection) / len(union) if union else 0.0


def construct_prompt(chunk_text: str, system_prompt: str, user_prefix: str, assistant_tag: str) -> str:
    """Construit le prompt complet pour le modèle."""
    # Format compatible avec celui utilisé lors de l'entraînement ou du merge
    # <|system|>: ...
    # <|user|>: ...
    # <|assistant|>:
    prompt = f"<|system|>: {system_prompt}\n<|user|>: {user_prefix}{chunk_text}\n{assistant_tag}"
    return prompt


SYSTEM_PROMPT_DEFAULT = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
USER_PREFIX_DEFAULT = "Quelles sont les expressions clés contenues à l'identique dans ce texte : "
ASSISTANT_TAG_DEFAULT = "<|assistant|>:"


def parse_args():
    p = argparse.ArgumentParser("Évaluation Jaccard d'un modèle GGUF.")
    p.add_argument("--input", default="data/corpus_gutenberg_indexed.jsonl", help="Fichier JSONL d'entrée.")
    p.add_argument("--model", default="test_wikipedia_gguf/model-Q6_K.gguf", help="Chemin du modèle GGUF.")
    p.add_argument("--n_ctx", type=int, default=2048, help="Taille du contexte.")
    p.add_argument("--n_gpu_layers", type=int, default=-1, help="Nombre de couches sur GPU (-1 = tout).")
    p.add_argument("--verbose", action="store_true", help="Afficher les détails.")
    return p.parse_args()


def main():
    args = parse_args()
    
    log.info(f"Modèle : {args.model}")
    log.info(f"Input : {args.input}")
    
    # Chargement du modèle
    try:
        llm = Llama(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose
        )
    except Exception as e:
        log.error(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)
        
    jaccard_scores = []
    
    # Lecture du fichier
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        log.error(f"Fichier introuvable : {args.input}")
        sys.exit(1)
        
    log.info(f"Nombre d'exemples : {len(lines)}")
    
    for line in tqdm(lines, desc="Évaluation"):
        line = line.strip()
        if not line:
            continue
            
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        # Extraction prompt / reference
        items_to_process = []
        
        if "chunks" in row:
            titre = row.get("titre")
            for ch in row["chunks"]:
                exprs = ch.get("expressions_caracteristiques", [])
                if not exprs:
                    continue
                
                chunk_txt = clean_chunk_text(ch.get("chunk", ""))
                if titre:
                    chunk_txt = f"{titre}\n\n{chunk_txt}"
                items_to_process.append((chunk_txt, exprs))
                
        elif "messages" in row:
             # Support basique pour format messages si besoin
             pass
            
        for chunk_txt, true_exprs in items_to_process:
            prompt = construct_prompt(
                chunk_txt, 
                SYSTEM_PROMPT_DEFAULT, 
                USER_PREFIX_DEFAULT, 
                ASSISTANT_TAG_DEFAULT
            )
            
            # Génération
            output = llm(
                prompt, 
                max_tokens=512, 
                stop=["<|endoftext|>", "<|end|>", "</s>"], # Ajout de </s> au cas où
                echo=False,
                temperature=0.0
            )
            generated_text = output["choices"][0]["text"]
            
            # Parsing et Score
            pred_list = extract_json_list(generated_text)
            score = compute_jaccard_similarity(pred_list, true_exprs)
            
            jaccard_scores.append(score)
            
            # Resultat intermédiaire si verbose
            len_ref = len(true_exprs)
            len_pred = len(pred_list)
            
            if args.verbose:
                print(f"\n[Ref] (len={len_ref}): {true_exprs}")
                print(f"[Pred] (len={len_pred}): {pred_list}")
                print(f"[Jaccard]: {score:.4f}")
            else:
                 # Afficher juste le score courant et les longueurs
                 tqdm.write(f"Jaccard: {score:.4f} | Len Ref: {len_ref} | Len Pred: {len_pred}")

    if jaccard_scores:
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        print(f"\n=== Résultat Final ===")
        print(f"Jaccard moyen : {avg_jaccard:.4f}")
    else:
        print("Aucun score calculé.")


if __name__ == "__main__":
    main()
