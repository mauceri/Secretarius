#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline complet : split du corpus → évaluation du modèle fusionné.

Exemple :
    python src/run_pipeline.py \\
        --input   data/corpus_wiki40b_fr_indexed_100.jsonl \\
        --model   output/phi4-mini_merged/merged_hf \\
        --metrics_out metrics/eval_v1.json
"""

import argparse
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split corpus puis évaluation modèle.")

    # Split
    p.add_argument("--input",        required=True, help="Corpus JSONL source.")
    p.add_argument("--train",        default="data/train.jsonl")
    p.add_argument("--test",         default="data/test.jsonl")
    p.add_argument("--test_ratio",   type=float, default=0.1)
    p.add_argument("--max_examples", type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)

    # Évaluation
    p.add_argument("--model",            required=True,
                   help="Répertoire du modèle HF fusionné.")
    p.add_argument("--assistant_tag",    default="<|assistant|>:")
    p.add_argument("--max_len",          type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--num_proc",         type=int, default=0)
    p.add_argument("--metrics_out",      default="",
                   help="Fichier JSON de sortie pour les métriques.")
    p.add_argument("--hf_cache",         default=".hf_cache",
                   help="Répertoire de cache HuggingFace.")

    return p.parse_args()


def main():
    args     = parse_args()
    src_dir  = Path(__file__).parent
    hf_cache = Path(args.hf_cache).resolve()

    # Étape 1 : split
    subprocess.run(
        [
            "python", str(src_dir / "split_corpus.py"),
            "--input",        args.input,
            "--train",        args.train,
            "--test",         args.test,
            "--test_ratio",   str(args.test_ratio),
            "--max_examples", str(args.max_examples),
            "--seed",         str(args.seed),
        ],
        check=True,
    )

    # Étape 2 : évaluation
    env = os.environ.copy()
    env["HF_HOME"]           = str(hf_cache)
    env["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")

    eval_cmd = [
        "python", str(src_dir / "evaluate.py"),
        "--model",            args.model,
        "--data_file",        args.test,
        "--assistant_tag",    args.assistant_tag,
        "--max_len",          str(args.max_len),
        "--per_device_batch", str(args.per_device_batch),
        "--num_proc",         str(args.num_proc),
    ]
    if args.metrics_out:
        eval_cmd += ["--metrics_out", args.metrics_out]

    subprocess.run(eval_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
