#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split a JSONL corpus, then run evaluation on the test split."""

import argparse
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Split corpus then evaluate")
    p.add_argument("--input", required=True, help="Source JSONL file")
    p.add_argument("--train", default="data/train.jsonl", help="Train output JSONL")
    p.add_argument("--test", default="data/test.jsonl", help="Test output JSONL")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (0-1)")
    p.add_argument("--max_examples", type=int, default=0, help="Limit total examples before split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", required=True, help="HF merged model directory")
    p.add_argument("--assistant_tag", default="<|assistant|>:")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=0)
    p.add_argument("--metrics_out", default="", help="Chemin fichier pour sauvegarder les metriques (JSON)")
    p.add_argument("--hf_cache", default=".hf_cache", help="HF cache directory")
    return p.parse_args()


def main():
    args = parse_args()
    hf_cache = Path(args.hf_cache)
    hf_home = str(hf_cache.resolve())
    hf_datasets = str((hf_cache / "datasets").resolve())

    split_cmd = [
        "python", "split_corpus.py",
        "--input", args.input,
        "--train", args.train,
        "--test", args.test,
        "--test_ratio", str(args.test_ratio),
        "--max_examples", str(args.max_examples),
        "--seed", str(args.seed),
    ]
    subprocess.run(split_cmd, check=True)

    env = os.environ.copy()
    env["HF_HOME"] = hf_home
    env["HF_DATASETS_CACHE"] = hf_datasets

    eval_cmd = [
        "python", "evaluate_merged.py",
        "--model", args.model,
        "--data_file", args.test,
        "--assistant_tag", args.assistant_tag,
        "--max_len", str(args.max_len),
        "--per_device_batch", str(args.per_device_batch),
        "--num_proc", str(args.num_proc),
    ]
    if args.metrics_out:
        eval_cmd += ["--metrics_out", args.metrics_out]
    subprocess.run(eval_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
