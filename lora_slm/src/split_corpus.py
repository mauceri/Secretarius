#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Découpe un corpus JSONL en jeux train et test avec une graine fixe."""

import argparse
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split JSONL → train / test")
    p.add_argument("--input",        required=True, help="Fichier JSONL source.")
    p.add_argument("--train",        required=True, help="Fichier train en sortie.")
    p.add_argument("--test",         required=True, help="Fichier test en sortie.")
    p.add_argument("--test_ratio",   type=float, default=0.1,
                   help="Part réservée au test (0–1, défaut 0.1).")
    p.add_argument("--max_examples", type=int, default=0,
                   help="Limite le nombre total d'exemples avant le split (0 = tout).")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    src  = Path(args.input)

    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise SystemExit("Aucune donnée trouvée dans le fichier source.")

    random.seed(args.seed)
    random.shuffle(lines)

    if args.max_examples and args.max_examples > 0:
        lines = lines[:args.max_examples]

    n_test  = max(1, int(round(len(lines) * args.test_ratio)))
    test_lines  = lines[:n_test]
    train_lines = lines[n_test:]

    Path(args.train).write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    Path(args.test ).write_text("\n".join(test_lines)  + "\n", encoding="utf-8")

    print(f"total={len(lines)}  train={len(train_lines)}  test={len(test_lines)}")


if __name__ == "__main__":
    main()
