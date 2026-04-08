#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split a JSONL corpus into train/test with a fixed seed."""

import argparse
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Split JSONL corpus into train/test")
    p.add_argument("--input", required=True, help="Source JSONL file")
    p.add_argument("--train", required=True, help="Train output JSONL")
    p.add_argument("--test", required=True, help="Test output JSONL")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (0-1)")
    p.add_argument("--max_examples", type=int, default=0, help="Limit total examples before split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    train_path = Path(args.train)
    test_path = Path(args.test)

    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("No data lines found")

    random.seed(args.seed)
    random.shuffle(lines)

    if args.max_examples and args.max_examples > 0:
        lines = lines[:args.max_examples]

    n_total = len(lines)
    n_test = max(1, int(round(n_total * args.test_ratio)))
    test_lines = lines[:n_test]
    train_lines = lines[n_test:]

    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    print(f"total={n_total} train={len(train_lines)} test={len(test_lines)}")


if __name__ == "__main__":
    main()
