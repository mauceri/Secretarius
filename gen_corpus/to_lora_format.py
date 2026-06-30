#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convertit corpus.jsonl en format ChatML pour fine-tuning LoRA (phi-4-mini-instruct)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    'Routeur de commandes Tiron. Pour chaque message, répondre uniquement avec un objet JSON : '
    '{"command": "/commande" ou null, "args": "arguments bruts ou chaîne vide"}.'
)


def convert_entry(entry: dict) -> dict:
    action = entry["action"]
    return {"messages": [
        {"role": "system",   "content": SYSTEM_PROMPT},
        {"role": "user",     "content": entry["text"]},
        {"role": "assistant","content": json.dumps(
            {"command": action["command"], "args": action["args"]}, ensure_ascii=False
        )},
    ]}


def to_lora(corpus_path: str, out_path: str, train_path: str, eval_path: str,
            eval_ratio: float = 0.1, seed: int = 42) -> None:
    lines = [l for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    converted = [convert_entry(json.loads(l)) for l in lines]
    random.seed(seed)
    random.shuffle(converted)
    n_eval = max(1, int(len(converted) * eval_ratio))
    for path, data in [(out_path, converted),
                       (train_path, converted[n_eval:]),
                       (eval_path, converted[:n_eval])]:
        Path(path).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in data), encoding="utf-8"
        )
    print(f"Total: {len(converted)} | Train: {len(converted)-n_eval} | Eval: {n_eval}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus.jsonl")
    p.add_argument("--output", default="corpus_lora.jsonl")
    p.add_argument("--train", default="corpus_lora_train.jsonl")
    p.add_argument("--eval", default="corpus_lora_eval.jsonl")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    a = p.parse_args()
    to_lora(a.corpus, a.output, a.train, a.eval, a.eval_ratio)


if __name__ == "__main__":
    main()
