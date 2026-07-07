#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convertit corpus_qa.jsonl en ChatML pour fine-tuning LoRA (phi-4-mini)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT_QA = (
    "Vous êtes Tiron. Répondez à la question en vous appuyant uniquement sur le "
    "document fourni. Soyez concis et répondez en français. Si la réponse ne figure "
    "pas dans le document, indiquez-le clairement sans rien inventer."
)


def convert_entry_qa(entry: dict) -> dict:
    user = f"Document:\n{entry['document']}\n\nQuestion: {entry['question']}"
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT_QA},
        {"role": "user", "content": user},
        {"role": "assistant", "content": entry["answer"]},
    ]}


def to_lora(corpus_path: str, train_path: str, eval_path: str,
            eval_ratio: float = 0.1, seed: int = 42) -> None:
    lines = [l for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    converted = [convert_entry_qa(json.loads(l)) for l in lines]
    random.seed(seed)
    random.shuffle(converted)
    n_eval = max(1, int(len(converted) * eval_ratio))
    for path, data in [(train_path, converted[n_eval:]), (eval_path, converted[:n_eval])]:
        Path(path).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in data), encoding="utf-8"
        )
    print(f"Total: {len(converted)} | Train: {len(converted)-n_eval} | Eval: {n_eval}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus_qa.jsonl")
    p.add_argument("--train", default="corpus_qa_train.jsonl")
    p.add_argument("--eval", default="corpus_qa_eval.jsonl")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    a = p.parse_args()
    to_lora(a.corpus, a.train, a.eval, a.eval_ratio)


if __name__ == "__main__":
    main()
