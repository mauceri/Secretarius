"""Évalue un LayaClassifier sur un split de validation
(gen_corpus/laya_val.jsonl) — exactitude simple (commande exacte, "aucune"
compris). Cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(classifier, val_path: Path) -> float:
    total = 0
    correct = 0
    for line in Path(val_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        expected = row["answers"]["command"]
        command, _ = classifier.classify(row["state"])
        predicted = command or "aucune"
        total += 1
        if predicted == expected:
            correct += 1
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--val", default="gen_corpus/laya_val.jsonl")
    args = parser.parse_args()

    from router_service.laya_classifier import LayaClassifier
    classifier = LayaClassifier(args.checkpoint)
    accuracy = evaluate(classifier, Path(args.val))
    print(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
