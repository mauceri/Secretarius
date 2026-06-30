#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation manuelle du corpus par échantillonnage et statistiques."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus.jsonl")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    content = Path(a.corpus).read_text(encoding="utf-8")

    # Support both JSON array and JSONL formats
    if content.strip().startswith("["):
        # JSON array format
        entries = json.loads(content)
    else:
        # JSONL format
        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            print("Corpus vide.")
            return
        entries = [json.loads(l) for l in lines]

    if not entries:
        print("Corpus vide.")
        return

    int_c  = Counter(e["intention"]             for e in entries)
    reg_c  = Counter(e.get("registre", "?")     for e in entries)
    cmd_c  = Counter(str(e["action"]["command"]) for e in entries)

    print(f"\nCorpus : {len(entries)} entrées — {a.corpus}")
    print("\nDistribution des intentions :")
    for k, v in sorted(int_c.items()):
        bar = "█" * (v * 30 // len(entries))
        print(f"  {k:22s} {v:4d} ({v/len(entries)*100:4.1f}%) {bar}")
    print("\nDistribution des registres :")
    for k, v in sorted(reg_c.items()):
        print(f"  {k:15s} {v:4d} ({v/len(entries)*100:4.1f}%)")
    print("\nDistribution des commandes :")
    for k, v in sorted(cmd_c.items()):
        print(f"  {k:15s} {v:4d}")

    if a.seed is not None:
        random.seed(a.seed)
    sample = random.sample(entries, min(a.sample, len(entries)))
    print(f"\n{'─'*60}\nÉchantillon ({len(sample)}) :\n{'─'*60}")
    for e in sample:
        print(f"\n[{e['intention']}] registre={e.get('registre','?')} variante={e.get('variante','?')}")
        print(f"  TEXT : {e['text']}")
        print(f"  CMD  : {e['action']['command']}   ARGS : {e['action']['args']}")


if __name__ == "__main__":
    main()
