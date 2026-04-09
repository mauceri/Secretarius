#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert-legacy_flat.py

Convertit un JSONL "legacy" (avec champs: contenu, expressions_clefs, url, date, etc.)
en un JSONL "flat" où l'on conserve seulement :
  - source
  - titre (null)
  - contenu
  - meta {dataset, lang, format}

Exemple entrée (1 ligne):
{"contenu": "...", "expressions_clefs": [...], "url": "...", "date": "...", ...}

Exemple sortie:
{"source":"legacy","titre":null,"contenu":"...","meta":{"dataset":"legacy","lang":"fr","format":"legacy_messages"}}

Usage :
  python convert-legacy_flat.py --input old.jsonl --output new.jsonl

Options :
  --source legacy
  --dataset legacy
  --lang fr
  --format legacy_messages
  --keep-fields        (optionnel) conserve aussi les champs legacy supplémentaires
"""

import argparse
import json
import sys
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Fichier JSONL legacy en entrée")
    ap.add_argument("--output", required=True, help="Fichier JSONL converti en sortie")
    ap.add_argument("--source", default="legacy", help="Valeur du champ 'source'")
    ap.add_argument("--dataset", default="legacy", help="meta.dataset")
    ap.add_argument("--lang", default="fr", help="meta.lang")
    ap.add_argument("--format", default="legacy_messages", help="meta.format")
    ap.add_argument("--keep-fields", action="store_true",
                    help="Conserver aussi les autres champs legacy (url/date/etc.) au niveau racine")
    args = ap.parse_args()

    n_in = n_out = n_skipped = 0

    with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            n_in += 1
            contenu = rec.get("contenu", "")
            if contenu is None:
                contenu = ""
            contenu = str(contenu)

            out: Dict[str, Any] = {
                "source": args.source,
                "titre": None,
                "contenu": contenu,
                "meta": {
                    "dataset": args.dataset,
                    "lang": args.lang,
                    "format": args.format,
                },
            }

            if args.keep_fields:
                # Copie des champs legacy (sauf contenu déjà pris)
                for k, v in rec.items():
                    if k == "contenu":
                        continue
                    if k in out:
                        out[f"legacy_{k}"] = v
                    else:
                        out[k] = v

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"OK: in={n_in} out={n_out} skipped={n_skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
