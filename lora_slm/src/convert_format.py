#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertit un JSONL "legacy" vers le format "chunks" attendu par les scripts
d'entraînement et d'évaluation.

Format legacy (entrée) :
    {"contenu": "...", "expressions_clefs": [...], "url": "...", ...}

Format chunks (sortie) :
    {"source": "legacy", "titre": null,
     "chunks": [{"chunk": "...", "expressions_caracteristiques": [...]}],
     "meta": {"dataset": "legacy", "lang": "fr", "format": "legacy_messages"}}

Exemple :
    python src/convert_format.py --input old.jsonl --output new.jsonl
    python src/convert_format.py --input old.jsonl --output new.jsonl --keep-fields
"""

import argparse
import json
import sys
from typing import Any, Dict, List


def coerce_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x if e is not None and str(e).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def parse_args():
    p = argparse.ArgumentParser(
        description="Conversion format JSONL legacy → format chunks."
    )
    p.add_argument("--input",       required=True,  help="JSONL legacy en entrée.")
    p.add_argument("--output",      required=True,  help="JSONL converti en sortie.")
    p.add_argument("--keep-fields", action="store_true",
                   help="Copier également les champs legacy (url, date, etc.) à la racine.")
    p.add_argument("--source",  default="legacy", help="Valeur du champ 'source'.")
    p.add_argument("--dataset", default="legacy", help="Valeur de meta.dataset.")
    p.add_argument("--lang",    default="fr",     help="Valeur de meta.lang.")
    p.add_argument("--format",  default="legacy_messages", help="Valeur de meta.format.")
    return p.parse_args()


def main():
    args     = parse_args()
    n_in = n_out = n_skipped = 0

    with (
        open(args.input,  "r", encoding="utf-8") as f_in,
        open(args.output, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            n_in += 1
            contenu = str(rec.get("contenu") or "").strip()

            # Accepte les deux noms possibles pour les expressions
            expr = rec.get("expressions_caracteristiques") or rec.get("expressions_clefs")
            expressions = coerce_list_str(expr)

            out: Dict[str, Any] = {
                "source": args.source,
                "titre":  None,
                "chunks": [{"chunk": contenu, "expressions_caracteristiques": expressions}],
                "meta":   {"dataset": args.dataset, "lang": args.lang, "format": args.format},
            }

            if args.keep_fields:
                skip = {"contenu", "expressions_clefs", "expressions_caracteristiques"}
                for k, v in rec.items():
                    if k in skip:
                        continue
                    out[f"legacy_{k}" if k in out else k] = v

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"OK : in={n_in}  out={n_out}  skipped={n_skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
