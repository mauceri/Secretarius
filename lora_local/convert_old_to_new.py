#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert-old_to-new.py

Transforme un JSONL "legacy" (une ligne = un objet) vers le format "chunks" attendu
par vos scripts récents.

Entrée (JSONL), exemple :
{"contenu": "...", "expressions_clefs": [...], "url": "...", "date": "...", "categorie": "...", "theme": "...", "type_du_document": "..."}

Sortie (JSONL), exemple :
{
  "source": "legacy",
  "titre": null,
  "chunks": [
    {"chunk": "...", "expressions_caracteristiques": [...]}
  ],
  "meta": {"dataset":"legacy","lang":"fr","format":"legacy_messages"}
  ... (optionnel: copie des champs legacy)
}

Usage :
  python convert-old_to-new.py --input old.jsonl --output new.jsonl
Options :
  --keep-fields          Copie aussi les champs legacy (url, date, etc.) au niveau racine
  --source legacy        Valeur du champ "source"
  --dataset legacy       Valeur meta.dataset
  --lang fr              Valeur meta.lang
  --format legacy_messages Valeur meta.format
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional, List


def coerce_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x if e is not None and str(e).strip() != ""]
    # tolère une chaîne unique
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Fichier JSONL legacy en entrée")
    p.add_argument("--output", required=True, help="Fichier JSONL converti en sortie")
    p.add_argument("--keep-fields", action="store_true",
                   help="Copier aussi les champs legacy (url/date/categorie/theme/type_du_document, etc.)")
    p.add_argument("--source", default="legacy", help="Valeur pour le champ 'source'")
    p.add_argument("--dataset", default="legacy", help="meta.dataset")
    p.add_argument("--lang", default="fr", help="meta.lang")
    p.add_argument("--format", default="legacy_messages", help="meta.format")
    args = p.parse_args()

    n_in = 0
    n_out = 0
    n_skipped = 0

    with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            n_in += 1

            contenu = rec.get("contenu", "")
            if contenu is None:
                contenu = ""
            contenu = str(contenu).strip()

            # Tolérer plusieurs noms possibles (au cas où)
            expr = rec.get("expressions_caracteristiques")
            if expr is None:
                expr = rec.get("expressions_clefs")
            expressions = coerce_list_str(expr)

            out: Dict[str, Any] = {
                "source": args.source,
                "titre": None,
                "chunks": [
                    {
                        "chunk": contenu,
                        "expressions_caracteristiques": expressions,
                    }
                ],
                "meta": {
                    "dataset": args.dataset,
                    "lang": args.lang,
                    "format": args.format,
                },
            }

            if args.keep_fields:
                # Copie des champs legacy au niveau racine (sauf contenu/expressions déjà transformés)
                for k, v in rec.items():
                    if k in ("contenu", "expressions_clefs", "expressions_caracteristiques"):
                        continue
                    # évite collision
                    if k in out:
                        out[f"legacy_{k}"] = v
                    else:
                        out[k] = v

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"OK: in={n_in} out={n_out} skipped={n_skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
