#!/usr/bin/env python3
"""
Constitue un corpus de référence en faisant tourner le wiki LoRA actuel
(llama.cpp port 8989) sur les textes du corpus Wikipedia.

Chaque ligne du corpus de sortie :
    {"text": "...", "expressions_lora": ["...", ...]}

Usage :
    source llenv/bin/activate
    python build_lora_corpus.py [--corpus data/corpus_wiki40b_fr_indexed_100.jsonl]
                                [--n 100] [--out data/corpus_lora_reference.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from urllib import request as urlrequest, error as urlerror
import re

LLAMA_URL   = "http://127.0.0.1:8989/v1/chat/completions"
LLAMA_MODEL = "local-llama-cpp"

SYSTEM_PROMPT = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
USER_PREFIX = "Quelles sont les expressions clés contenues à l'identique dans ce texte :\n"


def load_texts(jsonl_path: str, n: int, seed: int = 42) -> list[str]:
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for chunk in doc.get("chunks", []):
                text = chunk.get("chunk", "").strip()
                if text:
                    texts.append(text)
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:n]


def lora_extract(text: str, timeout: float = 120.0) -> list[str] | None:
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PREFIX + text},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urlrequest.Request(
        LLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data    = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return _parse_json_list(content)
    except (urlerror.URLError, TimeoutError) as exc:
        print(f"\n  [ERREUR réseau] {exc}")
        return None
    except Exception as exc:
        print(f"\n  [ERREUR] {exc}")
        return None


def _parse_json_list(text: str) -> list[str]:
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [x for x in result if isinstance(x, str)]
    except Exception:
        pass
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return [x for x in result if isinstance(x, str)]
        except Exception:
            pass
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus_wiki40b_fr_indexed_100.jsonl")
    parser.add_argument("--n",      type=int, default=100)
    parser.add_argument("--out",    default="data/corpus_lora_reference.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    print(f"Chargement de {args.n} textes depuis {args.corpus}…")
    texts = load_texts(args.corpus, args.n, args.seed)
    print(f"  {len(texts)} textes chargés.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    errors = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, text in enumerate(texts):
            print(f"  [{i+1}/{len(texts)}] {len(text)} chars… ", end="", flush=True)
            t0 = time.time()
            expressions = lora_extract(text)
            elapsed = time.time() - t0

            if expressions is None:
                print(f"ERREUR ({elapsed:.1f}s)")
                errors += 1
                continue

            print(f"{len(expressions)} expressions ({elapsed:.1f}s)")
            record = {
                "text": text,
                "expressions_lora": expressions,
                "meta": {"elapsed_s": round(elapsed, 2), "n_expressions": len(expressions)},
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ok += 1

    print(f"\nTerminé : {ok} exemples sauvegardés, {errors} erreurs.")
    print(f"Corpus de référence : {out_path}")


if __name__ == "__main__":
    main()
