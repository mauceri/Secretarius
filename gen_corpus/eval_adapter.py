#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Évalue un adaptateur LoRA unique sur corpus_lora_eval.jsonl : lance
llama-server avec l'adaptateur donné, interroge chaque exemple, compare la
commande extraite à la vérité terrain."""
import argparse, json, subprocess, sys, time, urllib.request

EVAL_PATH = "/home/mauceric/Secretarius/gen_corpus/corpus_lora_eval.jsonl"
SYSTEM_ROUTE = ('Routeur de commandes Tiron. Pour chaque message, répondre '
                'uniquement avec un objet JSON : {"command": "/commande" ou '
                'null, "args": "arguments bruts ou chaîne vide"}.')


def call_llm(base_url, msg, max_tokens=60):
    body = {"messages": [{"role": "system", "content": SYSTEM_ROUTE},
                         {"role": "user", "content": msg}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d["choices"][0]["message"]["content"].strip()


def wait_ready(base_url, timeout_s=60):
    for _ in range(timeout_s):
        try:
            urllib.request.urlopen(base_url + "/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server non prêt après {}s".format(timeout_s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8996")
    p.add_argument("--check", action="store_true",
                   help="Vérifie seulement le parsing JSON sur 3 exemples (fumée).")
    args = p.parse_args()

    wait_ready(args.base_url)
    rows = [json.loads(l) for l in open(EVAL_PATH) if l.strip()]
    if args.check:
        rows = rows[:3]

    ok = 0
    for r in rows:
        msg = r["messages"][-2]["content"]
        expected = json.loads(r["messages"][-1]["content"]).get("command")
        out = call_llm(args.base_url, msg)
        try:
            got = json.loads(out).get("command")
        except Exception:
            got = "<JSON invalide>"
        good = got == expected
        ok += good
        if not good:
            print(f"!! {msg!r} attendu={expected!r} obtenu={got!r}")

    print(f"=== {ok}/{len(rows)} corrects ({100*ok/len(rows):.1f}%) ===")
    if not args.check and ok / len(rows) < 0.90:
        sys.exit(1)


if __name__ == "__main__":
    main()
