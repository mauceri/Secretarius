#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Évaluation A/B d'un adaptateur QA : compare phi-4 nu vs phi-4+adaptateur.

Deux backends d'inférence :
  --backend llama  : interroge un llama-server (--base-url) ; le mode "nu" vs
                     "adapté" est réglé par le scale de l'adaptateur via
                     POST /lora-adapters (scale 0 = nu, scale 1 = adapté).
  --backend peft   : charge phi-4 + adaptateur PEFT en Python ; le mode "nu"
                     désactive l'adaptateur (model.disable_adapter()).
Le juge DeepSeek note chaque réponse candidate 1..5 (exactitude + ancrage).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT_QA = (
    "Vous êtes Tiron. Répondez à la question en vous appuyant uniquement sur le "
    "document fourni. Soyez concis et répondez en français. Si la réponse ne figure "
    "pas dans le document, indiquez-le clairement sans rien inventer."
)
REFUS_MARQUEURS = ("ne figure pas", "ne précise pas", "n'est pas dans", "pas dans le document",
                   "aucune information", "ne mentionne pas", "ne contient pas", "n'indique pas")


def _ressemble_refus(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in REFUS_MARQUEURS)


def parse_eval_row(row: dict) -> dict:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    ref = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    doc = user.split("Document:\n", 1)[1].split("\n\nQuestion: ", 1)[0]
    question = user.split("\n\nQuestion: ", 1)[1]
    return {"document": doc, "question": question, "reference": ref,
            "is_refus": _ressemble_refus(ref)}


def aggregate(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


# ---- backend llama-server -------------------------------------------------

def _http_json(url: str, body: dict, timeout=60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def set_lora_scale(base_url: str, scale: float) -> None:
    # id 0 = l'unique adaptateur chargé via --lora
    _http_json(base_url + "/lora-adapters", [{"id": 0, "scale": scale}])


def infer_llama(base_url: str, document: str, question: str) -> str:
    body = {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT_QA},
        {"role": "user", "content": f"Document:\n{document}\n\nQuestion: {question}"}],
        "max_tokens": 200, "temperature": 0}
    d = _http_json(base_url + "/v1/chat/completions", body)
    return d["choices"][0]["message"]["content"].strip()


# ---- backend PEFT (Jupyter/CLI) -------------------------------------------

def load_peft(model_path: str, adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tok


def infer_peft(model, tok, document: str, question: str, use_adapter: bool) -> str:
    import torch
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_QA},
            {"role": "user", "content": f"Document:\n{document}\n\nQuestion: {question}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ctx = model.disable_adapter() if not use_adapter else _nullctx()
    with torch.no_grad(), ctx:
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ---- juge DeepSeek --------------------------------------------------------

def judge_score(document: str, question: str, answer: str) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY")
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    prompt = (f"Document:\n{document}\n\nQuestion: {question}\n\nRéponse: {answer}\n\n"
              "La réponse est-elle exacte, concise et entièrement fondée sur le document "
              "(refus correct si l'information est absente) ? Répondez par un entier 1 à 5 uniquement.")
    body = {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    txt = d["choices"][0]["message"]["content"].strip()
    digits = "".join(c for c in txt if c.isdigit())
    return max(1, min(5, int(digits))) if digits else 3


def run_condition(rows, infer_fn) -> dict:
    scores = []
    for r in rows:
        p = parse_eval_row(r)
        answer = infer_fn(p["document"], p["question"])
        js = judge_score(p["document"], p["question"], answer)
        scores.append(js / 5.0)
    return {"note_moyenne": aggregate(scores), "n": len(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="corpus_qa_eval.jsonl")
    ap.add_argument("--backend", choices=["llama", "peft"], default="llama")
    ap.add_argument("--base-url", default="http://127.0.0.1:8996")
    ap.add_argument("--model-path", default="/home/mauceric/Modèles/phi4")
    ap.add_argument("--adapter-path", default="/home/mauceric/lora_slm/checkpoints/qa-document")
    ap.add_argument("--limit", type=int, default=0, help="0 = tout le jeu d'éval")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    if args.backend == "llama":
        set_lora_scale(args.base_url, 0.0)
        nu = run_condition(rows, lambda d, q: infer_llama(args.base_url, d, q))
        set_lora_scale(args.base_url, 1.0)
        ad = run_condition(rows, lambda d, q: infer_llama(args.base_url, d, q))
    else:
        model, tok = load_peft(args.model_path, args.adapter_path)
        nu = run_condition(rows, lambda d, q: infer_peft(model, tok, d, q, use_adapter=False))
        ad = run_condition(rows, lambda d, q: infer_peft(model, tok, d, q, use_adapter=True))

    print(f"=== NU       : {nu['note_moyenne']:.3f} (n={nu['n']}) ===")
    print(f"=== ADAPTÉ   : {ad['note_moyenne']:.3f} (n={ad['n']}) ===")
    print(f"=== DELTA    : {ad['note_moyenne']-nu['note_moyenne']:+.3f} ===")


if __name__ == "__main__":
    main()
