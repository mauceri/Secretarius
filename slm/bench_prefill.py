#!/usr/bin/env python3
"""Mesure le TTFT (time-to-first-token) de phi-4-mini sur deux prompts système.

Usage : python3 bench_prefill.py
Prérequis : service slm-llama-cpp actif (port 8998), workspaces
  ~/.openclaw/workspace/ (prod) et ~/.openclaw-slm/workspace/ (léger).
"""
import json
import os
import statistics
import time

import requests

LLAMA_URL = "http://127.0.0.1:8998/v1/chat/completions"
MODEL = "phi-4-mini-instruct"
PROD_WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
SLM_WORKSPACE = os.path.expanduser("~/.openclaw-slm/workspace")


def load_workspace(path: str) -> str:
    """Concatène tous les fichiers .md d'un workspace en un prompt système."""
    parts = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".md"):
            with open(os.path.join(path, name)) as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


def measure_ttft(system_prompt: str, n: int = 3) -> float:
    """Retourne le TTFT médian (secondes) sur n appels streaming."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Bonjour."},
        ],
        "max_tokens": 10,
        "stream": True,
    }
    times = []
    for _ in range(n):
        start = time.perf_counter()
        with requests.post(LLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                raw = line.decode().removeprefix("data: ")
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        times.append(time.perf_counter() - start)
                        break
                except json.JSONDecodeError:
                    continue
    return statistics.median(times)


if __name__ == "__main__":
    print("Chargement des workspaces...")
    prod_prompt = load_workspace(PROD_WORKSPACE)
    slm_prompt = load_workspace(SLM_WORKSPACE)
    prod_words = len(prod_prompt.split())
    slm_words = len(slm_prompt.split())
    print(f"  Prompt prod  : {prod_words} mots")
    print(f"  Prompt léger : {slm_words} mots  (ratio {prod_words/slm_words:.1f}x)")
    print()
    print("Mesure prod (3 appels, patience ~2 min)...")
    prod_ttft = measure_ttft(prod_prompt)
    print(f"  TTFT prod  (médiane) : {prod_ttft:.2f}s")
    print("Mesure léger (3 appels)...")
    slm_ttft = measure_ttft(slm_prompt)
    print(f"  TTFT léger (médiane) : {slm_ttft:.2f}s")
    if slm_ttft > 0:
        print(f"\n  Gain : {prod_ttft / slm_ttft:.1f}x")
