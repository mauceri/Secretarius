#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère corpus.jsonl à partir du prompt optimisé par GEPA."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy.clients import configure_cache as dspy_configure_cache

try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None


@dataclass
class Config:
    count: int = 1000
    batch_size: int = 50
    report_every: int = 50
    prompt_path: str = "GEPAPrompt.txt"
    prompt_fallback: str = "prompt-init.txt"
    intentions_path: str = "intentions.json"
    registres_path: str = "registres.json"
    output: str = "corpus.jsonl"
    generator_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    temperature: float = 0.9


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--prompt", default="GEPAPrompt.txt")
    p.add_argument("--intentions", default="intentions.json")
    p.add_argument("--registres", default="registres.json")
    p.add_argument("--output", default="corpus.jsonl")
    p.add_argument("--model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=0.9)
    a = p.parse_args(argv)
    return Config(count=a.count, batch_size=a.batch_size, report_every=a.report_every,
                  prompt_path=a.prompt, intentions_path=a.intentions, registres_path=a.registres,
                  output=a.output, generator_model=a.model,
                  deepseek_api_base=a.deepseek_api_base, temperature=a.temperature)


def _build_signature(prompt_text: str):
    class GenerateExample(dspy.Signature):
        __doc__ = prompt_text
        intention: str = dspy.InputField(desc="Intention Tiron à illustrer")
        registre:  str = dspy.InputField(desc="Registre du message")
        variante:  str = dspy.InputField(desc="Type de variante")
        text:    str = dspy.OutputField(desc="Message utilisateur réaliste en français")
        command: str = dspy.OutputField(desc="Commande Tiron ou null")
        args:    str = dspy.OutputField(desc="Arguments bruts (chaîne vide si sans args)")
    return GenerateExample


def generate_one(predict, intention: str, registre: str, variante: str) -> dict:
    result = predict(intention=intention, registre=registre, variante=variante)
    cmd = result.command if result.command and result.command.lower() not in ("null", "none") else None
    args = result.args.strip()
    if args in ('""', "''"):
        args = ""
    return {"text": result.text, "intention": intention, "registre": registre,
            "variante": variante, "action": {"command": cmd, "args": args}}


def main(argv=None) -> int:
    cfg = parse_args(argv)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    base = os.getenv("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
    lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base,
                 model_type="chat", temperature=cfg.temperature, max_tokens=256, cache=False)
    dspy.settings.configure(lm=lm)

    prompt_p = Path(cfg.prompt_path)
    prompt_text = (prompt_p if prompt_p.exists() else Path(cfg.prompt_fallback)).read_text(encoding="utf-8")
    predict = dspy.Predict(_build_signature(prompt_text))
    intentions = json.loads(Path(cfg.intentions_path).read_text(encoding="utf-8"))
    registres = json.loads(Path(cfg.registres_path).read_text(encoding="utf-8"))

    buffer = []
    stime = time.time()
    consecutive_errors = 0
    max_consecutive = max(10, cfg.count // 10)
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in range(cfg.count):
            obj = random.choice(intentions)
            try:
                entry = generate_one(predict, obj["intention"], random.choice(registres),
                                     random.choice(obj["variantes"]))
                buffer.append(entry)
                consecutive_errors = 0
            except Exception as e:
                print(f"[{i+1}] Erreur: {e}", flush=True)
                consecutive_errors += 1
                if consecutive_errors > max_consecutive:
                    raise RuntimeError(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt.") from e
                continue
            if len(buffer) >= cfg.batch_size:
                for e in buffer:
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                buffer = []
            if (i + 1) % cfg.report_every == 0:
                print(f"[{i+1}/{cfg.count}] {time.time()-stime:.1f}s", flush=True)
        for e in buffer:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Corpus sauvegardé dans {cfg.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
