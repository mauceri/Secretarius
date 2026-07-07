#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère corpus_qa.jsonl (triplets document/question/réponse) via DeepSeek.

Généralisation de gen_corpus/generate_corpus.py : pour chaque domaine, charge
le document associé et demande au modèle une paire (question, réponse ancrée)
selon un type de question et un registre tirés au hasard.
"""
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
    count: int = 1500
    batch_size: int = 50
    report_every: int = 50
    prompt_path: str = "GEPAPrompt.txt"
    prompt_fallback: str = "prompt-init.txt"
    domaines_path: str = "domaines.json"
    registres_path: str = "registres.json"
    output: str = "corpus_qa.jsonl"
    generator_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    temperature: float = 0.9


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--prompt", default="GEPAPrompt.txt")
    p.add_argument("--domaines", default="domaines.json")
    p.add_argument("--registres", default="registres.json")
    p.add_argument("--output", default="corpus_qa.jsonl")
    p.add_argument("--model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=0.9)
    a = p.parse_args(argv)
    return Config(count=a.count, batch_size=a.batch_size, report_every=a.report_every,
                  prompt_path=a.prompt, domaines_path=a.domaines, registres_path=a.registres,
                  output=a.output, generator_model=a.model,
                  deepseek_api_base=a.deepseek_api_base, temperature=a.temperature)


def _build_signature(prompt_text: str):
    class GenerateQA(dspy.Signature):
        __doc__ = prompt_text
        document:      str = dspy.InputField(desc="Texte du document de référence")
        type_question: str = dspy.InputField(desc="factuelle, reformulation ou hors_document")
        registre:      str = dspy.InputField(desc="Registre de la question")
        question: str = dspy.OutputField(desc="Question utilisateur en français")
        answer:   str = dspy.OutputField(desc="Réponse ancrée dans le document, ou refus si hors_document")
    return GenerateQA


def build_entry(result, document_id: str, document: str, type_question: str, registre: str) -> dict:
    return {"document_id": document_id, "document": document,
            "question": result.question.strip(), "answer": result.answer.strip(),
            "type_question": type_question, "registre": registre}


def _load_documents(domaines: list[dict], base: Path) -> dict:
    return {d["domaine"]: (base / d["document"]).read_text(encoding="utf-8") for d in domaines}


def main(argv=None) -> int:
    cfg = parse_args(argv)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    base_url = os.getenv("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
    lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base_url,
                 model_type="chat", temperature=cfg.temperature, max_tokens=512, cache=False)
    dspy.settings.configure(lm=lm)

    here = Path(cfg.domaines_path).resolve().parent
    prompt_p = Path(cfg.prompt_path)
    prompt_text = (prompt_p if prompt_p.exists() else Path(cfg.prompt_fallback)).read_text(encoding="utf-8")
    predict = dspy.Predict(_build_signature(prompt_text))
    domaines = json.loads(Path(cfg.domaines_path).read_text(encoding="utf-8"))
    registres = json.loads(Path(cfg.registres_path).read_text(encoding="utf-8"))
    documents = _load_documents(domaines, here)

    buffer = []
    stime = time.time()
    consecutive_errors = 0
    max_consecutive = max(10, cfg.count // 10)
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in range(cfg.count):
            dom = random.choice(domaines)
            doc_text = documents[dom["domaine"]]
            tq = random.choice(dom["types_question"])
            reg = random.choice(registres)
            try:
                result = predict(document=doc_text, type_question=tq, registre=reg)
                buffer.append(build_entry(result, dom["domaine"], doc_text, tq, reg))
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
