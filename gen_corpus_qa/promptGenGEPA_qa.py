#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation GEPA du prompt de génération QA (généralise gen_corpus/promptGenGEPA.py)."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy.clients import configure_cache as dspy_configure_cache

try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None

REFUS_MARQUEURS = ("ne figure pas", "ne précise pas", "n'est pas dans", "pas dans le document",
                   "aucune information", "ne mentionne pas", "ne contient pas", "n'indique pas")


def _ressemble_refus(answer: str) -> bool:
    a = answer.replace("'", "'").lower()
    return any(m in a for m in REFUS_MARQUEURS)


def note_paire(judge_score: int, type_question: str, answer: str) -> float:
    """Note 0..1 d'une paire générée. judge_score est un entier 1..5.
    Pour hors_document, un refus est exigé : sinon la note est plafonnée à 0.2."""
    base = max(1, min(5, int(judge_score))) / 5.0
    if type_question == "hors_document" and not _ressemble_refus(answer):
        return 0.2
    return base


@dataclass
class Config:
    seed_path: str = "seed.json"
    domaines_path: str = "domaines.json"
    prompt_path: str = "prompt-init.txt"
    gepa_prompt_path: str = "GEPAPrompt.txt"
    generator_model: str = "openai/deepseek-chat"
    eval_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    reflection_temperature: float = 1.0
    max_metric_calls: int = 200


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="seed.json")
    p.add_argument("--domaines", default="domaines.json")
    p.add_argument("--prompt", default="prompt-init.txt")
    p.add_argument("--gepa-prompt", default="GEPAPrompt.txt")
    p.add_argument("--generator-model", default="openai/deepseek-chat")
    p.add_argument("--eval-model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-metric-calls", type=int, default=200)
    a = p.parse_args(argv)
    return Config(seed_path=a.seed, domaines_path=a.domaines, prompt_path=a.prompt,
                  gepa_prompt_path=a.gepa_prompt, generator_model=a.generator_model,
                  eval_model=a.eval_model, deepseek_api_base=a.deepseek_api_base,
                  reflection_temperature=a.temperature, max_metric_calls=a.max_metric_calls)


def _ensure_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    return key


def configure_lm(model, temperature, api_base) -> dspy.LM:
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    lm = dspy.LM(model=model, api_key=_ensure_key(), api_base=base, model_type="chat",
                 temperature=temperature, max_tokens=512, cache=False)
    dspy.settings.configure(lm=lm)
    return lm


def configure_eval_lm(model, api_base) -> dspy.LM:
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    return dspy.LM(model=model, api_key=_ensure_key(), api_base=base, model_type="chat",
                   temperature=0.0, max_tokens=16, cache=False)


def build_example_generator(prompt_text: str, documents: dict) -> dspy.Module:
    class GenerateQA(dspy.Signature):
        __doc__ = prompt_text
        document:      str = dspy.InputField(desc="Texte du document de référence")
        type_question: str = dspy.InputField(desc="factuelle, reformulation ou hors_document")
        registre:      str = dspy.InputField(desc="Registre de la question")
        question: str = dspy.OutputField(desc="Question utilisateur en français")
        answer:   str = dspy.OutputField(desc="Réponse ancrée, ou refus si hors_document")

    class QAGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(GenerateQA)

        def forward(self, document_id, type_question, registre):
            doc = documents[document_id]
            r = self.generate(document=doc, type_question=type_question, registre=registre)
            return {"question": r.question, "answer": r.answer,
                    "type_question": type_question, "document_id": document_id}

    return QAGenerator()


def build_trainset(seed: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(document_id=ex["document_id"], type_question=ex["type_question"],
                     registre=ex["registre"], question=ex["question"], answer=ex["answer"]
                     ).with_inputs("document_id", "type_question", "registre")
        for ex in seed
    ]


class EvalQualite(dspy.Signature):
    """La réponse est-elle exacte, concise et entièrement fondée sur le document ?
    Répondre avec un entier 1..5 uniquement, sans commentaire."""
    document: str = dspy.InputField()
    question: str = dspy.InputField()
    answer:   str = dspy.InputField()
    score:    int = dspy.OutputField(desc="Entier 1..5")


def make_metric(eval_lm: dspy.LM, documents: dict):
    judge = dspy.Predict(EvalQualite)
    counter = {"n": 0}

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        p = pred or {}
        answer = str(p.get("answer") or "")
        if not answer:
            return 0.0
        doc = documents[p["document_id"]]
        with dspy.settings.context(lm=eval_lm):
            out = judge(document=doc, question=str(p.get("question") or ""), answer=answer)
        try:
            js = int(out.score)
        except Exception:
            js = 3
        s = note_paire(js, p.get("type_question", ""), answer)
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"[métrique] appel {counter['n']} note={s:.2f}")
        return s

    return metric


def _extract_best_prompt(compiled, teleprompter, initial: str) -> str:
    candidates = []
    for attr in ("best_prompt", "best_prompt_str", "best_prompt_text"):
        v = getattr(teleprompter, attr, None)
        if v:
            candidates.append(str(v))
    bp = getattr(teleprompter, "best_prompts", None)
    if isinstance(bp, dict):
        candidates.extend(str(v) for v in bp.values() if v)
    sig = getattr(getattr(compiled, "generate", None), "signature", None)
    if sig:
        instr = getattr(sig, "instructions", None) or getattr(sig, "__doc__", None)
        if instr:
            candidates.append(str(instr))
    candidates.append(initial)
    for c in candidates:
        if c and len(c.strip()) > 40 and "given the fields" not in c.lower():
            return c
    return initial


def main(argv=None) -> int:
    cfg = parse_args(argv)
    here = Path(cfg.domaines_path).resolve().parent
    domaines = json.loads(Path(cfg.domaines_path).read_text(encoding="utf-8"))
    documents = {d["domaine"]: (here / d["document"]).read_text(encoding="utf-8") for d in domaines}
    initial_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")
    seed = json.loads(Path(cfg.seed_path).read_text(encoding="utf-8"))
    configure_lm(cfg.generator_model, cfg.reflection_temperature, cfg.deepseek_api_base)
    eval_lm = configure_eval_lm(cfg.eval_model, cfg.deepseek_api_base)
    generator = build_example_generator(initial_prompt, documents)
    trainset = build_trainset(seed)
    teleprompter = dspy.GEPA(
        metric=make_metric(eval_lm, documents),
        reflection_lm=dspy.settings.lm,
        max_metric_calls=cfg.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
    )
    compiled = teleprompter.compile(generator, trainset=trainset)
    best_prompt = _extract_best_prompt(compiled, teleprompter, initial_prompt)
    Path(cfg.gepa_prompt_path).write_text(best_prompt, encoding="utf-8")
    print(f"Prompt optimisé sauvegardé dans {cfg.gepa_prompt_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(filename="gepa_qa_llm_calls.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    raise SystemExit(main())
