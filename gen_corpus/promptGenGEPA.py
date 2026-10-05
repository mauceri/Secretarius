#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation de prompt via GEPA pour le corpus d'intentions Tiron."""
from __future__ import annotations

import argparse
import random
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy.clients import configure_cache as dspy_configure_cache

from chaine import (EvalFidelite, EvalRealisme, INTENTION_DESCRIPTIONS, build_signature,
                    tirer_contraintes, tirer_demos)

try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None

COMMANDS_KNOWN = {
    "/c", "/ingest", "/wikistatus", "/q", "/source", "/r", "/tags", "/kbupdate",
    "/supprimer", "/relire", "/verifie", "/chercher", "/connecter", "/inbox",
    "/drive", "/repondre", "/lire", "/cron",
}
REQUIRES_ARGS = {
    "wiki_capture", "wiki_query", "source_read", "gog_search", "gog_reply",
    "gog_drive", "wiki_search", "gog_get", "wiki_delete", "wiki_verify",
    "cron_schedule",
}


def _is_null_command(cmd: str) -> bool:
    return not cmd or cmd.strip().lower() in ("null", "none", "")


def structural_score(pred: dict, gold: dict) -> float:
    command = str(pred.get("command") or "").strip()
    args = str(pred.get("args") or "").strip()
    intention = str(gold.get("intention") or "").strip()
    cmd_ok = _is_null_command(command) if intention == "out_of_scope" else command in COMMANDS_KNOWN
    args_ok = bool(args) if intention in REQUIRES_ARGS else True
    return 0.5 * int(cmd_ok) + 0.5 * int(args_ok)


@dataclass
class Config:
    seed_path: str = "seed.json"
    prompt_path: str = "prompt-init.txt"
    gepa_prompt_path: str = "GEPAPrompt.txt"
    generator_model: str = "openai/deepseek-chat"
    eval_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    reflection_temperature: float = 1.0
    student_temperature: float = 0.7
    program_path: str = "gepa_program.json"
    max_metric_calls: int = 300
    valset_size: int = 40


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="seed.json")
    p.add_argument("--prompt", default="prompt-init.txt")
    p.add_argument("--gepa-prompt", default="GEPAPrompt.txt")
    p.add_argument("--generator-model", default="openai/deepseek-chat")
    p.add_argument("--eval-model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-metric-calls", type=int, default=300)
    p.add_argument("--valset-size", type=int, default=40)
    p.add_argument("--student-temperature", type=float, default=0.7)
    p.add_argument("--program", default="gepa_program.json")
    a = p.parse_args(argv)
    return Config(seed_path=a.seed, prompt_path=a.prompt, gepa_prompt_path=a.gepa_prompt,
                  generator_model=a.generator_model, eval_model=a.eval_model,
                  deepseek_api_base=a.deepseek_api_base,
                  reflection_temperature=a.temperature, max_metric_calls=a.max_metric_calls,
                  valset_size=a.valset_size, student_temperature=a.student_temperature,
                  program_path=a.program)


def _ensure_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    return key


def make_lm(model: str, temperature: float, api_base: str, max_tokens: int = 512) -> dspy.LM:
    key = _ensure_key()
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    return dspy.LM(model=model, api_key=key, api_base=base, model_type="chat",
                   temperature=temperature, max_tokens=max_tokens, cache=False)


def build_example_generator(prompt_text: str, seed: list[dict], rng: random.Random) -> dspy.Module:
    class ExampleGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(build_signature(prompt_text))

        def forward(self, intention, registre, variante, contraintes):
            # Exemples de départ de la même intention montrés au générateur (D1),
            # tirés à chaque appel ; les contraintes imposent la diversité (D4).
            r = self.generate(intention=intention, registre=registre, variante=variante,
                              contraintes=contraintes, demos=tirer_demos(seed, intention, 3, rng))
            return {"text": r.text, "intention": intention, "command": r.command,
                    "args": r.args, "registre": registre, "variante": variante}

    return ExampleGenerator()


def split_trainset(examples: list, n_val: int, seed: int = 0) -> tuple[list, list]:
    """Valset séparé et petit : GEPA dépense sinon son budget max_metric_calls à
    réévaluer tout le trainset au lieu de réfléchir (piste du 2026-09-30)."""
    n_val = min(n_val, len(examples) // 2)
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:n_val])
    train = [e for i, e in enumerate(examples) if i not in val_idx]
    val = [e for i, e in enumerate(examples) if i in val_idx]
    return train, val


def build_trainset(seed: list[dict], rng: random.Random | None = None) -> list[dspy.Example]:
    rng = rng or random.Random(0)
    return [
        dspy.Example(
            intention=ex["intention"],
            registre=ex.get("registre", "poli"),
            variante=ex.get("variante", "sans_args"),
            contraintes=tirer_contraintes(ex["intention"], rng),
            text=ex["text"],  # référence conservée (piste : juge de ressemblance)
            command=ex["action"]["command"] if ex["action"]["command"] is not None else "null",
            args=ex["action"]["args"],
        ).with_inputs("intention", "registre", "variante", "contraintes")
        for ex in seed
    ]


def construire_feedback(text: str, s_struct: float, s_real: float, raison_real: str,
                        s_fidele: float, raison_fidele: str) -> str:
    """Retour écrit pour la réflexion de GEPA (D2) : sans lui, GEPA ne voit
    qu'un nombre et ne sait pas quoi corriger dans l'instruction."""
    remarques = []
    if s_fidele < 1:
        remarques.append(f"Infidèle à l'intention : {raison_fidele}")
    if s_real < 0.8:
        remarques.append(f"Peu réaliste ({s_real * 5:.0f}/5) : {raison_real}")
    if s_struct < 1:
        remarques.append("Commande ou arguments manquants ou inconnus.")
    n = len(text.split())
    if n > 20:
        remarques.append(f"Trop long : {n} mots (viser moins de 20).")
    if text.lower().startswith(("bonjour", "tiron")):
        remarques.append("Commence par « Bonjour » ou « Tiron » : varier l'attaque.")
    return "Parfait." if not remarques else " ".join(remarques)


def make_metric(eval_lm: dspy.LM):
    real_pred = dspy.Predict(EvalRealisme)
    fid_pred = dspy.Predict(EvalFidelite)
    counter = {"n": 0}

    def juger(pred, defaut: int, borne: tuple[int, int], **kw) -> tuple[int, str]:
        try:
            with dspy.settings.context(lm=eval_lm):
                out = pred(**kw)
        except Exception:
            return defaut, "réponse du juge illisible"
        try:
            return max(borne[0], min(borne[1], int(out.score))), str(getattr(out, "raison", "") or "")
        except (TypeError, ValueError):
            return defaut, "réponse du juge illisible"

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        text = str((pred or {}).get("text") or "")
        if not text:
            return dspy.Prediction(score=0.0, feedback="Aucun texte produit.")
        s_struct = structural_score(pred, gold)
        note, raison_real = juger(real_pred, 3, (1, 5), text=text)
        s_real = note / 5.0
        intention = str(gold.get("intention") or "")
        desc = INTENTION_DESCRIPTIONS.get(intention, intention)
        s_fidele, raison_fidele = juger(fid_pred, 0, (0, 1), text=text, intention_description=desc)
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"[métrique] appel {counter['n']} struct={s_struct:.2f} réalisme={s_real:.2f} fidélité={s_fidele:.2f}")
        # La fidélité pèse le plus lourd : un texte hors-sujet mais bien formé
        # et réaliste (ex. exemple /lire parlant d'une capture wiki) est le
        # défaut qui a motivé cet ajout — cf. mémoire projet routage Laya.
        return dspy.Prediction(
            score=0.25 * s_real + 0.25 * s_struct + 0.5 * s_fidele,
            feedback=construire_feedback(text, s_struct, s_real, raison_real, float(s_fidele), raison_fidele),
        )

    return metric


def main(argv=None) -> int:
    cfg = parse_args(argv)
    initial_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")
    seed = json.loads(Path(cfg.seed_path).read_text(encoding="utf-8"))
    # Générateur évalué (température modérée) et réflexion (température haute)
    # séparés : un seul LM à 1,0 rendait les scores trop bruités (D3).
    dspy.settings.configure(lm=make_lm(cfg.generator_model, cfg.student_temperature, cfg.deepseek_api_base))
    reflection_lm = make_lm(cfg.generator_model, cfg.reflection_temperature, cfg.deepseek_api_base)
    eval_lm = make_lm(cfg.eval_model, 0.0, cfg.deepseek_api_base, max_tokens=128)
    rng = random.Random(0)
    generator = build_example_generator(initial_prompt, seed, rng)
    trainset, valset = split_trainset(build_trainset(seed, rng), cfg.valset_size)
    teleprompter = dspy.GEPA(
        metric=make_metric(eval_lm),
        reflection_lm=reflection_lm,
        max_metric_calls=cfg.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
        num_threads=2,  # sanroque déjà proche de la limite mémoire (autres services actifs)
    )
    compiled = teleprompter.compile(generator, trainset=trainset, valset=valset)
    # Le programme entier est transmis à generate_corpus.py (D5) ; GEPAPrompt.txt
    # reste écrit pour lecture humaine et comme repli.
    compiled.save(cfg.program_path)
    Path(cfg.gepa_prompt_path).write_text(compiled.generate.signature.instructions, encoding="utf-8")
    print(f"Programme optimisé sauvegardé dans {cfg.program_path} (instruction : {cfg.gepa_prompt_path})")
    return 0


if __name__ == "__main__":
    # logging.basicConfig() ne suffit pas : le logger "dspy" a propagate=False
    # et son propre StreamHandler (stdout) — les messages n'atteignent jamais
    # le logger racine, donc jamais gepa_llm_calls.log. Constaté le 30/09/2026,
    # fichier resté à 0 octet malgré des passages réels. Handler attaché
    # directement sur le logger "dspy" pour contourner.
    _dspy_logger = logging.getLogger("dspy")
    _dspy_logger.handlers.clear()
    _fh = logging.FileHandler("gepa_llm_calls.log", mode="a")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    _dspy_logger.addHandler(_fh)
    _dspy_logger.setLevel(logging.INFO)
    raise SystemExit(main())
