#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation de prompt via GEPA pour le corpus d'intentions Tiron."""
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

COMMANDS_KNOWN = {
    "/c", "/ingest", "/wikistatus", "/q", "/source", "/r", "/tags", "/kbupdate",
    "/supprimer", "/relire", "/verifie", "/chercher", "/connecter", "/inbox",
    "/drive", "/repondre", "/lire",
}
REQUIRES_ARGS = {
    "wiki_capture", "wiki_query", "source_read", "gog_search", "gog_reply",
    "gog_drive", "wiki_search", "gog_get", "wiki_delete", "wiki_verify",
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
    max_metric_calls: int = 300


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
    a = p.parse_args(argv)
    return Config(seed_path=a.seed, prompt_path=a.prompt, gepa_prompt_path=a.gepa_prompt,
                  generator_model=a.generator_model, eval_model=a.eval_model,
                  deepseek_api_base=a.deepseek_api_base,
                  reflection_temperature=a.temperature, max_metric_calls=a.max_metric_calls)


def _ensure_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    return key


def configure_lm(model: str, temperature: float, api_base: str) -> dspy.LM:
    key = _ensure_key()
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    lm = dspy.LM(model=model, api_key=key, api_base=base, model_type="chat",
                 temperature=temperature, max_tokens=512, cache=False)
    dspy.settings.configure(lm=lm)
    return lm


def configure_eval_lm(model: str, api_base: str) -> dspy.LM:
    key = _ensure_key()
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    return dspy.LM(model=model, api_key=key, api_base=base, model_type="chat",
                   temperature=0.0, max_tokens=16, cache=False)


def build_example_generator(prompt_text: str) -> dspy.Module:
    class GenerateExample(dspy.Signature):
        __doc__ = prompt_text
        intention: str = dspy.InputField(desc="Intention Tiron à illustrer (ex: wiki_capture)")
        registre:  str = dspy.InputField(desc="Registre du message: formel, familier, télégraphique, poli, abrégé")
        variante:  str = dspy.InputField(desc="Type de variante (ex: url_avec_tags, question_courte, sans_args…)")
        text:    str = dspy.OutputField(desc="Message utilisateur réaliste en français")
        command: str = dspy.OutputField(desc=f"Commande Tiron ({' '.join(sorted(COMMANDS_KNOWN))}) ou null")
        args:    str = dspy.OutputField(desc="Arguments bruts de la commande (chaîne vide si pas d'args)")

    class ExampleGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(GenerateExample)

        def forward(self, intention, registre, variante):
            r = self.generate(intention=intention, registre=registre, variante=variante)
            return {"text": r.text, "intention": intention, "command": r.command,
                    "args": r.args, "registre": registre, "variante": variante}

    return ExampleGenerator()


def build_trainset(seed: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(
            intention=ex["intention"],
            registre=ex.get("registre", "poli"),
            variante=ex.get("variante", "sans_args"),
            text=ex["text"],
            command=ex["action"]["command"] if ex["action"]["command"] is not None else "null",
            args=ex["action"]["args"],
        ).with_inputs("intention", "registre", "variante")
        for ex in seed
    ]


class EvalRealisme(dspy.Signature):
    """Ce message ressemble-t-il à une vraie requête utilisateur adressée à un assistant ?
    Répondre avec un entier 1..5 uniquement, sans commentaire."""
    text:  str = dspy.InputField(desc="Message utilisateur généré")
    score: int = dspy.OutputField(desc="Entier 1..5")


# Descriptions courtes utilisées uniquement pour juger la fidélité sémantique
# texte↔intention — indépendantes de GEPAPrompt.txt (qui, lui, est optimisé),
# pour ne jamais faire dépendre le juge du texte qu'il évalue indirectement.
INTENTION_DESCRIPTIONS = {
    "wiki_capture": "capturer une URL ou une note dans le wiki",
    "wiki_ingest": "lancer l'ingestion des captures en attente du wiki",
    "wiki_status": "consulter l'état de l'ingestion du wiki",
    "wiki_query": "poser une question au wiki et obtenir une réponse synthétisée par un LLM",
    "source_read": "lire une page web externe immédiatement, sans la sauvegarder",
    "gog_search": "rechercher des emails Gmail par mot-clé, expéditeur ou période",
    "gog_connect": "autoriser l'accès au compte Google",
    "gog_inbox": "lister les emails récents de la boîte de réception",
    "gog_reply": "préparer un brouillon de réponse à un email, sans l'envoyer",
    "gog_drive": "rechercher des fichiers sur Google Drive",
    "wiki_search": "rechercher par mots-clés dans le wiki, résultats bruts SANS synthèse ni résumé LLM (différent de wiki_query qui synthétise une réponse)",
    "wiki_tags": "lister les tags disponibles dans le wiki",
    "wiki_kb_update": "reconstruire la base de connaissances du wiki depuis le dernier clustering",
    "gog_get": "lire le contenu d'un email précis (par identifiant, ou le dernier/un email récent) — UNIQUEMENT un email, jamais une page wiki, une capture, un document Drive ou l'état du wiki",
    "wiki_delete": "supprimer une page du wiki (par son slug)",
    "wiki_reread": "obtenir la prochaine page du wiki à relire",
    "wiki_verify": "marquer une page du wiki comme vérifiée (par son slug)",
    "out_of_scope": "demande hors périmètre de Tiron (aucune des commandes ci-dessus ne s'applique)",
}


class EvalFidelite(dspy.Signature):
    """Ce message correspond-il vraiment à l'intention décrite, et à elle seule
    (pas à une intention voisine ni à une autre commande) ? Répondre 1 si le
    message décrit sans ambiguïté cette intention précise, 0 sinon — y compris
    si le message est plausible pour une AUTRE intention que celle donnée.
    Répondre avec 0 ou 1 uniquement, sans commentaire."""
    text: str = dspy.InputField(desc="Message utilisateur généré")
    intention_description: str = dspy.InputField(desc="Description de l'intention que le message doit illustrer")
    score: int = dspy.OutputField(desc="0 ou 1")


def make_metric(eval_lm: dspy.LM):
    class Evaluator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict(EvalRealisme)

        def forward(self, text: str) -> int:
            with dspy.settings.context(lm=eval_lm):
                out = self.pred(text=text)
            try:
                return max(1, min(5, int(out.score)))
            except Exception:
                return 3

    class FideliteEvaluator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict(EvalFidelite)

        def forward(self, text: str, intention_description: str) -> int:
            with dspy.settings.context(lm=eval_lm):
                out = self.pred(text=text, intention_description=intention_description)
            try:
                return max(0, min(1, int(out.score)))
            except Exception:
                return 0

    evaluator = Evaluator()
    fidelite_evaluator = FideliteEvaluator()
    counter = {"n": 0}

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        text = str((pred or {}).get("text") or "")
        if not text:
            return 0.0
        s_struct = structural_score(pred, gold)
        try:
            s_real = evaluator(text=text) / 5.0
        except Exception:
            s_real = 0.6
        intention = str(gold.get("intention") or "")
        desc = INTENTION_DESCRIPTIONS.get(intention, intention)
        try:
            s_fidele = float(fidelite_evaluator(text=text, intention_description=desc))
        except Exception:
            s_fidele = 0.5
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"[métrique] appel {counter['n']} struct={s_struct:.2f} réalisme={s_real:.2f} fidélité={s_fidele:.2f}")
        # La fidélité pèse le plus lourd : un texte hors-sujet mais bien formé
        # et réaliste (ex. exemple /lire parlant d'une capture wiki) est le
        # défaut qui a motivé cet ajout — cf. mémoire projet routage Laya.
        return 0.25 * s_real + 0.25 * s_struct + 0.5 * s_fidele

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
    initial_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")
    seed = json.loads(Path(cfg.seed_path).read_text(encoding="utf-8"))
    configure_lm(cfg.generator_model, cfg.reflection_temperature, cfg.deepseek_api_base)
    eval_lm = configure_eval_lm(cfg.eval_model, cfg.deepseek_api_base)
    generator = build_example_generator(initial_prompt)
    trainset = build_trainset(seed)
    teleprompter = dspy.GEPA(
        metric=make_metric(eval_lm),
        reflection_lm=dspy.settings.lm,
        max_metric_calls=cfg.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
        num_threads=2,  # sanroque déjà proche de la limite mémoire (autres services actifs)
    )
    compiled = teleprompter.compile(generator, trainset=trainset)
    best_prompt = _extract_best_prompt(compiled, teleprompter, initial_prompt)
    Path(cfg.gepa_prompt_path).write_text(best_prompt, encoding="utf-8")
    print(f"Prompt optimisé sauvegardé dans {cfg.gepa_prompt_path}")
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
