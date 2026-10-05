#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère corpus.jsonl à partir du prompt optimisé par GEPA."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
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

from chaine import EvalFidelite, INTENTION_DESCRIPTIONS, est_quasi_doublon, tirer_contraintes  # noqa: E402
from promptGenGEPA import build_example_generator  # noqa: E402


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
    seed_path: str = "seed.json"
    program_path: str = "gepa_program.json"
    sondage_every: int = 50


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
    p.add_argument("--seed", default="seed.json")
    p.add_argument("--program", default="gepa_program.json")
    p.add_argument("--sondage-every", type=int, default=50)
    a = p.parse_args(argv)
    return Config(count=a.count, batch_size=a.batch_size, report_every=a.report_every,
                  prompt_path=a.prompt, intentions_path=a.intentions, registres_path=a.registres,
                  output=a.output, generator_model=a.model,
                  deepseek_api_base=a.deepseek_api_base, temperature=a.temperature,
                  seed_path=a.seed, program_path=a.program, sondage_every=a.sondage_every)


def make_fidelity_check(eval_lm: "dspy.LM"):
    fidelite_pred = dspy.Predict(EvalFidelite)

    def check(text: str, intention: str) -> bool:
        desc = INTENTION_DESCRIPTIONS.get(intention, intention)
        try:
            with dspy.settings.context(lm=eval_lm):
                out = fidelite_pred(text=text, intention_description=desc)
            return int(out.score) == 1
        except Exception:
            return True  # échec du juge lui-même : ne pas bloquer la génération

    return check


def generate_one(predict, fidelity_check, intention: str, registre: str, variante: str,
                 command: str | None, max_attempts: int = 3, *, contraintes: str = "",
                 deja_vus: list[str] | None = None) -> tuple[dict | None, bool]:
    # La commande est déterminée par l'intention (intentions.json), pas par le LLM :
    # demander au modèle de la re-choisir introduit du bruit d'étiquetage quand deux
    # commandes sont sémantiquement proches (ex. /q vs /r — constaté ~99% d'erreur
    # sur wiki_search lors de l'ajout de /r). Seuls text/args restent générés.
    # Un quasi-doublon d'un exemple déjà gardé compte comme un essai raté ; s'il
    # persiste, l'exemple est écarté (None) plutôt que gardé (D4).
    result = None
    accepted = False
    doublon = False
    for attempt in range(max_attempts):
        result = predict(intention=intention, registre=registre, variante=variante, contraintes=contraintes)
        text = result["text"] if isinstance(result, dict) else result.text
        doublon = bool(deja_vus) and est_quasi_doublon(text, deja_vus)
        if doublon:
            continue
        accepted = fidelity_check(text, intention)
        if accepted:
            break
    if doublon:
        return None, False
    raw_args = result["args"] if isinstance(result, dict) else result.args
    args = str(raw_args or "").strip()
    if args in ('""', "''"):
        args = ""
    entry = {"text": text, "intention": intention, "registre": registre,
             "variante": variante, "contraintes": contraintes,
             "action": {"command": command, "args": args}}
    return entry, accepted


def charger_programme(prompt_path: str, fallback: str, program_path: str,
                      seed: list[dict], rng: random.Random):
    """Recharge le programme optimisé par GEPA tel quel (D5) ; à défaut, le
    construit depuis GEPAPrompt.txt, puis prompt-init.txt."""
    p = Path(prompt_path)
    texte = (p if p.exists() else Path(fallback)).read_text(encoding="utf-8")
    prog = build_example_generator(texte, seed, rng)
    if Path(program_path).exists():
        prog.load(program_path)
    return prog


_MARQUEURS = [(j, rf"\b{j}") for j in ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")] + [
    ("tous les jours", r"tous les jours|chaque jour"), ("chaque matin", r"chaque matin|tous les matins"),
    ("soir", r"\bsoir"), ("demain", r"\bdemain"), ("dans …", r"\bdans \d|\bdans (une|un|deux|trois)\b"),
    ("date (le N)", r"\ble \d"), ("mois", r"\bmois\b"),
]


def _compte_marqueurs(textes: list[str]) -> list[tuple[str, int]]:
    return [(nom, sum(bool(re.search(rx, t.lower())) for t in textes)) for nom, rx in _MARQUEURS]


def rapport_diversite(entries: list[dict]) -> str:
    par_intention: dict[str, list[str]] = {}
    for e in entries:
        par_intention.setdefault(e["intention"], []).append(e["text"])
    lignes = []
    for intention, textes in sorted(par_intention.items()):
        n = len(textes)
        moy = sum(len(t.split()) for t in textes) / n
        attaques = sum(t.lower().startswith(("bonjour", "tiron")) for t in textes)
        doublons = n - len(set(textes))
        lignes.append(f"{intention} ({n}) : {moy:.1f} mots en moyenne ; Bonjour/Tiron : {attaques} ; doublons exacts : {doublons}")
        if intention == "cron_schedule":
            for nom, c in sorted(_compte_marqueurs(textes), key=lambda x: -x[1]):
                if c:
                    lignes.append(f"  {nom} : {c}/{n} ({100 * c / n:.0f} %)")
    return "\n".join(lignes)


def sonder(entries: list[dict], rejets_fidelite: int, random_: random.Random, k: int = 5) -> tuple[str, str | None]:
    """Sondage en cours de génération (demande de l'utilisateur, 2026-10-05) :
    rapport + échantillon, et alerte de dérive qui arrête la génération."""
    echantillon = random_.sample(entries, min(k, len(entries)))
    texte = rapport_diversite(entries) + "\n\nÉchantillon :" + "".join(f"\n- {e['text']}" for e in echantillon)
    alerte = None
    cron = [e["text"] for e in entries if e["intention"] == "cron_schedule"]
    if len(cron) >= 50:
        for nom, c in _compte_marqueurs(cron):
            if c / len(cron) > 0.4:
                alerte = f"« {nom} » dans {c}/{len(cron)} exemples cron ({100 * c / len(cron):.0f} %)"
                break
    if alerte is None:
        # Ventilé par intention : un taux global sous le seuil peut masquer une
        # intention presque entièrement infidèle (wiki_kb_update, 2026-10-05).
        par_intention: dict[str, list[bool]] = {}
        for e in entries:
            if "fidelity_ok" in e:
                par_intention.setdefault(e["intention"], []).append(bool(e["fidelity_ok"]))
        for intention, oks in sorted(par_intention.items()):
            echecs = oks.count(False)
            if len(oks) >= 20 and echecs / len(oks) > 0.4:
                alerte = f"fidélité de {intention} : {echecs}/{len(oks)} échecs ({100 * echecs / len(oks):.0f} %)"
                break
    if alerte is None and entries and rejets_fidelite / len(entries) > 0.3:
        alerte = f"rejets de fidélité : {rejets_fidelite}/{len(entries)} ({100 * rejets_fidelite / len(entries):.0f} %)"
    return texte, alerte


def main(argv=None) -> int:
    cfg = parse_args(argv)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    base = os.getenv("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
    lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base,
                 model_type="chat", temperature=cfg.temperature, max_tokens=256, cache=False)
    eval_lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base,
                       model_type="chat", temperature=0.0, max_tokens=128, cache=False)  # le juge écrit aussi une raison
    dspy.settings.configure(lm=lm)
    fidelity_check = make_fidelity_check(eval_lm)

    seed = json.loads(Path(cfg.seed_path).read_text(encoding="utf-8"))
    rng = random.Random()
    predict = charger_programme(cfg.prompt_path, cfg.prompt_fallback, cfg.program_path, seed, rng)
    intentions = json.loads(Path(cfg.intentions_path).read_text(encoding="utf-8"))
    registres = json.loads(Path(cfg.registres_path).read_text(encoding="utf-8"))

    buffer = []
    stime = time.time()
    consecutive_errors = 0
    max_consecutive = max(10, cfg.count // 10)
    rejected_kept = 0  # exemples gardés malgré un échec de fidélité après max_attempts
    ecartes_doublons = 0
    deja_vus: dict[str, list[str]] = {}
    gardes: list[dict] = []
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in range(cfg.count):
            obj = random.choice(intentions)
            try:
                vus = deja_vus.setdefault(obj["intention"], [])
                entry, accepted = generate_one(predict, fidelity_check, obj["intention"],
                                                random.choice(registres), random.choice(obj["variantes"]),
                                                obj["command"],
                                                contraintes=tirer_contraintes(obj["intention"], rng),
                                                deja_vus=vus)
                if entry is None:
                    ecartes_doublons += 1
                    print(f"[{i+1}] Quasi-doublon persistant, écarté ({obj['intention']})", flush=True)
                    continue
                vus.append(entry["text"])
                gardes.append(entry)
                entry["fidelity_ok"] = accepted  # retiré au post-traitement, jamais lu en aval
                if not accepted:
                    rejected_kept += 1
                    print(f"[{i+1}] Fidélité non confirmée après relances (gardé quand même) : "
                          f"{obj['intention']} — {entry['text'][:80]!r}", flush=True)
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
            if cfg.sondage_every and (i + 1) % cfg.sondage_every == 0 and gardes:
                texte, alerte = sonder(gardes, rejected_kept, rng)
                print(f"--- Sondage à {i+1} ---\n{texte}", flush=True)
                if alerte:
                    print(f"ARRÊT : dérive — {alerte}", flush=True)
                    break
        for e in buffer:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Corpus sauvegardé dans {cfg.output} ({rejected_kept}/{cfg.count} exemples gardés malgré un échec de fidélité, "
          f"{ecartes_doublons} quasi-doublons écartés)")
    print(rapport_diversite(gardes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
