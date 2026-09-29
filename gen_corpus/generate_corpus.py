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


# Descriptions courtes pour le juge de fidélité texte↔intention — mêmes
# définitions que dans promptGenGEPA.py (dupliquées, chaque script gen_corpus
# reste indépendant et exécutable seul).
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
                  command: str | None, max_attempts: int = 3) -> tuple[dict, bool]:
    # La commande est déterminée par l'intention (intentions.json), pas par le LLM :
    # demander au modèle de la re-choisir introduit du bruit d'étiquetage quand deux
    # commandes sont sémantiquement proches (ex. /q vs /r — constaté ~99% d'erreur
    # sur wiki_search lors de l'ajout de /r). Seuls text/args restent générés.
    result = None
    accepted = False
    for attempt in range(max_attempts):
        result = predict(intention=intention, registre=registre, variante=variante)
        accepted = fidelity_check(result.text, intention)
        if accepted:
            break
    args = result.args.strip()
    if args in ('""', "''"):
        args = ""
    entry = {"text": result.text, "intention": intention, "registre": registre,
             "variante": variante, "action": {"command": command, "args": args}}
    return entry, accepted


def main(argv=None) -> int:
    cfg = parse_args(argv)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    base = os.getenv("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
    lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base,
                 model_type="chat", temperature=cfg.temperature, max_tokens=256, cache=False)
    eval_lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base,
                       model_type="chat", temperature=0.0, max_tokens=16, cache=False)
    dspy.settings.configure(lm=lm)
    fidelity_check = make_fidelity_check(eval_lm)

    prompt_p = Path(cfg.prompt_path)
    prompt_text = (prompt_p if prompt_p.exists() else Path(cfg.prompt_fallback)).read_text(encoding="utf-8")
    predict = dspy.Predict(_build_signature(prompt_text))
    intentions = json.loads(Path(cfg.intentions_path).read_text(encoding="utf-8"))
    registres = json.loads(Path(cfg.registres_path).read_text(encoding="utf-8"))

    buffer = []
    stime = time.time()
    consecutive_errors = 0
    max_consecutive = max(10, cfg.count // 10)
    rejected_kept = 0  # exemples gardés malgré un échec de fidélité après max_attempts
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in range(cfg.count):
            obj = random.choice(intentions)
            try:
                entry, accepted = generate_one(predict, fidelity_check, obj["intention"],
                                                random.choice(registres), random.choice(obj["variantes"]),
                                                obj["command"])
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
        for e in buffer:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Corpus sauvegardé dans {cfg.output} ({rejected_kept}/{cfg.count} exemples gardés malgré un échec de fidélité)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
