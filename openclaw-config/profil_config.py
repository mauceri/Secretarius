"""Élagage d'openclaw.json selon le profil d'installation (spec 2026-10-12).

Le gabarit décrit l'installation complète ; un module absent du profil ne doit
laisser aucune trace : ni agent déclaré, ni outil autorisé ou refusé.

Usage : python3 profil_config.py <openclaw.json> --modules "wiki scout obfuscateur" --cerveau modal|ollama
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

AGENTS_DE = {"wiki": ["wiki"], "scout": ["scout"], "courrier": ["consignes"], "gog": ["gog"]}
OUTILS_DE = {
    "wiki": ["wiki_capture", "wiki_status", "wiki_ingest", "wiki_query", "wiki_tags",
             "wiki_kb_update", "wiki_search", "wiki_review", "wiki_verify", "wiki_delete"],
    "scout": ["source_read"],
    "gog": ["gog", "gog_connect_start", "gog_drive_search"],
    "courrier": ["mail_inbox", "mail_chercher", "mail_lire", "mail_envoyer"],
}
OBFUSQUE_URL = "http://172.17.0.1:8001/v1"
MODELE_OBFUSQUE_ID = "qwen3-14b-h128-a1-h02"
MODELE_OBFUSQUE = f"obfusque/{MODELE_OBFUSQUE_ID}"


def _retirer(liste, outils: set[str]) -> None:
    if isinstance(liste, list):
        liste[:] = [x for x in liste if x not in outils]


def _listes_d_outils(c: dict):
    yield c.get("tools", {}).get("sandbox", {}).get("tools", {})
    for agent in c["agents"]["entries"].values():
        t = agent.get("tools", {})
        yield t
        yield t.get("sandbox", {}).get("tools", {})


def appliquer_profil(config: dict, modules: set[str], cerveau: str) -> dict:
    c = copy.deepcopy(config)
    entries = c["agents"]["entries"]

    absents = set(AGENTS_DE) | set(OUTILS_DE)
    absents -= modules
    outils_absents = {o for m in absents for o in OUTILS_DE.get(m, [])}
    for m in absents:
        for agent in AGENTS_DE.get(m, []):
            entries.pop(agent, None)
    for bloc in _listes_d_outils(c):
        _retirer(bloc.get("allow"), outils_absents)
        _retirer(bloc.get("deny"), outils_absents)

    providers = c["models"]["providers"]
    if "obfuscateur" in modules:
        providers["obfusque"]["baseUrl"] = OBFUSQUE_URL
    else:
        providers.pop("obfusque", None)
        c["agents"]["defaults"].get("models", {}).pop(MODELE_OBFUSQUE, None)

    if cerveau == "modal":
        entries["main"]["model"] = {"primary": MODELE_OBFUSQUE}
        c["agents"]["defaults"]["model"]["primary"] = MODELE_OBFUSQUE

    if "laya" not in modules:
        c["plugins"]["entries"]["derisk-deleg"]["config"] = {"routeur": False}

    wiki = entries.get("wiki")
    if wiki is not None and cerveau != "ollama":
        docker = wiki["sandbox"]["docker"]
        env = docker["env"]
        for k in [k for k in env if k.startswith("OLLAMA_") or k.startswith("WIKI_INGEST_LLM_FALLBACK_")]:
            del env[k]
        env.pop("OPENAI_API_KEY_FILE", None)
        env.update({
            "WIKI_LLM_BACKEND": "openai",
            "WIKI_INGEST_LLM_BACKEND": "openai",
            "OPENAI_BASE_URL": OBFUSQUE_URL,
            "OPENAI_MODEL": MODELE_OBFUSQUE_ID,
            "WIKI_INGEST_LLM_MODEL": MODELE_OBFUSQUE_ID,
            "OPENAI_API_KEY": "local",
        })
        # Sans clé Infomaniak, le fichier monté n'existe pas : Docker refuserait le conteneur.
        docker["binds"] = [b for b in docker["binds"] if "infomaniak-key" not in b]
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fichier")
    p.add_argument("--modules", required=True)
    p.add_argument("--cerveau", required=True, choices=["modal", "ollama"])
    a = p.parse_args()
    chemin = Path(a.fichier)
    c = appliquer_profil(json.loads(chemin.read_text()), set(a.modules.split()), a.cerveau)
    chemin.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
    # .bak synchronisé : sinon la gateway prend notre écriture pour un « clobber »
    # et restaure l'ancienne configuration au démarrage (même remède que install.sh).
    shutil.copyfile(chemin, chemin.with_name("openclaw.json.bak"))


if __name__ == "__main__":
    main()
