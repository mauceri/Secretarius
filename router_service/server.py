#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service HTTP du routeur Tiron : POST /route {message} -> {status, command, args}."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router_service.router import WIKI_CMDS, GOG_CMDS, CRON_CMDS, LIBELLES, embed_bge_m3
from router_service.faq import FaqIndex, FAQ_PATH
from router_service.laya_classifier import LayaClassifier

LAYA_CHECKPOINT = os.environ["LAYA_CHECKPOINT"]
# Confiance jointe (domaine × commande) au-dessous de laquelle Tiron propose un
# choix de commandes au lieu de deviner. 0.75 : choix de l'utilisateur
# (2026-10-06), plus prudent que le 0.60 calculé par eval_laya sur la validation
# (97,2 % de décisions justes, 3,8 % de questions à 0.60).
SEUIL = 0.75
# Options proposées : jusqu'à 3, celles sous 5 % écartées, jamais moins de 2.
MAX_OPTIONS = 3
P_MIN_OPTION = 0.05


def options_proposees(ranking: list) -> list[dict]:
    retenues = [c for c, p in ranking[:MAX_OPTIONS] if p >= P_MIN_OPTION]
    if len(retenues) < 2:
        retenues = [c for c, _ in ranking[:2]]
    return [{"command": c, "label": LIBELLES[c]} for c in retenues]

# Commandes qui exigent un argument : un appel sans argument est une erreur
# d'usage, pas une capture/requête vide à déléguer.
NEEDS_ARG = {"/c", "/q", "/source", "/chercher", "/repondre", "/r", "/lire", "/supprimer", "/verifie", "/cron"}

_classifier = None  # chargé au démarrage (Step 5)
_faq = None         # FaqIndex, chargé au démarrage


# FAQ consultée avant Laya, en deux bandes (2026-10-07). Mesures BGE-M3 :
# questions visées 0,797–0,999 ; vraies demandes 0,309–0,738 (« cherche les
# mails de la banque » 0,738). Marge étroite : écrire dans faits.md des
# formulations proches de celles de l'utilisateur.
SEUIL_FAQ_DIRECT = 0.85        # réponse donnée telle quelle
SEUIL_FAQ_PROPOSITION = 0.75   # réponse proposée, avec « continuer » possible


def _consulter_faq(message: str) -> dict | None:
    if _faq is None or message.lstrip().startswith("/"):
        return None
    try:
        score, entry = _faq.meilleure(message)
    except Exception:
        return None
    if entry is None or score < SEUIL_FAQ_PROPOSITION:
        return None
    if score >= SEUIL_FAQ_DIRECT:
        return {"status": "answer", "reply": entry["answer"]}
    return {"status": "faq_proposee", "reply": entry["answer"], "args": message}


def route_message(message: str, sans_faq: bool = False) -> dict:
    # Commande explicite (l'utilisateur a tapé /c, /q, …) : honorée telle quelle,
    # jamais soumise à Laya. Le routeur ne classe que le texte libre ; re-classer
    # une commande explicite ne peut que la corrompre (ex. /c pris pour /source).
    stripped = message.strip()
    if stripped.startswith("/"):
        parts = stripped.split(None, 1)
        cmd = parts[0]
        if cmd in WIKI_CMDS or cmd in GOG_CMDS or cmd in CRON_CMDS:
            args = parts[1] if len(parts) > 1 else ""
            if cmd in NEEDS_ARG and not args.strip():
                return {"status": "answer", "reply": f"Usage : {cmd} <argument>"}
            return {"status": "ok", "command": cmd, "args": args}
    if not sans_faq and (faq := _consulter_faq(message)) is not None:
        return faq
    ranking = _classifier.classer(message) if _classifier is not None else [(None, 1.0)]
    command, p = ranking[0]
    if p >= SEUIL:
        if command is None:
            return {"status": "no_match"}
        # args : jamais généré, toujours le message brut (décision du 2026-09-28).
        return {"status": "ok", "command": command, "args": message}
    return {"status": "clarify", "options": options_proposees(ranking), "args": message}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/route":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        result = route_message(body.get("message", ""), sans_faq=bool(body.get("sans_faq")))
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # silence les logs d'accès par défaut


def main() -> None:
    global _classifier, _faq
    print("Chargement du classifieur Laya...", flush=True)
    _classifier = LayaClassifier(LAYA_CHECKPOINT)
    _faq = FaqIndex(embed_bge_m3)
    print(f"FAQ chargée ({FAQ_PATH})", flush=True)
    print("Prêt, écoute sur :8999", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8999), Handler).serve_forever()


if __name__ == "__main__":
    main()
