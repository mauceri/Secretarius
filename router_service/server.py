#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service HTTP du routeur Tiron : POST /route {message} -> {status, command, args}."""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router_service.router import WIKI_CMDS, GOG_CMDS, CRON_CMDS, LIBELLES, embed_bge_m3
from router_service.faq import FaqIndex, FAQ_PATH
from router_service.laya_classifier import LayaClassifier
from tiron_cron.nl import detecter_planification

LAYA_CHECKPOINT = os.environ["LAYA_CHECKPOINT"]
# Confiance jointe (domaine × commande) au-dessous de laquelle Tiron demande
# « 1 ou 2 » au lieu de deviner. Choisi le 2026-10-06 par eval_laya sur
# laya_val.jsonl (639) : 97,2 % de décisions justes, 3,8 % de questions ;
# jeu manuel (71) : 81,7 % juste ou question pertinente.
SEUIL = 0.60

# Commandes qui exigent un argument : un appel sans argument est une erreur
# d'usage, pas une capture/requête vide à déléguer.
NEEDS_ARG = {"/c", "/q", "/source", "/chercher", "/repondre", "/r", "/lire", "/supprimer", "/verifie", "/cron"}

_classifier = None  # chargé au démarrage (Step 5)
_faq = None         # FaqIndex, chargé au démarrage


def route_message(message: str) -> dict:
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
    # Planification reconnue à sa forme, avant tout classifieur (2026-10-06 :
    # « rappelez-moi dans 3 minutes… » était classé hors périmètre par Laya).
    if not stripped.startswith("/") and detecter_planification(stripped, datetime.now(ZoneInfo("Europe/Paris"))):
        return {"status": "ok", "command": "/cron", "args": message}
    if _faq is not None and not message.lstrip().startswith("/"):
        try:
            entry = _faq.lookup(message)
        except Exception:
            entry = None
        if entry is not None:
            return {"status": "answer", "reply": entry["answer"]}

    if _classifier is None:
        return {"status": "no_match"}
    ranking = _classifier.classer(message)
    command, p = ranking[0]
    if p >= SEUIL:
        if command is None:
            return {"status": "no_match"}
        # args : jamais généré, toujours le message brut (décision du 2026-09-28).
        return {"status": "ok", "command": command, "args": message}
    options = [{"command": c, "label": LIBELLES[c]} for c, _ in ranking[:2]]
    return {"status": "clarify", "options": options, "args": message}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/route":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        result = route_message(body.get("message", ""))
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
