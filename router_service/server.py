#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service HTTP du routeur Tiron : POST /route {message} -> {status, command, args}."""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router_service.router import GogGate, WIKI_CMDS, GOG_CMDS
from router_service.faq import FaqIndex, FAQ_PATH

LLAMA_BASE = os.environ.get("TIRON_LLAMA_BASE", "http://127.0.0.1:8998")
LLAMA_KEY = os.environ.get("TIRON_LLAMA_KEY", "")
SYSTEM_ROUTE = ('Routeur de commandes Tiron. Pour chaque message, répondre '
                'uniquement avec un objet JSON : {"command": "/commande" ou '
                'null, "args": "arguments bruts ou chaîne vide"}.')
# Grammaire de sortie : force un JSON conforme, élimine le texte parasite
# (ex. "}]" en trop) que le modèle ajoutait parfois après un JSON par ailleurs
# correct (constaté 2026-07-04, cf. rapport d'éval).
COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": ["string", "null"]},
        "args": {"type": "string"},
    },
    "required": ["command", "args"],
}

_gate = None  # chargé au démarrage (Step 5)
_faq = None   # FaqIndex, chargé au démarrage


def call_adapter(message: str):
    body = {"messages": [{"role": "system", "content": SYSTEM_ROUTE},
                         {"role": "user", "content": message}],
            "max_tokens": 60, "temperature": 0,
            "json_schema": COMMAND_SCHEMA}
    headers = {"Content-Type": "application/json"}
    if LLAMA_KEY:
        headers["Authorization"] = "Bearer " + LLAMA_KEY
    req = urllib.request.Request(LLAMA_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=headers)
    d = json.load(urllib.request.urlopen(req, timeout=30))
    raw = d["choices"][0]["message"]["content"].strip()
    parsed = json.loads(raw)
    return parsed.get("command"), parsed.get("args", "")


# Commandes qui exigent un argument : un appel sans argument est une erreur
# d'usage, pas une capture/requête vide à déléguer.
NEEDS_ARG = {"/c", "/q", "/source", "/chercher", "/repondre"}


def route_message(message: str) -> dict:
    # Commande explicite (l'utilisateur a tapé /c, /q, …) : honorée telle quelle,
    # jamais soumise au SLM. Le routeur ne classe que le texte libre ; re-classer
    # une commande explicite ne peut que la corrompre (ex. /c pris pour /source).
    stripped = message.strip()
    if stripped.startswith("/"):
        parts = stripped.split(None, 1)
        cmd = parts[0]
        if cmd in WIKI_CMDS or cmd in GOG_CMDS:
            args = parts[1] if len(parts) > 1 else ""
            if cmd in NEEDS_ARG and not args.strip():
                return {"status": "answer", "reply": f"Usage : {cmd} <argument>"}
            return {"status": "ok", "command": cmd, "args": args}
    if _faq is not None and not message.lstrip().startswith("/"):
        try:
            entry = _faq.lookup(message)
        except Exception:
            entry = None
        if entry is not None:
            return {"status": "answer", "reply": entry["answer"]}
    try:
        command, args = call_adapter(message)
    except Exception:
        return {"status": "no_match"}

    if command in GOG_CMDS:
        if not _gate.gog_confident(message):
            return {"status": "no_match"}
        return {"status": "ok", "command": command, "args": args}
    if command in WIKI_CMDS:
        return {"status": "ok", "command": command, "args": args}
    return {"status": "no_match"}


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


def main():
    global _gate, _faq
    print("Chargement BGE-M3...", flush=True)
    _gate = GogGate()
    _faq = FaqIndex(_gate._embed)
    print(f"FAQ chargée ({FAQ_PATH})", flush=True)
    print("Prêt, écoute sur :8999", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8999), Handler).serve_forever()


if __name__ == "__main__":
    main()
