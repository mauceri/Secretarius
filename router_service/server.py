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

LLAMA_BASE = os.environ.get("TIRON_LLAMA_BASE", "http://127.0.0.1:8998")
SYSTEM_ROUTE = ('Routeur de commandes Tiron. Pour chaque message, répondre '
                'uniquement avec un objet JSON : {"command": "/commande" ou '
                'null, "args": "arguments bruts ou chaîne vide"}.')

_gate = None  # chargé au démarrage (Step 5)


def call_adapter(message: str):
    body = {"messages": [{"role": "system", "content": SYSTEM_ROUTE},
                         {"role": "user", "content": message}],
            "max_tokens": 60, "temperature": 0}
    req = urllib.request.Request(LLAMA_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    raw = d["choices"][0]["message"]["content"].strip()
    parsed = json.loads(raw)
    return parsed.get("command"), parsed.get("args", "")


def route_message(message: str) -> dict:
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
    global _gate
    print("Chargement BGE-M3...", flush=True)
    _gate = GogGate()
    print("Prêt, écoute sur :8999", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8999), Handler).serve_forever()


if __name__ == "__main__":
    main()
