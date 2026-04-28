"""
Serveur Flask local pour interroger Wiki_LM depuis Obsidian.

Usage :
    python server.py [--port 5051] [--mode hybrid]

Endpoint :
    POST /query
    Body  : {"question": "...", "top_k": 5, "save": false, "mode": "hybrid"}
    Reply : {"text": "...", "references": [...], "saved_slug": ""}

    GET /health
    Reply : {"status": "ok", "pages": <n>}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, jsonify, request

from llm import LLM
from query import WikiQuery

app = Flask(__name__)
_wq: WikiQuery | None = None


@app.post("/query")
def handle_query():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Paramètre 'question' manquant"}), 400

    top_k = int(data.get("top_k", 5))
    save = bool(data.get("save", False))
    mode = str(data.get("mode", _wq.mode))

    if mode != _wq.mode:
        _wq.mode = mode

    result = _wq.query(question, top_k=top_k, save=save)
    return jsonify({
        "text": result.text,
        "references": result.references,
        "saved_slug": result.saved_slug,
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "pages": len(_wq._search._pages)})


def main() -> None:
    global _wq

    parser = argparse.ArgumentParser(description="Serveur Wiki_LM pour Obsidian")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WIKI_SERVER_PORT", 5051)))
    parser.add_argument(
        "--wiki",
        default=os.environ.get("WIKI_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM")),
    )
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "semantic", "hybrid"])
    parser.add_argument("--backend", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    llm = LLM(backend=args.backend, model=args.model) if (args.backend or args.model) else LLM()
    print(f"[server] Chargement Wiki_LM ({args.wiki})…")
    _wq = WikiQuery(args.wiki, llm=llm, mode=args.mode)
    print(f"[server] {len(_wq._search._pages)} pages indexées, mode={_wq.mode}")
    print(f"[server] Écoute sur http://127.0.0.1:{args.port}")

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
