"""Serveur MCP router-mcp — outil unique : route_intent(message) -> agent_name.

Charge le corpus Wiki_LM/routing/corpus.jsonl et agents.json au premier appel.
Utilise EmbedRouter (prototype BGE-M3) validé par l'expérience (94.7% @ 6 ex/agent).
"""
from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("router-mcp")

_HERE = Path(__file__).resolve().parent
AGENTS_PATH = _HERE / "agents.json"
CORPUS_PATH = _HERE / "corpus.jsonl"

_router = None


def _get_router(encode_fn=None):
    global _router
    if _router is None:
        from router_base import load_agents, load_corpus
        from router_embed import EmbedRouter
        agents = load_agents(AGENTS_PATH)
        corpus = load_corpus(CORPUS_PATH)
        kw = {"encode_fn": encode_fn} if encode_fn else {}
        _router = EmbedRouter.from_corpus(corpus, **kw)
    return _router


def _route(message: str, encode_fn=None) -> str:
    """Logique testable : retourne l'agent cible pour ce message."""
    router = _get_router(encode_fn)
    return router.route(message).agent


@mcp.tool()
def route_intent(message: str) -> str:
    """Détecte l'intention du message et retourne le nom de l'agent cible.

    Valeurs possibles : wikilm | gog | superpowers | clarify
    """
    return _route(message)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8903)
