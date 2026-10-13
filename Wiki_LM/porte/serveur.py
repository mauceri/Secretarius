"""Serveur MCP de la porte Grok (spec 2026-10-13).

Écoute sur 127.0.0.1:5052 ; Tailscale Funnel le publie sur 8443. Chaque
requête /mcp doit porter « Authorization: Bearer <jeton> » d'un agent connu
(fichier relu à chaque requête : un retrait est immédiat). Les outils
synchrones tournent dans un pool de threads (fastmcp) : une question longue
ne bloque pas les autres agents.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from starlette.middleware import Middleware
from starlette.responses import JSONResponse

from porte import coffre, jetons, lecture_wiki
from porte.coffre import Refus

journal = logging.getLogger("porte-grok")

_INSTRUCTIONS = (
    "Wiki et coffre Obsidian « Publication » de Secretarius. Lecture du wiki "
    "(wiki_recherche avant wiki_question, gratuite et instantanée) et lecture/"
    "écriture de fichiers .md du coffre ; vos fichiers doivent commencer par "
    "votre préfixe d'agent. Rien n'entre dans le wiki sans validation humaine."
)


def creer_app(racine: Path, fichier_agents: Path, wiki_url: str):
    mcp = FastMCP("secretarius-porte", instructions=_INSTRUCTIONS)
    wiki_dir = racine / "Wiki_LM" / "wiki"

    def _appel(outil: str, detail: str, fn):
        agent = jetons.agent_pour(fichier_agents, get_http_request().headers.get("authorization"))
        if agent is None:  # le middleware a déjà filtré ; défense en profondeur
            raise ToolError("jeton invalide")
        try:
            resultat = fn(agent)
        except (Refus, lecture_wiki.WikiIndisponible) as e:
            journal.info("%s %s %s refus : %s", agent, outil, detail, e)
            raise ToolError(str(e)) from None
        except Exception:
            journal.exception("%s %s %s erreur interne", agent, outil, detail)
            raise ToolError("erreur interne de la porte") from None
        journal.info("%s %s %s ok", agent, outil, detail)
        return resultat

    @mcp.tool()
    def wiki_question(question: str) -> dict:
        """Pose une question au wiki ; renvoie une synthèse et les slugs cités. ~1 min si le modèle dort."""
        return _appel("wiki_question", question[:80], lambda a: lecture_wiki.question(wiki_url, question))

    @mcp.tool()
    def wiki_recherche(mots: str) -> list[dict]:
        """Recherche par mots-clés (sans LLM, instantanée) ; renvoie titres et extraits."""
        return _appel("wiki_recherche", mots[:80], lambda a: lecture_wiki.recherche(wiki_url, mots))

    @mcp.tool()
    def wiki_page(slug: str) -> str:
        """Lit une page du wiki par son slug (src-…, c-…, e-…)."""
        return _appel("wiki_page", slug[:80], lambda a: lecture_wiki.page(wiki_dir, slug))

    @mcp.tool()
    def coffre_lister(dossier: str = "") -> list[str]:
        """Liste les fichiers .md du coffre (hors Wiki_LM/ et dossiers cachés)."""
        return _appel("coffre_lister", dossier[:80], lambda a: coffre.lister(racine, dossier))

    @mcp.tool()
    def coffre_lire(chemin: str) -> str:
        """Lit un fichier .md du coffre (chemin relatif)."""
        return _appel("coffre_lire", chemin[:80], lambda a: coffre.lire(racine, chemin))

    @mcp.tool()
    def coffre_ecrire(chemin: str, contenu: str) -> str:
        """Crée ou remplace un fichier .md ; son nom doit commencer par votre préfixe suivi de « - »."""
        return _appel("coffre_ecrire", chemin[:80], lambda a: coffre.ecrire(racine, a, chemin, contenu))

    @mcp.custom_route("/health", methods=["GET"])
    async def sante(request):
        return JSONResponse({"status": "ok"})

    class _Authentification:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["path"].startswith("/mcp"):
                en_tete = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
                if jetons.agent_pour(fichier_agents, en_tete) is None:
                    journal.info("refus 401 depuis %s", (scope.get("client") or ("?",))[0])
                    await JSONResponse({"error": "jeton absent ou invalide"}, status_code=401)(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    return mcp.http_app(path="/mcp", middleware=[Middleware(_Authentification)], stateless_http=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    racine = Path(os.environ["PORTE_COFFRE"])
    agents = Path(os.environ.get("PORTE_AGENTS", str(jetons.FICHIER_DEFAUT)))
    wiki_url = os.environ.get("PORTE_WIKI_URL", "http://127.0.0.1:5051")
    app = creer_app(racine, agents, wiki_url)
    uvicorn.run(app, host="127.0.0.1", port=5052, log_level="warning")


if __name__ == "__main__":
    main()
