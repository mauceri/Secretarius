"""
Serveur MCP Secretarius RAG.

Expose trois outils : indexer, rechercher, modifier.
"""
from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)


def _mcp():
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError:
        from fastmcp import FastMCP
        return FastMCP


def create_server():
    FastMCP = _mcp()
    mcp = FastMCP("secretarius-rag")

    from . import pipeline

    @mcp.tool()
    def indexer(texte: str, doc_id: str = "") -> str:
        """
        Indexe un document texte dans la mémoire de Secretarius.

        Args:
            texte:  Contenu du document à indexer.
            doc_id: Identifiant optionnel. Si vide, un identifiant est généré automatiquement.

        Returns:
            Confirmation JSON avec doc_id et nombre de vecteurs indexés.
        """
        result = pipeline.index(texte, doc_id=doc_id or None)
        if result.get("warning"):
            return f"Erreur : {result['warning']}"
        return json.dumps({
            "statut": "indexé",
            "doc_id": result["doc_id"],
            "vecteurs": result["n_tokens"],
        }, ensure_ascii=False)

    @mcp.tool()
    def rechercher(requete: str, top_k: int = 5) -> str:
        """
        Recherche dans la mémoire de Secretarius.

        Args:
            requete: Question ou texte à rechercher.
            top_k:   Nombre de résultats à retourner (défaut : 5).

        Returns:
            JSON avec la liste des documents les plus pertinents.
        """
        result = pipeline.search(requete, top_k=top_k)
        if result.get("warning"):
            return f"Avertissement : {result['warning']}"
        results = result.get("results", [])
        if not results:
            return "Aucun document trouvé."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"[{i}] (score={r['score']}) {r['text'][:300]}")
        return "\n\n".join(output)

    @mcp.tool()
    def modifier(doc_id: str, nouveau_texte: str) -> str:
        """
        Met à jour un document existant dans la mémoire de Secretarius.

        Args:
            doc_id:        Identifiant du document à modifier.
            nouveau_texte: Nouveau contenu du document.

        Returns:
            Confirmation de la mise à jour.
        """
        result = pipeline.update(doc_id, nouveau_texte)
        if result.get("warning"):
            return f"Erreur : {result['warning']}"
        return json.dumps({
            "statut": "mis à jour",
            "doc_id": result["doc_id"],
            "vecteurs": result["n_tokens"],
        }, ensure_ascii=False)

    return mcp


def main():
    logging.basicConfig(level=logging.INFO)
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
