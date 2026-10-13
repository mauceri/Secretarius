"""Serveur MCP de la porte Grok, en HTTP réel sur un port local libre."""
import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

from porte import jetons
from porte.serveur import creer_app


class _FauxWiki(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        data = {"text": "Synthèse.", "references": ["src-a"]} if self.path == "/query" \
            else {"results": [{"title": "A", "excerpt": "ex"}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass


def _port_libre():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def porte(tmp_path):
    racine = tmp_path / "Publication"
    (racine / "Wiki_LM" / "wiki" / "sources").mkdir(parents=True)
    (racine / "Wiki_LM" / "wiki" / "sources" / "src-a.md").write_text("page A")
    (racine / "Notes.md").write_text("perso")
    agents = tmp_path / "agents"
    jeton = jetons.ajouter(agents, "veille")
    wiki = ThreadingHTTPServer(("127.0.0.1", 0), _FauxWiki)
    threading.Thread(target=wiki.serve_forever, daemon=True).start()
    app = creer_app(racine, agents, f"http://127.0.0.1:{wiki.server_address[1]}")
    port = _port_libre()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield {"url": f"http://127.0.0.1:{port}", "jeton": jeton, "racine": racine, "agents": agents}
    srv.should_exit = True
    wiki.shutdown()


def _appeler(porte, jeton, outil=None, args=None):
    async def run():
        transport = StreamableHttpTransport(f"{porte['url']}/mcp", headers={"Authorization": f"Bearer {jeton}"})
        async with Client(transport) as c:
            if outil is None:
                return sorted(t.name for t in await c.list_tools())
            return (await c.call_tool(outil, args or {})).data
    return asyncio.run(run())


def test_sans_jeton_401(porte):
    assert httpx.post(f"{porte['url']}/mcp", json={}).status_code == 401
    assert httpx.post(f"{porte['url']}/mcp", json={}, headers={"Authorization": "Bearer faux"}).status_code == 401


def test_health_sans_jeton(porte):
    r = httpx.get(f"{porte['url']}/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_six_outils(porte):
    assert _appeler(porte, porte["jeton"]) == [
        "coffre_ecrire", "coffre_lire", "coffre_lister", "wiki_page", "wiki_question", "wiki_recherche"]


def test_outils_wiki(porte):
    assert _appeler(porte, porte["jeton"], "wiki_question", {"question": "q"}) == \
        {"synthese": "Synthèse.", "references": ["src-a"]}
    assert _appeler(porte, porte["jeton"], "wiki_recherche", {"mots": "a"}) == [{"title": "A", "excerpt": "ex"}]
    assert _appeler(porte, porte["jeton"], "wiki_page", {"slug": "src-a"}) == "page A"


def test_ecriture_prefixee_et_refus(porte):
    assert _appeler(porte, porte["jeton"], "coffre_ecrire",
                    {"chemin": "Brouillons/veille-1.md", "contenu": "v"}) == "Brouillons/veille-1.md"
    assert (porte["racine"] / "Brouillons" / "veille-1.md").read_text() == "v"
    with pytest.raises(ToolError, match="veille-"):
        _appeler(porte, porte["jeton"], "coffre_ecrire", {"chemin": "Notes.md", "contenu": "x"})
    assert (porte["racine"] / "Notes.md").read_text() == "perso"
    with pytest.raises(ToolError, match="zone interdite"):
        _appeler(porte, porte["jeton"], "coffre_lire", {"chemin": "Wiki_LM/wiki/sources/src-a.md"})
    assert _appeler(porte, porte["jeton"], "coffre_lister", {}) == ["Brouillons/veille-1.md", "Notes.md"]


def test_revocation_sans_redemarrage(porte):
    assert _appeler(porte, porte["jeton"], "wiki_page", {"slug": "src-a"}) == "page A"
    jetons.retirer(porte["agents"], "veille")
    assert httpx.post(f"{porte['url']}/mcp", json={},
                      headers={"Authorization": f"Bearer {porte['jeton']}"}).status_code == 401


def test_journal_sans_contenu(porte, caplog):
    caplog.set_level("INFO", logger="porte-grok")
    _appeler(porte, porte["jeton"], "coffre_ecrire", {"chemin": "veille-j.md", "contenu": "CONTENU-SECRET"})
    lignes = [r.getMessage() for r in caplog.records if r.name == "porte-grok"]
    assert any("veille" in l and "coffre_ecrire" in l and "veille-j.md" in l and "ok" in l for l in lignes)
    assert not any("CONTENU-SECRET" in l for l in lignes)
