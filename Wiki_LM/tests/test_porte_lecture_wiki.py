"""Lecture du wiki par la porte Grok, contre un faux wiki-lm-server."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from porte import lecture_wiki
from porte.coffre import Refus
from porte.lecture_wiki import WikiIndisponible


class _Faux(BaseHTTPRequestHandler):
    reponses = {}
    en_cours = 0
    max_simultanes = 0
    recus = []

    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).recus.append((self.path, corps))
        type(self).en_cours += 1
        type(self).max_simultanes = max(type(self).max_simultanes, type(self).en_cours)
        time.sleep(0.2 if self.path == "/query" else 0)
        type(self).en_cours -= 1
        statut, data = type(self).reponses[self.path]
        self.send_response(statut)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def faux_wiki():
    _Faux.reponses = {
        "/query": (200, {"text": "Synthèse.", "references": ["src-a", "c-b"], "brief": "b"}),
        "/run": (200, {"results": [{"title": "A", "excerpt": "ex"}]}),
    }
    _Faux.en_cours = _Faux.max_simultanes = 0
    _Faux.recus = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Faux)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_question(faux_wiki):
    assert lecture_wiki.question(faux_wiki, "Substack ?") == {"synthese": "Synthèse.", "references": ["src-a", "c-b"]}
    assert _Faux.recus == [("/query", {"question": "Substack ?"})]


def test_questions_une_a_la_fois(faux_wiki):
    fils = [threading.Thread(target=lecture_wiki.question, args=(faux_wiki, f"q{i}")) for i in range(3)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    assert _Faux.max_simultanes == 1


def test_question_erreur_du_wiki(faux_wiki):
    _Faux.reponses["/query"] = (500, {"error": "KB vide — lancer ingest d'abord"})
    with pytest.raises(WikiIndisponible, match="KB vide"):
        lecture_wiki.question(faux_wiki, "x")
    # le verrou est relâché : une nouvelle question passe
    _Faux.reponses["/query"] = (200, {"text": "ok", "references": []})
    assert lecture_wiki.question(faux_wiki, "y")["synthese"] == "ok"


def test_wiki_injoignable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(WikiIndisponible, match="injoignable"):
        lecture_wiki.question(f"http://127.0.0.1:{port}", "x", delai=2)


def test_recherche_utilise_seulement_r(faux_wiki):
    assert lecture_wiki.recherche(faux_wiki, "écriture IA") == [{"title": "A", "excerpt": "ex"}]
    assert _Faux.recus == [("/run", {"command": "/r", "arg": "écriture IA"})]


def test_page(tmp_path):
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "sources" / "src-a.md").write_text("contenu A")
    (wiki / "poubelle").mkdir()
    (wiki / "poubelle" / "src-vieux.md").write_text("jeté")
    assert lecture_wiki.page(wiki, "src-a") == "contenu A"
    for slug in ["src-vieux", "src-absent", "../x", "SRC-a", "x-a", "src-a.md"]:
        with pytest.raises(Refus):
            lecture_wiki.page(wiki, slug)
