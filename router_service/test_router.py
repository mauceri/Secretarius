from router_service.router import embed_bge_m3, WIKI_CMDS, GOG_CMDS


def test_embed_bge_m3_returns_normalized_vectors():
    import torch
    vecs = embed_bge_m3(["bonjour", "au revoir"])
    assert vecs.shape[0] == 2
    norms = torch.linalg.norm(vecs, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_command_sets_disjoint():
    assert WIKI_CMDS.isdisjoint(GOG_CMDS)


import json
import threading
import time
import urllib.request

from router_service import server as router_server


class _FakeClassifier:
    def __init__(self, ranking):
        self._ranking = ranking

    def classer(self, message):
        return self._ranking


def test_nl_au_dessus_du_seuil_decide(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.95), ("/c", 0.03)]))
    assert router_server.route_message("c koi SPLADE ?") == {"status": "ok", "command": "/q", "args": "c koi SPLADE ?"}


def test_nl_aucune_au_dessus_du_seuil(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([(None, 0.97), ("/q", 0.02)]))
    assert router_server.route_message("il fait beau") == {"status": "no_match"}


def test_seuil_est_0_75():
    assert router_server.SEUIL == 0.75


def test_jusqu_a_trois_options_les_improbables_ecartees(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier",
                        _FakeClassifier([("/chercher", 0.5), ("/lire", 0.3), (None, 0.15), ("/q", 0.04)]))
    r = router_server.route_message("un mail de Paul ?")
    assert [o["command"] for o in r["options"]] == ["/chercher", "/lire", None]


def test_toujours_au_moins_deux_options(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.70), ("/c", 0.02)]))
    assert len(router_server.route_message("x")["options"]) == 2


def test_faq_dans_le_chemin_de_q(monkeypatch):
    class _Faq:
        def lookup(self, message):
            return {"answer": "Je suis animé par qwen3:8b."}
    monkeypatch.setattr(router_server, "_faq", _Faq())
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.9)]))
    assert router_server.route_message("quel modèle t'anime ?") == {"status": "answer", "reply": "Je suis animé par qwen3:8b."}


def test_q_sans_reponse_faq_va_au_wiki(monkeypatch):
    class _Faq:
        def lookup(self, message):
            return None
    monkeypatch.setattr(router_server, "_faq", _Faq())
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.9)]))
    assert router_server.route_message("c koi SPLADE")["command"] == "/q"


def test_nl_sous_le_seuil_demande_precision(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/lire", 0.45), ("/chercher", 0.40), (None, 0.1)]))
    r = router_server.route_message("le dernier mail de ma sœur")
    assert r["status"] == "clarify" and r["args"] == "le dernier mail de ma sœur"
    assert [o["command"] for o in r["options"]] == ["/lire", "/chercher", None]
    assert r["options"][0]["label"].startswith("lire un email")


def test_nl_option_aucune_libellee(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier([("/q", 0.5), (None, 0.45)]))
    r = router_server.route_message("astuce tomates ?")
    assert r["options"][1] == {"command": None, "label": "rien de tout ça"}


def test_commande_inconnue_tapee_passe_par_laya(monkeypatch):
    called = {"n": 0}

    class _Counting:
        def classer(self, message):
            called["n"] += 1
            return [(None, 1.0)]

    monkeypatch.setattr(router_server, "_classifier", _Counting())
    router_server.route_message("/inconnu bla")
    assert called["n"] == 1


def test_route_endpoint_end_to_end():
    router_server._classifier = _FakeClassifier([(None, 1.0)])
    router_server._faq = None
    httpd = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
        ("127.0.0.1", 0), router_server.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/route",
            data=json.dumps({"message": "/wikistatus"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.load(urllib.request.urlopen(req, timeout=5))
        assert resp["status"] in ("ok", "no_match")
    finally:
        httpd.shutdown()


def test_explicit_command_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier(("boom")))
    r = router_server.route_message("/wikistatus")
    assert r == {"status": "ok", "command": "/wikistatus", "args": ""}


def test_explicit_gog_command_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/inbox")
    assert r == {"status": "ok", "command": "/inbox", "args": ""}


def test_explicit_command_empty_required_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/c")
    assert r == {"status": "answer", "reply": "Usage : /c <argument>"}


def test_explicit_r_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/r transformers attention")
    assert r == {"status": "ok", "command": "/r", "args": "transformers attention"}


def test_explicit_r_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/r")
    assert r == {"status": "answer", "reply": "Usage : /r <argument>"}


def test_explicit_supprimer_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/supprimer src-a")
    assert r == {"status": "ok", "command": "/supprimer", "args": "src-a"}


def test_explicit_supprimer_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/supprimer")
    assert r == {"status": "answer", "reply": "Usage : /supprimer <argument>"}


def test_explicit_relire_no_arg_needed(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    assert router_server.route_message("/relire") == {"status": "ok", "command": "/relire", "args": ""}


def test_explicit_verifie_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/verifie src-a")
    assert r == {"status": "ok", "command": "/verifie", "args": "src-a"}


def test_explicit_verifie_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/verifie")
    assert r == {"status": "answer", "reply": "Usage : /verifie <argument>"}


def test_explicit_tags_kbupdate_no_arg_needed(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    assert router_server.route_message("/tags") == {"status": "ok", "command": "/tags", "args": ""}
    assert router_server.route_message("/kbupdate") == {"status": "ok", "command": "/kbupdate", "args": ""}


def test_explicit_lire_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/lire 18ab3f2")
    assert r == {"status": "ok", "command": "/lire", "args": "18ab3f2"}


def test_explicit_lire_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/lire")
    assert r == {"status": "answer", "reply": "Usage : /lire <argument>"}


def test_explicit_cron_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message('/cron add "0 8 * * 1" rappel x')
    assert r == {"status": "ok", "command": "/cron", "args": 'add "0 8 * * 1" rappel x'}


def test_explicit_cron_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    assert router_server.route_message("/cron") == {"status": "answer", "reply": "Usage : /cron <argument>"}


def test_cron_cmds_disjoint():
    from router_service.router import CRON_CMDS
    assert CRON_CMDS.isdisjoint(WIKI_CMDS | GOG_CMDS)


def test_domaines_et_commandes_couvrent_toutes_les_commandes_nl():
    from router_service.router import DOMAINE_DE, QUESTIONS, LIBELLES, WIKI_CMDS, GOG_CMDS, CRON_CMDS
    attendues = (WIKI_CMDS - {"/r"}) | GOG_CMDS | CRON_CMDS | {"/source"}
    assert set(DOMAINE_DE) == attendues
    assert set(QUESTIONS) == {"domaine", "commande_wiki", "commande_google"}
    assert set(QUESTIONS["domaine"]["criteria"]) == {"wiki", "google", "web", "planification", "aucune"}
    assert len(QUESTIONS["commande_wiki"]["criteria"]) == 9
    assert len(QUESTIONS["commande_google"]["criteria"]) == 6
    assert LIBELLES[None] == "rien de tout ça" and all(c in LIBELLES for c in DOMAINE_DE)


def test_aucune_question_ne_depasse_le_budget_laya():
    # head_max_len = 256 : au-delà, Laya tronque chaque option (constaté à 13 jetons sur 18 options).
    from router_service.router import QUESTIONS
    for q in QUESTIONS.values():
        assert sum(len(f"{k}: {v}".split()) for k, v in q["criteria"].items()) < 120

