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
    def __init__(self, command, prob=0.9):
        self._command = command
        self._prob = prob

    def classify(self, message):
        return self._command, self._prob


def test_route_endpoint_end_to_end():
    router_server._classifier = _FakeClassifier(None)
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


def test_unknown_slash_command_still_reaches_slm(monkeypatch):
    called = {"n": 0}

    class _Counting:
        def classify(self, message):
            called["n"] += 1
            return None, 0.0

    monkeypatch.setattr(router_server, "_classifier", _Counting())
    router_server.route_message("/inconnu bla")
    assert called["n"] == 1


def test_nl_inferred_command_uses_full_message_as_args(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/q", 0.92))
    r = router_server.route_message("qu'est-ce que le SPLADE ?")
    assert r == {"status": "ok", "command": "/q", "args": "qu'est-ce que le SPLADE ?"}


def test_nl_no_match_returns_no_match(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier(None))
    r = router_server.route_message("il fait beau aujourd'hui")
    assert r == {"status": "no_match"}


def test_gog_command_below_confidence_threshold_returns_no_match(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/inbox", 0.30))
    r = router_server.route_message("y a-t-il du nouveau ?")
    assert r == {"status": "no_match"}


def test_gog_command_above_confidence_threshold_is_ok(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/inbox", 0.80))
    r = router_server.route_message("y a-t-il du nouveau ?")
    assert r == {"status": "ok", "command": "/inbox", "args": "y a-t-il du nouveau ?"}


def test_wiki_command_from_slm_does_not_use_confidence_threshold(monkeypatch):
    # Seules les commandes gog passent par le seuil de confiance — le
    # classifieur ne renvoie de toute façon que des commandes connues.
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/q", 0.10))
    r = router_server.route_message("dis-moi ce que dit le wiki sur X")
    assert r == {"status": "ok", "command": "/q", "args": "dis-moi ce que dit le wiki sur X"}
