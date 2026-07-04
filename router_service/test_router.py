from router_service.router import GogGate, WIKI_CMDS, GOG_CMDS


def test_gog_confident_on_clear_gog_message():
    gate = GogGate()
    assert gate.gog_confident("cherche les mails de Paul cette semaine") is True


def test_gog_not_confident_on_wiki_message():
    gate = GogGate()
    assert gate.gog_confident("que dit le wiki sur le projet Alpha ?") is False


def test_command_sets_disjoint():
    assert WIKI_CMDS.isdisjoint(GOG_CMDS)


import json
import threading
import time
import urllib.request

from router_service import server as router_server


def test_route_endpoint_end_to_end():
    router_server._gate = router_server.GogGate()
    httpd = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
        ("127.0.0.1", 8999), router_server.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8999/route",
            data=json.dumps({"message": "cherche les mails de Paul"}).encode(),
            headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=30))
        assert resp["status"] in ("ok", "no_match")
    finally:
        httpd.shutdown()
