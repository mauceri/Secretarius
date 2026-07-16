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


def _start_stub(received):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            received["auth"] = self.headers.get("Authorization")
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = json.dumps({"choices": [{"message": {
                "content": '{"command": null, "args": ""}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_call_adapter_sends_bearer_when_key_set(monkeypatch):
    received = {}
    httpd = _start_stub(received)
    try:
        monkeypatch.setattr(router_server, "LLAMA_BASE",
                            f"http://127.0.0.1:{httpd.server_address[1]}")
        monkeypatch.setattr(router_server, "LLAMA_KEY", "secret123")
        router_server.call_adapter("bonjour")
        assert received["auth"] == "Bearer secret123"
    finally:
        httpd.shutdown()


def test_call_adapter_no_header_without_key(monkeypatch):
    received = {}
    httpd = _start_stub(received)
    try:
        monkeypatch.setattr(router_server, "LLAMA_BASE",
                            f"http://127.0.0.1:{httpd.server_address[1]}")
        monkeypatch.setattr(router_server, "LLAMA_KEY", "")
        router_server.call_adapter("bonjour")
        assert received["auth"] is None
    finally:
        httpd.shutdown()
