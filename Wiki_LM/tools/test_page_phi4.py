import io
import json
import urllib.request
from page_phi4 import generate_page_content


def test_generate_envoie_lora_nu_et_schema(monkeypatch):
    captured = {}

    class FakeResp(io.BytesIO):
        pass

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        payload = {"choices": [{"message": {"content": json.dumps({
            "resume": "Un résumé.", "points_cles": ["p1"],
            "concepts": ["c1"], "entites": ["e1"], "tags": ["t1"]})}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = generate_page_content("des passages", base_url="http://x")
    assert out["resume"] == "Un résumé."
    assert out["entites"] == ["e1"]
    # phi-4 nu par-requête + schéma contraint
    assert captured["body"]["lora"] == [{"id": 0, "scale": 0}]
    assert "json_schema" in captured["body"]
