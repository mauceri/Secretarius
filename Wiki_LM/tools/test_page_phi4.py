import json
import urllib.request
import frontmatter
from page_phi4 import generate_page_content, assemble_source_page


def test_generate_envoie_lora_nu_et_schema(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        payload = {"choices": [{"message": {"content": json.dumps({
            "resume": "Un résumé.", "points_cles": ["p1"],
            "concepts": ["c1"], "entites": ["e1"], "tags": ["t1"]})}}]}
        return __import__("io").BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = generate_page_content("des passages", base_url="http://x")
    assert out["resume"] == "Un résumé."
    assert out["entites"] == ["e1"]
    # phi-4 nu par-requête + schéma contraint
    assert captured["body"]["lora"] == [{"id": 0, "scale": 0}]
    assert "json_schema" in captured["body"]


def test_assemble_format_wiki():
    data = {"resume": "Résumé ici.", "points_cles": ["pt un", "pt deux"],
            "concepts": ["mémex"], "entites": ["Vannevar Bush"], "tags": ["histoire"]}
    md = assemble_source_page("As We May Think", "2026-07-09", data, ["technique"])
    post = frontmatter.loads(md)                 # YAML valide, sinon lève
    assert post["category"] == "source"
    assert post["title"] == "As We May Think"
    assert "technique" in post["tags"] and "histoire" in post["tags"]
    body = post.content
    assert "## Résumé" in body and "Résumé ici." in body
    assert "## Points clés" in body and "- pt un" in body
    assert "- concept: mémex" in body
    assert "- entité: Vannevar Bush" in body
    assert "## Liens internes suggérés" in body
