import io
import json
import urllib.request
import eval_resume


def test_judge_resume_parse_les_deux_notes(monkeypatch):
    def fake_urlopen(req, timeout=0):
        payload = {"choices": [{"message": {"content": json.dumps(
            {"fidelite": 5, "couverture": 4})}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    out = eval_resume.judge_resume("source", "reference", "candidat")
    assert out == {"fidelite": 5, "couverture": 4}
