import json

from llm import _OllamaBackend


def _fake_urlopen(payload):
    """Retourne un urlopen factice qui capture le corps envoyé et renvoie payload."""
    captured = {}

    def fake(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        return __import__("io").BytesIO(json.dumps(payload).encode())

    return fake, captured


def test_ollama_desactive_toujours_le_thinking(monkeypatch):
    import urllib.request

    fake, captured = _fake_urlopen({"message": {"content": "ok"}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    _OllamaBackend("phi4-mini").complete([{"role": "user", "content": "salut"}])

    assert captured["body"]["think"] is False


def test_ollama_num_gpu_absent_par_defaut(monkeypatch):
    import urllib.request

    fake, captured = _fake_urlopen({"message": {"content": "ok"}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.delenv("OLLAMA_NUM_GPU", raising=False)

    _OllamaBackend("phi4-mini").complete([{"role": "user", "content": "salut"}])

    assert "num_gpu" not in captured["body"]["options"]


def test_ollama_num_gpu_force_cpu_si_variable_definie(monkeypatch):
    import urllib.request

    fake, captured = _fake_urlopen({"message": {"content": "ok"}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("OLLAMA_NUM_GPU", "0")

    _OllamaBackend("phi4-mini").complete([{"role": "user", "content": "salut"}])

    assert captured["body"]["options"]["num_gpu"] == 0


def test_ollama_repli_sur_thinking_si_content_vide(monkeypatch):
    """Comportement existant à préserver : si content est vide (modèle qui aurait
    quand même pensé malgré think:false), on retombe sur le champ thinking."""
    import urllib.request

    fake, captured = _fake_urlopen(
        {"message": {"content": "", "thinking": "raisonnement…"}}
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    out = _OllamaBackend("phi4-mini").complete([{"role": "user", "content": "salut"}])

    assert out == "raisonnement…"
