"""_OpenAIBackend : désactivation du raisonnement de DeepSeek."""
from types import SimpleNamespace

from llm import _OpenAIBackend


def _backend(base_url):
    b = _OpenAIBackend("modele", base_url=base_url, api_key="x")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    b._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return b, captured


def test_deepseek_raisonnement_desactive():
    """deepseek-v4-flash raisonne par défaut et ces tokens comptent dans
    max_tokens : pages web tronquées avant concepts/entités (2026-10-13)."""
    b, captured = _backend("https://api.deepseek.com/v1")
    b.complete([{"role": "user", "content": "salut"}])
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_autres_fournisseurs_sans_parametre_thinking():
    b, captured = _backend("http://172.17.0.1:8001/v1")
    b.complete([{"role": "user", "content": "salut"}])
    assert "extra_body" not in captured
