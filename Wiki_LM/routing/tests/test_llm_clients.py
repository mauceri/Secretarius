from types import SimpleNamespace

from llm_clients import _extract


def test_extract_text_and_usage():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="bonjour"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )
    text, usage = _extract(resp)
    assert text == "bonjour"
    assert usage == {"prompt_tokens": 12, "completion_tokens": 3}
