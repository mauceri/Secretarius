import io
from urllib import error as urlerror
from unittest.mock import patch

from secretarius.expression_extractor import (
    _filter_verbatim_expressions,
    _parse_expressions_output,
    _post_chat_completion,
)


def test_filter_verbatim_expressions_keeps_only_exact_substrings() -> None:
    chunk = "Le camail vert contraste avec le voile blanc."
    expressions = [
        "camail vert",
        "voile blanc",
        "camail rouge",
        "Contraste",
        "",
        "voile blanc",
    ]
    kept, removed = _filter_verbatim_expressions(chunk, expressions)
    assert kept == ["camail vert", "voile blanc"]
    assert removed == 2


def test_parse_expressions_output_recovers_from_truncated_json_list() -> None:
    raw = '["têtes entassées", "charnier", "chambre aux deniers", "porte-paniers", "é'
    parsed, warning = _parse_expressions_output(raw)
    assert parsed == ["têtes entassées", "charnier", "chambre aux deniers", "porte-paniers"]
    assert warning is not None
    assert "recovered from partial json string list" in warning


def test_post_chat_completion_calls_only_configured_url() -> None:
    calls: list[str] = []

    class _Resp:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        url = req.full_url
        calls.append(url)
        if url == "http://127.0.0.1:8989/v1/chat/completions":
            payload = '{"choices":[{"message":{"content":"[]"}}]}'
            return _Resp(payload)
        raise urlerror.HTTPError(url, 405, "Method Not Allowed", hdrs=None, fp=io.BytesIO())

    with patch("secretarius.expression_extractor.urlrequest.urlopen", new=fake_urlopen):
        raw, warning = _post_chat_completion(
            llama_cpp_url="http://127.0.0.1:8989/v1/chat/completions",
            payload={
                "model": "local-llama-cpp",
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                "max_tokens": 32,
            },
            timeout_s=5.0,
        )
    assert raw == "[]"
    assert warning is None
    assert calls == ["http://127.0.0.1:8989/v1/chat/completions"]
