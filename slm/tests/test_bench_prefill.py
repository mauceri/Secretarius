import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench_prefill import load_workspace, measure_ttft


def _fake_response(content="Bon"):
    chunk = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
    done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.iter_lines.return_value = [
        b"data: " + json.dumps(chunk).encode(),
        b"data: " + json.dumps(done).encode(),
        b"data: [DONE]",
    ]
    return mock


def test_measure_ttft_returns_float():
    with patch("requests.post", return_value=_fake_response()):
        result = measure_ttft("system prompt", n=3)
    assert isinstance(result, float)
    assert result >= 0.0


def test_measure_ttft_makes_n_calls():
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return _fake_response()

    with patch("requests.post", side_effect=fake_post):
        measure_ttft("system prompt", n=3)
    assert len(calls) == 3


def test_load_workspace_concatenates_md_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agents content")
    (tmp_path / "SOUL.md").write_text("soul content")
    (tmp_path / "other.txt").write_text("should be ignored")
    result = load_workspace(str(tmp_path))
    assert "agents content" in result
    assert "soul content" in result
    assert "should be ignored" not in result


def test_load_workspace_excludes_non_md(tmp_path):
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "config.json").write_text("{}")
    result = load_workspace(str(tmp_path))
    assert "{}" not in result
