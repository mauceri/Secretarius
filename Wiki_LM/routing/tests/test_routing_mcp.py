import json
from pathlib import Path

import numpy as np


def _fake_encode(texts):
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low:
            vecs.append([1.0, 0.0, 0.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0, 0.0, 0.0])
        elif "rédige" in low or "plan" in low:
            vecs.append([0.0, 0.0, 1.0, 0.0])
        else:
            vecs.append([0.0, 0.0, 0.0, 1.0])
    return np.array(vecs, dtype=np.float32)


def _write_test_corpus(tmp_path):
    agents = tmp_path / "agents.json"
    agents.write_text(json.dumps({"agents": [
        {"name": "gog",        "description": "email et agenda"},
        {"name": "wikilm",     "description": "base de connaissances"},
        {"name": "superpowers","description": "rédaction"},
        {"name": "clarify",    "description": "intention floue"},
    ]}), encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"message": "envoie un mail", "agent": "gog"}\n'
        '{"message": "envoie un mail urgent", "agent": "gog"}\n'
        '{"message": "capture url wiki", "agent": "wikilm"}\n'
        '{"message": "note dans le wiki", "agent": "wikilm"}\n'
        '{"message": "rédige un plan", "agent": "superpowers"}\n'
        '{"message": "rédige une spec", "agent": "superpowers"}\n'
        '{"message": "aide-moi", "agent": "clarify"}\n'
        '{"message": "bla bla flou", "agent": "clarify"}\n',
        encoding="utf-8",
    )
    return agents, corpus


def test_route_intent_real_agent(tmp_path):
    import routing_mcp
    agents_path, corpus_path = _write_test_corpus(tmp_path)
    routing_mcp.AGENTS_PATH = agents_path
    routing_mcp.CORPUS_PATH = corpus_path
    routing_mcp._router = None

    result = routing_mcp._route("envoie un mail", encode_fn=_fake_encode)
    assert result == "gog"


def test_route_intent_clarify(tmp_path):
    import routing_mcp
    agents_path, corpus_path = _write_test_corpus(tmp_path)
    routing_mcp.AGENTS_PATH = agents_path
    routing_mcp.CORPUS_PATH = corpus_path
    routing_mcp._router = None

    result = routing_mcp._route("aide-moi", encode_fn=_fake_encode)
    assert result == "clarify"


def test_route_intent_wikilm(tmp_path):
    import routing_mcp
    agents_path, corpus_path = _write_test_corpus(tmp_path)
    routing_mcp.AGENTS_PATH = agents_path
    routing_mcp.CORPUS_PATH = corpus_path
    routing_mcp._router = None

    result = routing_mcp._route("capture url wiki", encode_fn=_fake_encode)
    assert result == "wikilm"
