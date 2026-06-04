import json

from router_base import RouteResult, Router, load_agents, load_corpus


def test_route_result_fields():
    r = RouteResult(agent="gog", confidence=0.9)
    assert r.agent == "gog"
    assert r.confidence == 0.9


class _KeywordRouter(Router):
    def route(self, message):
        agent = "gog" if "mail" in message.lower() else "clarify"
        return RouteResult(agent, 1.0)


def test_router_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        Router()


def test_concrete_router_routes():
    assert _KeywordRouter().route("Envoie un mail").agent == "gog"
    assert _KeywordRouter().route("Bonjour").agent == "clarify"


def test_load_agents(tmp_path):
    p = tmp_path / "agents.json"
    p.write_text('{"agents":[{"name":"gog","description":"mail"}]}', encoding="utf-8")
    agents = load_agents(p)
    assert agents[0]["name"] == "gog"
    assert agents[0]["description"] == "mail"


def test_load_corpus_skips_blank_lines(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"message":"salut","agent":"gog"}\n\n{"message":"yo","agent":"wikilm"}\n',
        encoding="utf-8",
    )
    rows = load_corpus(p)
    assert len(rows) == 2
    assert rows[1]["agent"] == "wikilm"
