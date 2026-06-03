from router_llm import LlmRouter, build_prompt


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


def _fake_post_returning(content):
    def _post(url, payload):
        return {"choices": [{"message": {"content": content}}]}
    return _post


def test_build_prompt_lists_agents():
    prompt = build_prompt(_AGENTS)
    assert "gog" in prompt and "wikilm" in prompt and "clarify" in prompt
    assert "JSON" in prompt


def test_parses_clean_json():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('{"agent": "gog"}'))
    res = router.route("envoie un mail")
    assert res.agent == "gog"
    assert res.confidence == 1.0


def test_parses_json_embedded_in_prose():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('Voici: {"agent": "wikilm"} voilà'))
    assert router.route("capture url").agent == "wikilm"


def test_unknown_agent_is_clarify():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('{"agent": "inexistant"}'))
    assert router.route("xxx").agent == "clarify"


def test_garbage_output_is_clarify():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('je ne sais pas'))
    assert router.route("xxx").agent == "clarify"


def test_post_exception_is_clarify():
    def _boom(url, payload):
        raise RuntimeError("connexion refusée")
    router = LlmRouter(_AGENTS, post_fn=_boom)
    res = router.route("xxx")
    assert res.agent == "clarify"
    assert res.confidence == 0.0
