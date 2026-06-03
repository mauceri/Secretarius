from critique import build_critique_prompt, parse_verdict, critique_candidates


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


def test_build_critique_prompt_mentions_target_and_message():
    prompt = build_critique_prompt({"message": "envoie un mail", "agent": "gog"}, _AGENTS)
    assert "gog" in prompt
    assert "envoie un mail" in prompt
    assert "GARDER" in prompt and "REJETER" in prompt


def test_parse_verdict_keep():
    assert parse_verdict("GARDER") is True
    assert parse_verdict("  garder  ") is True
    assert parse_verdict("Verdict : GARDER") is True


def test_parse_verdict_reject():
    assert parse_verdict("REJETER") is False
    assert parse_verdict("je ne sais pas") is False
    assert parse_verdict("GARDER ou REJETER ? REJETER") is False


def test_critique_candidates_filters_and_sums_usage():
    candidates = [
        {"message": "bon", "agent": "gog"},
        {"message": "mauvais", "agent": "gog"},
        {"message": "bon2", "agent": "gog"},
    ]

    def _fake_critique(prompt):
        verdict = "GARDER" if ("bon" in prompt) else "REJETER"
        return verdict, {"prompt_tokens": 10, "completion_tokens": 1}

    kept, usage = critique_candidates(candidates, _AGENTS, _fake_critique)
    assert [c["message"] for c in kept] == ["bon", "bon2"]
    assert usage == {"prompt_tokens": 30, "completion_tokens": 3}
