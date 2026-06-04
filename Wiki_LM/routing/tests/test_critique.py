from critique import (
    build_critique_prompt, parse_verdict, critique_candidates,
    build_batch_critique_prompt, parse_batch_verdicts, critique_batch,
)


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


# ── Tests mode individuel (inchangés) ─────────────────────────────────────────

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


# ── Tests mode batch ──────────────────────────────────────────────────────────

def test_build_batch_prompt_lists_all_messages():
    cands = [
        {"message": "envoie un mail", "agent": "gog"},
        {"message": "lis mon agenda", "agent": "gog"},
    ]
    prompt = build_batch_critique_prompt(cands, _AGENTS)
    assert "gog" in prompt
    assert "envoie un mail" in prompt
    assert "lis mon agenda" in prompt
    assert "1." in prompt and "2." in prompt


def test_parse_batch_verdicts_normal():
    assert parse_batch_verdicts("GARDER\nREJETER\nGARDER", 3) == [True, False, True]


def test_parse_batch_verdicts_with_numbers():
    assert parse_batch_verdicts("1. GARDER\n2. REJETER", 2) == [True, False]


def test_parse_batch_verdicts_truncated_fills_false():
    assert parse_batch_verdicts("GARDER", 3) == [True, False, False]


def test_critique_batch_one_call_filters_correctly():
    candidates = [
        {"message": "msg1", "agent": "gog"},
        {"message": "msg2", "agent": "gog"},
        {"message": "msg3", "agent": "gog"},
    ]

    def _fake_batch(prompt, max_tokens):
        # Garde 1 et 3, rejette 2
        return "GARDER\nREJETER\nGARDER", {"prompt_tokens": 50, "completion_tokens": 3}

    kept, usage = critique_batch(candidates, _AGENTS, _fake_batch)
    assert [c["message"] for c in kept] == ["msg1", "msg3"]
    assert usage == {"prompt_tokens": 50, "completion_tokens": 3}


def test_critique_batch_empty_candidates():
    kept, usage = critique_batch([], _AGENTS, lambda p, n: ("", {}))
    assert kept == []
    assert usage["prompt_tokens"] == 0


def test_critique_batch_max_tokens_scales_with_size():
    """Vérifie que max_tokens passe bien à critique_fn."""
    received_max = []

    def _capture(prompt, max_tokens):
        received_max.append(max_tokens)
        return "GARDER\nGARDER", {"prompt_tokens": 10, "completion_tokens": 2}

    cands = [{"message": f"m{i}", "agent": "gog"} for i in range(2)]
    critique_batch(cands, _AGENTS, _capture)
    assert received_max[0] == max(16, 2 * 8)  # = 16
