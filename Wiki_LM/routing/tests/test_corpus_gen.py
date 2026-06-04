import json

from corpus_gen import (
    build_generation_prompt,
    parse_candidates,
    existing_examples,
    commit_candidates,
)


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
]


def test_prompt_zero_shot_when_no_examples():
    prompt = build_generation_prompt(_AGENTS[0], _AGENTS, examples=[], negatives=[], n=5)
    assert "gog" in prompt
    assert "wikilm" in prompt
    assert "5" in prompt


def test_prompt_includes_fewshot_examples():
    prompt = build_generation_prompt(
        _AGENTS[0], _AGENTS,
        examples=["envoie un mail à Paul"], negatives=["capture cette url"], n=3,
    )
    assert "envoie un mail à Paul" in prompt
    assert "capture cette url" in prompt


def test_parse_candidates_keeps_valid_skips_garbage():
    text = (
        '{"message": "envoie un mail", "agent": "gog"}\n'
        "blabla pas du json\n"
        '- {"message": "cherche mon agenda", "agent": "gog"}\n'
        '{"message": "", "agent": "gog"}\n'
    )
    cands = parse_candidates(text, "gog")
    assert len(cands) == 2
    assert all(c["agent"] == "gog" for c in cands)
    assert cands[1]["message"] == "cherche mon agenda"


def test_existing_examples_filters_by_agent():
    corpus = [
        {"message": "m1", "agent": "gog"},
        {"message": "w1", "agent": "wikilm"},
        {"message": "m2", "agent": "gog"},
    ]
    assert existing_examples(corpus, "gog") == ["m1", "m2"]


def test_commit_appends_valid_skips_malformed(tmp_path):
    candidates = tmp_path / "candidates_gog.jsonl"
    candidates.write_text(
        '{"message": "bon", "agent": "gog"}\n'
        "pas du json\n"
        '{"message": "aussi bon", "agent": "gog"}\n',
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"message": "déjà là", "agent": "wikilm"}\n', encoding="utf-8")

    added = commit_candidates(candidates, corpus)
    assert added == 2

    rows = [json.loads(l) for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3
    assert rows[-1]["message"] == "aussi bon"
