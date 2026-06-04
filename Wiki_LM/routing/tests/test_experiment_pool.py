import re

from cost import CostTracker
from experiment import build_pool


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


def _fake_generate(prompt):
    # Détecte l'agent cible via la phrase distinctive 'vers l'agent "X"'
    target = "clarify"
    for agent in ("gog", "wikilm", "clarify"):
        if f'vers l\'agent "{agent}"' in prompt:
            target = agent
            break
    text = (
        f'{{"message": "exemple A pour {target}", "agent": "{target}"}}\n'
        f'{{"message": "exemple B pour {target}", "agent": "{target}"}}\n'
    )
    return text, {"prompt_tokens": 50, "completion_tokens": 20}


def _fake_critique_batch(prompt, max_tokens):
    # Compte les items numérotés, garde les impairs (1, 3…), rejette les pairs (2, 4…)
    items = re.findall(r'^\d+\.', prompt, re.MULTILINE)
    n = len(items)
    verdicts = ["GARDER" if i % 2 == 1 else "REJETER" for i in range(1, n + 1)]
    return "\n".join(verdicts), {"prompt_tokens": 10 * n, "completion_tokens": n}


def test_build_pool_generates_critiques_and_tracks_cost():
    cost = CostTracker(prices={})
    pool, clarify_pool = build_pool(
        _AGENTS, max_per_agent=2, clarify_k=2,
        generate_fn=_fake_generate, critique_fn=_fake_critique_batch, cost=cost,
    )
    # 2 agents réels × 1 gardé (item 1) = 2 ; clarify séparé = 1
    assert len(pool) == 2
    assert {r["agent"] for r in pool} == {"gog", "wikilm"}
    assert all("exemple A" in r["message"] for r in pool)
    assert len(clarify_pool) == 1
    assert clarify_pool[0]["agent"] == "clarify"
    assert cost.tokens("deepseek-chat")[0] > 0
    assert cost.tokens("mistralai/Mistral-Small-4-119B-2603")[0] > 0
