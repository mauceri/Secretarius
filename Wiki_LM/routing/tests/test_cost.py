from cost import CostTracker


def test_add_accumulates_tokens():
    c = CostTracker(prices={"m": {"input": 0.0, "output": 0.0}})
    c.add("m", {"prompt_tokens": 100, "completion_tokens": 20})
    c.add("m", {"prompt_tokens": 50, "completion_tokens": 5})
    assert c.tokens("m") == (150, 25)


def test_cost_uses_price_table():
    c = CostTracker(prices={"m": {"input": 1.0, "output": 2.0}})
    c.add("m", {"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
    assert abs(c.cost("m") - 2.0) < 1e-9


def test_unknown_model_zero_cost_but_tokens_tracked():
    c = CostTracker(prices={})
    c.add("inconnu", {"prompt_tokens": 10, "completion_tokens": 3})
    assert c.tokens("inconnu") == (10, 3)
    assert c.cost("inconnu") == 0.0


def test_summary_contains_total():
    c = CostTracker(prices={"m": {"input": 1.0, "output": 1.0}})
    c.add("m", {"prompt_tokens": 1_000_000, "completion_tokens": 0})
    s = c.summary()
    assert "m" in s
    assert "TOTAL" in s
