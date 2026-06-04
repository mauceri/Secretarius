"""Suivi du coût LLM : tokens cumulés par modèle + conversion en coût."""
from __future__ import annotations

from collections import defaultdict

# Prix par défaut, en $/million de tokens. Mistral/Euria à 0 : à mesurer
# (les tokens bruts sont toujours rapportés, c'est la donnée recherchée).
PRICES = {
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    "mistralai/Mistral-Small-4-119B-2603": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    def __init__(self, prices: dict | None = None):
        self.prices = dict(PRICES) if prices is None else dict(prices)
        self._in: dict = defaultdict(int)
        self._out: dict = defaultdict(int)

    def add(self, model: str, usage: dict) -> None:
        self._in[model] += usage.get("prompt_tokens", 0)
        self._out[model] += usage.get("completion_tokens", 0)

    def tokens(self, model: str) -> tuple[int, int]:
        return (self._in[model], self._out[model])

    def cost(self, model: str) -> float:
        p = self.prices.get(model, {"input": 0.0, "output": 0.0})
        return self._in[model] / 1e6 * p["input"] + self._out[model] / 1e6 * p["output"]

    def summary(self) -> str:
        lines: list[str] = []
        total = 0.0
        for model in sorted(set(self._in) | set(self._out)):
            i, o = self._in[model], self._out[model]
            c = self.cost(model)
            total += c
            lines.append(f"  {model}: {i} in + {o} out tokens → {c:.4f} $")
        lines.append(f"  TOTAL: {total:.4f} $")
        return "\n".join(lines)
