"""Interface commune des routeurs d'intention + helpers de chargement."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RouteResult:
    """Décision de routage : agent choisi + confiance dans [0, 1]."""
    agent: str
    confidence: float


class Router(ABC):
    """Tout routeur expose route(message) -> RouteResult."""

    @abstractmethod
    def route(self, message: str) -> RouteResult:
        ...


def load_agents(path: str | Path) -> list[dict]:
    """Lit agents.json, retourne la liste [{name, description}, ...]."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["agents"]


def load_corpus(path: str | Path) -> list[dict]:
    """Lit corpus.jsonl, retourne [{message, agent}, ...] (lignes vides ignorées)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
