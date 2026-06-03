"""Routeur par embeddings BGE-M3 : un prototype par agent, cosinus, argmax.

Clarify est traité comme une classe normale (prototype dédié). Le seuil ne sert de
repli que si clarify n'est pas présent dans les prototypes (mode héritage)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from router_base import Router, RouteResult

_MODEL_NAME = "BAAI/bge-m3"
_model = None


def _default_encode(texts: list[str]) -> np.ndarray:
    """Encode via BGE-M3 (chargé paresseusement), vecteurs L2-normalisés float32."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model.encode(texts, normalize_embeddings=True).astype(np.float32)


class EmbedRouter(Router):
    """Route vers l'agent au prototype le plus proche (cosinus) ; sous le seuil → clarify."""

    def __init__(self, prototypes: dict, threshold: float = 0.55, encode_fn=_default_encode):
        self.prototypes = prototypes
        self.threshold = threshold
        self.encode_fn = encode_fn
        self._agents = list(prototypes.keys())
        self._matrix = (
            np.vstack([prototypes[a] for a in self._agents])
            if prototypes else np.zeros((0, 0), dtype=np.float32)
        )

    @classmethod
    def from_corpus(cls, train: list[dict], threshold: float = 0.55,
                    encode_fn=_default_encode, exclude=()):
        """Construit un prototype par agent (moyenne L2-normalisée), hors agents exclus."""
        msgs_by_agent: dict = defaultdict(list)
        for row in train:
            if row["agent"] in exclude:
                continue
            msgs_by_agent[row["agent"]].append(row["message"])
        prototypes: dict = {}
        for agent, msgs in msgs_by_agent.items():
            vecs = encode_fn(msgs)
            proto = vecs.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            prototypes[agent] = proto.astype(np.float32)
        return cls(prototypes, threshold, encode_fn)

    def route(self, message: str) -> RouteResult:
        if not self._agents:
            return RouteResult("clarify", 0.0)
        vec = self.encode_fn([message])[0]
        sims = self._matrix @ vec  # produit scalaire = cosinus (vecteurs normalisés)
        best = int(np.argmax(sims))
        score = float(sims[best])
        # Seuil de repli seulement si clarify n'est pas déjà un prototype entraîné
        if score < self.threshold and "clarify" not in self._agents:
            return RouteResult("clarify", score)
        return RouteResult(self._agents[best], score)
