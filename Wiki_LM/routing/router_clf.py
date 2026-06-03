"""Routeur par tête de classification sur embeddings BGE-M3 gelés.

Le modèle d'embedding reste gelé ; on entraîne seulement une régression logistique
(rapide, CPU) sur (embedding → agent). Même convention que EmbedRouter : clarify
exclu de l'entraînement, atteint via le seuil de confiance.
"""
from __future__ import annotations

import numpy as np

from router_base import Router, RouteResult
from router_embed import _default_encode


class ClfRouter(Router):
    def __init__(self, clf, threshold: float = 0.55, encode_fn=_default_encode):
        self.clf = clf
        self.threshold = threshold
        self.encode_fn = encode_fn

    @classmethod
    def from_corpus(cls, train: list[dict], threshold: float = 0.55,
                    encode_fn=_default_encode, exclude=("clarify",)):
        from sklearn.linear_model import LogisticRegression
        msgs: list[str] = []
        labels: list[str] = []
        for row in train:
            if row["agent"] in exclude:
                continue
            msgs.append(row["message"])
            labels.append(row["agent"])
        X = encode_fn(msgs)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, labels)
        return cls(clf, threshold, encode_fn)

    def route(self, message: str) -> RouteResult:
        vec = self.encode_fn([message])  # forme (1, d)
        proba = self.clf.predict_proba(vec)[0]
        best = int(np.argmax(proba))
        score = float(proba[best])
        if score < self.threshold:
            return RouteResult("clarify", score)
        return RouteResult(str(self.clf.classes_[best]), score)
