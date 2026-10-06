"""Classifieur Laya à deux niveaux (domaine puis commande), une seule passe,
probabilités jointes — cf. docs/superpowers/specs/2026-10-06-routage-deux-niveaux-design.md."""
from __future__ import annotations

from router_service.router import COMMANDES, DOMAINE_UNIQUE, QUESTIONS


def classement(answers: dict) -> list[tuple[str | None, float]]:
    dom = answers["domaine"]["probabilities"]
    out: list[tuple[str | None, float]] = []
    for d, p_d in dom.items():
        if d == "aucune":
            out.append((None, round(p_d, 4)))
        elif d in DOMAINE_UNIQUE:
            out.append((DOMAINE_UNIQUE[d], round(p_d, 4)))
        elif d in COMMANDES:
            sous = answers.get(f"commande_{d}", {}).get("probabilities", {})
            for c, p_c in sous.items():
                out.append((c, round(p_d * p_c, 4)))
    return sorted(out, key=lambda x: -x[1])


class LayaClassifier:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._agent = self._load(checkpoint)

    def _load(self, checkpoint: str):
        import laya
        return laya.load(checkpoint)

    def classer(self, message: str) -> list[tuple[str | None, float]]:
        return classement(self._agent.predict(message, QUESTIONS)["answers"])
