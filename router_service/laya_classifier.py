"""Classifieur de commandes Tiron basé sur Laya (décision calibrée, une
seule passe) — remplace l'appel génératif à phi-4-mini+LoRA. Cf.
docs/superpowers/specs/2026-09-28-routeur-laya-design.md."""
from __future__ import annotations

from gen_corpus.to_laya_format import COMMAND_CRITERIA

_INSTRUCTIONS = "Quelle commande Tiron ce message déclenche-t-il ?"


class LayaClassifier:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._agent = self._load(checkpoint)

    def _load(self, checkpoint: str):
        import laya
        return laya.load(checkpoint)

    def _query(self, message: str) -> dict:
        questions = {
            "command": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": COMMAND_CRITERIA,
            }
        }
        return self._agent.predict(message, questions)

    def classify(self, message: str) -> tuple[str | None, float]:
        result = self._query(message)
        answer = result["answers"]["command"]
        value = answer["choice"]
        if value == "aucune":
            return None, 0.0
        return value, answer["confidence"]
