import json
from pathlib import Path

from router_service.eval_laya import evaluate


class _FakeClassifier:
    def __init__(self, answers):
        self._answers = answers  # dict: message -> (command, prob)

    def classify(self, message):
        return self._answers[message]


def test_evaluate_computes_accuracy(tmp_path):
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("\n".join([
        json.dumps({"state": "a", "answers": {"command": "/q"}}),
        json.dumps({"state": "b", "answers": {"command": "/r"}}),
        json.dumps({"state": "c", "answers": {"command": "aucune"}}),
    ]) + "\n", encoding="utf-8")
    fake = _FakeClassifier({
        "a": ("/q", 0.9),
        "b": ("/q", 0.6),   # faux
        "c": (None, 0.0),
    })

    accuracy = evaluate(fake, val_path)

    assert accuracy == 2 / 3
