# router_service/test_laya_classifier.py
from router_service.laya_classifier import LayaClassifier


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_classify_returns_top_command_and_probability(monkeypatch):
    clf = LayaClassifier.__new__(LayaClassifier)  # évite de charger un vrai modèle
    clf.checkpoint = "test"

    def fake_query(self, message):
        return {
            "answers": {
                "command": {
                    "value": "/q",
                    "probabilities": {"/q": 0.91, "/r": 0.05, "aucune": 0.04},
                }
            }
        }

    monkeypatch.setattr(LayaClassifier, "_query", fake_query)

    command, prob = clf.classify("qu'est-ce que le SPLADE ?")

    assert command == "/q"
    assert prob == 0.91


def test_classify_returns_none_when_top_choice_is_aucune(monkeypatch):
    clf = LayaClassifier.__new__(LayaClassifier)
    clf.checkpoint = "test"

    def fake_query(self, message):
        return {
            "answers": {
                "command": {
                    "value": "aucune",
                    "probabilities": {"/q": 0.10, "aucune": 0.80},
                }
            }
        }

    monkeypatch.setattr(LayaClassifier, "_query", fake_query)

    command, prob = clf.classify("il fait beau aujourd'hui")

    assert command is None
    assert prob == 0.0
