# router_service/test_laya_classifier.py
from router_service.laya_classifier import LayaClassifier


def test_classify_returns_top_command_and_confidence(monkeypatch):
    clf = LayaClassifier.__new__(LayaClassifier)  # évite de charger un vrai modèle
    clf.checkpoint = "test"

    def fake_query(self, message):
        return {"answers": {"command": {"choice": "/q", "confidence": 0.91}}}

    monkeypatch.setattr(LayaClassifier, "_query", fake_query)

    command, confidence = clf.classify("qu'est-ce que le SPLADE ?")

    assert command == "/q"
    assert confidence == 0.91


def test_classify_returns_none_when_top_choice_is_aucune(monkeypatch):
    clf = LayaClassifier.__new__(LayaClassifier)
    clf.checkpoint = "test"

    def fake_query(self, message):
        return {"answers": {"command": {"choice": "aucune", "confidence": 0.80}}}

    monkeypatch.setattr(LayaClassifier, "_query", fake_query)

    command, confidence = clf.classify("il fait beau aujourd'hui")

    assert command is None
    assert confidence == 0.0
