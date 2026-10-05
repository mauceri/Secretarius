# gen_corpus/tests/test_gepa_metric.py
from pathlib import Path
import random
import sys
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_feedback_signale_les_defauts():
    from promptGenGEPA import construire_feedback
    fb = construire_feedback("Bonjour Tiron, pourriez-vous avoir l'amabilité de bien vouloir me rappeler "
                             "chaque lundi à huit heures précises de payer le loyer de l'appartement svp",
                             1.0, 0.4, "trop poli", 0.0, "évoque /q")
    assert "trop poli" in fb and "évoque /q" in fb
    assert "Bonjour" in fb and "mots" in fb


def test_feedback_parfait():
    from promptGenGEPA import construire_feedback
    assert construire_feedback("rappel loyer lundi 8h", 1.0, 1.0, "rien", 1.0, "ok").startswith("Parfait")


def test_metrique_renvoie_score_et_feedback(monkeypatch):
    import dspy
    import promptGenGEPA as g
    real = MagicMock(return_value=dspy.Prediction(score=5, raison="rien"))
    fid = MagicMock(return_value=dspy.Prediction(score=0, raison="évoque /q"))
    monkeypatch.setattr(g.dspy, "Predict", MagicMock(side_effect=[real, fid]))
    metric = g.make_metric(MagicMock())
    gold = dspy.Example(intention="cron_schedule")
    out = metric(gold, {"text": "rappel loyer lundi 8h", "command": "/cron", "args": "x"})
    assert isinstance(out, dspy.Prediction)
    assert out.score == 0.25 * 1.0 + 0.25 * 1.0 + 0.5 * 0.0
    assert "évoque /q" in out.feedback


def test_juge_non_numerique_ne_plante_pas(monkeypatch):
    import dspy
    import promptGenGEPA as g
    real = MagicMock(return_value=dspy.Prediction(score="très", raison="?"))
    fid = MagicMock(return_value=dspy.Prediction(score="oui", raison="?"))
    monkeypatch.setattr(g.dspy, "Predict", MagicMock(side_effect=[real, fid]))
    out = g.make_metric(MagicMock())(dspy.Example(intention="wiki_tags"), {"text": "tags ?", "command": "/tags", "args": ""})
    assert 0.0 <= out.score <= 1.0 and "illisible" in out.feedback


def test_generateur_montre_des_exemples_et_les_contraintes(monkeypatch):
    import promptGenGEPA as g
    seed = [{"text": "rappel jeudi 18h bac vert", "intention": "cron_schedule", "registre": "abrégé",
             "variante": "x", "action": {"command": "/cron", "args": "x"}}]
    gen = g.build_example_generator("Consigne de test assez longue pour être une instruction.", seed, random.Random(0))
    fake = MagicMock(return_value=MagicMock(text="t", command="/cron", args="a"))
    gen.generate = fake
    gen(intention="cron_schedule", registre="poli", variante="x", contraintes="moment : demain")
    kw = fake.call_args.kwargs
    assert kw["contraintes"] == "moment : demain" and len(kw["demos"]) == 1


def test_reflexion_peut_reecrire_toute_l_instruction():
    # Le 2026-10-05, max_tokens=512 a tronqué l'instruction proposée par la réflexion.
    import promptGenGEPA as g
    instruction = (Path(__file__).parent.parent / "prompt-init.txt").read_text(encoding="utf-8")
    assert g.REFLECTION_MAX_TOKENS >= 4 * len(instruction.split())


def test_contrat_du_prompt_mentionne_les_contraintes():
    texte = (Path(__file__).parent.parent / "prompt-init.txt").read_text(encoding="utf-8")
    assert "Entrées : intention, registre, variante, contraintes" in texte
