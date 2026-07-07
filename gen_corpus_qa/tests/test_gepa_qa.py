import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from promptGenGEPA_qa import note_paire


def test_note_factuelle_suit_le_juge():
    assert note_paire(5, "factuelle", "Phi-4-mini.") == 1.0
    assert note_paire(3, "factuelle", "Phi-4-mini.") == 0.6


def test_hors_document_avec_refus_ok():
    assert note_paire(5, "hors_document", "Cette information ne figure pas dans le document.") == 1.0


def test_hors_document_sans_refus_penalise():
    # réponse inventée au lieu d'un refus -> plafonnée à 0.2
    assert note_paire(5, "hors_document", "Il fait 22 degrés à Paris.") == 0.2
