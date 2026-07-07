import sys, types
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from generate_corpus_qa import build_entry


class FakeResult:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


def test_build_entry_structure():
    r = FakeResult("  Quel modèle ? ", "  Phi-4-mini.  ")
    e = build_entry(r, "config", "DOC-TEXTE", "factuelle", "poli")
    assert e == {
        "document_id": "config",
        "document": "DOC-TEXTE",
        "question": "Quel modèle ?",
        "answer": "Phi-4-mini.",
        "type_question": "factuelle",
        "registre": "poli",
    }
