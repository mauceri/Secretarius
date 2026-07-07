import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _load(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_domaines_pointent_des_documents_existants():
    for d in _load("domaines.json"):
        assert (BASE / d["document"]).exists(), f"document manquant: {d['document']}"
        assert d["types_question"], "types_question vide"


def test_seed_schema_et_ids_connus():
    domaines = {d["domaine"] for d in _load("domaines.json")}
    seed = _load("seed.json")
    assert len(seed) >= 9
    for ex in seed:
        assert set(ex) == {"document_id", "type_question", "registre", "question", "answer"}
        assert ex["document_id"] in domaines
        assert ex["question"].strip() and ex["answer"].strip()


def test_seed_contient_des_exemples_negatifs():
    seed = _load("seed.json")
    negatifs = [e for e in seed if e["type_question"] == "hors_document"]
    assert len(negatifs) >= 3, "il faut des exemples hors_document (refus)"
