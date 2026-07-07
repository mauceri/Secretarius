import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from eval_qa import parse_eval_row, aggregate, _ressemble_refus, run_condition


def test_parse_eval_row():
    row = {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Document:\nDOC\n\nQuestion: Q ?"},
        {"role": "assistant", "content": "Cette information ne figure pas dans le document."},
    ]}
    p = parse_eval_row(row)
    assert p["document"] == "DOC"
    assert p["question"] == "Q ?"
    assert p["reference"] == "Cette information ne figure pas dans le document."
    assert p["is_refus"] is True


def test_aggregate_moyenne():
    assert aggregate([1.0, 0.5, 0.0]) == 0.5
    assert aggregate([]) == 0.0


def test_ressemble_refus_normalise_apostrophe_typographique():
    # apostrophe typographique ' doit matcher le marqueur "n'est pas dans"
    assert _ressemble_refus("La réponse n’est pas dans le document.") is True
    assert _ressemble_refus("Phi-4-mini.") is False


def test_run_condition_ventile_refus(monkeypatch):
    import eval_qa
    monkeypatch.setattr(eval_qa, "judge_score", lambda d, q, a: 5)
    rows = [
        {"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Document:\nD\n\nQuestion: q1"},
            {"role": "assistant", "content": "Cette information ne figure pas dans le document."}]},
        {"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Document:\nD\n\nQuestion: q2"},
            {"role": "assistant", "content": "Une réponse factuelle."}]},
    ]
    out = run_condition(rows, lambda d, q: "peu importe")
    assert out["n"] == 2 and out["n_refus"] == 1 and out["n_non_refus"] == 1
    assert out["note_refus"] == 1.0 and out["note_non_refus"] == 1.0
