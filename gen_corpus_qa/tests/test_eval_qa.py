import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from eval_qa import parse_eval_row, aggregate


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
