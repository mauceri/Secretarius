from pathlib import Path
from unittest.mock import MagicMock
import sys
import json
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_generate_one_structure():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "garde cet article https://example.com"
    mock_result.command = "/c"
    mock_result.args = "https://example.com"
    mock_predict = MagicMock(return_value=mock_result)

    entry = generate_one(mock_predict, "wiki_capture", "familier", "url_seule")

    assert entry["text"] == "garde cet article https://example.com"
    assert entry["intention"] == "wiki_capture"
    assert entry["registre"] == "familier"
    assert entry["variante"] == "url_seule"
    assert entry["action"]["command"] == "/c"
    assert entry["action"]["args"] == "https://example.com"


def test_generate_one_null_command():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "commande une pizza"
    mock_result.command = "null"
    mock_result.args = ""
    mock_predict = MagicMock(return_value=mock_result)

    entry = generate_one(mock_predict, "out_of_scope", "familier", "action_impossible")
    assert entry["action"]["command"] is None


def test_convert_entry_chatML():
    from to_lora_format import convert_entry, SYSTEM_PROMPT
    entry = {"text": "garde ce lien", "intention": "wiki_capture",
             "action": {"command": "/c", "args": "https://ex.com"}}
    result = convert_entry(entry)
    msgs = result["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1] == {"role": "user", "content": "garde ce lien"}
    assert msgs[2]["role"] == "assistant"
    parsed = json.loads(msgs[2]["content"])
    assert parsed == {"command": "/c", "args": "https://ex.com"}


def test_convert_entry_out_of_scope():
    from to_lora_format import convert_entry
    entry = {"text": "commande une pizza", "intention": "out_of_scope",
             "action": {"command": None, "args": ""}}
    parsed = json.loads(convert_entry(entry)["messages"][2]["content"])
    assert parsed["command"] is None
    assert parsed["args"] == ""


def test_split_90_10(tmp_path):
    from to_lora_format import to_lora
    entries = [{"text": f"msg {i}", "intention": "wiki_capture",
                "action": {"command": "/c", "args": f"https://ex{i}.com"}}
               for i in range(100)]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    out, train, eval_ = tmp_path / "out.jsonl", tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    to_lora(str(corpus), str(out), str(train), str(eval_))
    assert len(train.read_text().strip().splitlines()) == 90
    assert len(eval_.read_text().strip().splitlines()) == 10
    for line in out.read_text().strip().splitlines():
        msg = json.loads(line)
        assert len(msg["messages"]) == 3
