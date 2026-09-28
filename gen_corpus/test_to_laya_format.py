import json
from pathlib import Path

from gen_corpus.to_laya_format import COMMAND_CRITERIA, reformat


def test_command_criteria_covers_all_router_commands():
    from router_service.router import WIKI_CMDS, GOG_CMDS
    known = WIKI_CMDS | GOG_CMDS | {"aucune"}
    assert set(COMMAND_CRITERIA) == known


def test_reformat_converts_one_example_with_command(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "ignoré"},
            {"role": "user", "content": "cherche transformers dans le wiki"},
            {"role": "assistant", "content": json.dumps(
                {"command": "/r", "args": "transformers"})},
        ]
    }) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    n_train, n_val = reformat(src, train_path, val_path, val_ratio=0.0)

    assert n_train == 1
    assert n_val == 0
    row = json.loads(train_path.read_text(encoding="utf-8").strip())
    assert row["state"] == "cherche transformers dans le wiki"
    assert row["questions"]["command"]["type"] == "choice"
    assert row["questions"]["command"]["criteria"] == COMMAND_CRITERIA
    assert row["answers"]["command"] == "/r"


def test_reformat_converts_null_command_to_aucune(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "ignoré"},
            {"role": "user", "content": "il fait beau aujourd'hui"},
            {"role": "assistant", "content": json.dumps(
                {"command": None, "args": ""})},
        ]
    }) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    reformat(src, train_path, val_path, val_ratio=0.0)

    row = json.loads(train_path.read_text(encoding="utf-8").strip())
    assert row["answers"]["command"] == "aucune"


def test_reformat_splits_train_and_val_deterministically(tmp_path):
    src = tmp_path / "corpus.jsonl"
    lines = []
    for i in range(20):
        lines.append(json.dumps({
            "messages": [
                {"role": "system", "content": "ignoré"},
                {"role": "user", "content": f"message {i}"},
                {"role": "assistant", "content": json.dumps(
                    {"command": "/q", "args": f"message {i}"})},
            ]
        }))
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    n_train, n_val = reformat(src, train_path, val_path, val_ratio=0.2, seed=0)

    assert n_train == 16
    assert n_val == 4
    # déterministe : un deuxième run avec la même seed donne la même coupe
    train_path2 = tmp_path / "train2.jsonl"
    val_path2 = tmp_path / "val2.jsonl"
    reformat(src, train_path2, val_path2, val_ratio=0.2, seed=0)
    assert train_path.read_text() == train_path2.read_text()
    assert val_path.read_text() == val_path2.read_text()
