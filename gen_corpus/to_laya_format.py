"""Reformate gen_corpus/corpus_lora_train.jsonl (ChatML) vers le schéma Laya
(state/questions/answers) — cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md.
Aucune collecte de données neuve : seule la mise en forme change, et
`args` du corpus d'origine n'est pas repris (le routeur n'en génère plus)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Permet `python3 gen_corpus/to_laya_format.py` en exécution directe (sinon
# seul le répertoire gen_corpus/ est sur sys.path et router_service reste
# introuvable).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router_service.router import COMMAND_CRITERIA, _INSTRUCTIONS  # noqa: E402


def _iter_examples(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = row["messages"]
        user_text = msgs[-2]["content"]
        assistant = json.loads(msgs[-1]["content"])
        command = assistant.get("command") or "aucune"
        yield user_text, command


def _to_laya_row(user_text: str, command: str) -> dict:
    return {
        "state": user_text,
        "questions": {
            "command": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": COMMAND_CRITERIA,
            }
        },
        "answers": {"command": command},
    }


def reformat(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[int, int]:
    rows = [_to_laya_row(text, cmd) for text, cmd in _iter_examples(Path(input_path))]
    seen_states: set[str] = set()
    deduped_rows = []
    for row in rows:
        if row["state"] in seen_states:
            continue
        seen_states.add(row["state"])
        deduped_rows.append(row)
    rows = deduped_rows

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = round(len(rows) * val_ratio)
    val_indices = set(indices[:n_val])

    train_lines = []
    val_lines = []
    for i, row in enumerate(rows):
        line = json.dumps(row, ensure_ascii=False)
        if i in val_indices:
            val_lines.append(line)
        else:
            train_lines.append(line)

    Path(train_path).write_text(
        ("\n".join(train_lines) + "\n") if train_lines else "", encoding="utf-8"
    )
    Path(val_path).write_text(
        ("\n".join(val_lines) + "\n") if val_lines else "", encoding="utf-8"
    )
    return len(train_lines), len(val_lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    n_train, n_val = reformat(
        root / "corpus_lora_train.jsonl",
        root / "laya_train.jsonl",
        root / "laya_val.jsonl",
    )
    print(f"train={n_train} val={n_val}")


if __name__ == "__main__":
    main()
