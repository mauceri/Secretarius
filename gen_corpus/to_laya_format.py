"""Reformate gen_corpus/corpus_lora_train.jsonl (ChatML) vers le schéma Laya à deux
niveaux : question `domaine`, plus `commande_<domaine>` pour wiki et google
(cf. docs/superpowers/specs/2026-10-06-routage-deux-niveaux-design.md). La validation
est stratifiée par commande : min(40, n // 4) exemples par commande. `args` du corpus
d'origine n'est pas repris (le routeur n'en génère plus)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Permet `python3 gen_corpus/to_laya_format.py` en exécution directe (sinon
# seul le répertoire gen_corpus/ est sur sys.path et router_service reste
# introuvable).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router_service.router import DOMAINE_DE, QUESTIONS  # noqa: E402


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
    domaine = "aucune" if command == "aucune" else DOMAINE_DE[command]  # KeyError si commande inconnue
    questions = {"domaine": QUESTIONS["domaine"]}
    answers = {"domaine": domaine}
    if f"commande_{domaine}" in QUESTIONS:
        questions[f"commande_{domaine}"] = QUESTIONS[f"commande_{domaine}"]
        answers[f"commande_{domaine}"] = command
    return {"state": user_text, "questions": questions, "answers": answers}


def reformat(input_path: Path, train_path: Path, val_path: Path, seed: int = 0) -> tuple[int, int]:
    seen_states: set[str] = set()
    groupes: dict[str, list[dict]] = {}
    for text, cmd in _iter_examples(Path(input_path)):
        if text in seen_states:
            continue
        seen_states.add(text)
        groupes.setdefault(cmd, []).append(_to_laya_row(text, cmd))

    rng = random.Random(seed)
    train_lines: list[str] = []
    val_lines: list[str] = []
    for cmd in sorted(groupes):
        rows = groupes[cmd]
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_val = min(40, len(rows) // 4)
        for k, i in enumerate(idx):
            (val_lines if k < n_val else train_lines).append(json.dumps(rows[i], ensure_ascii=False))
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

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
