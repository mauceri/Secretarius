"""Reformate gen_corpus/corpus_lora_train.jsonl (ChatML) vers le schéma Laya
(state/questions/answers) — cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md.
Aucune collecte de données neuve : seule la mise en forme change, et
`args` du corpus d'origine n'est pas repris (le routeur n'en génère plus)."""
from __future__ import annotations

import json
import random
from pathlib import Path

COMMAND_CRITERIA: dict[str, str] = {
    "/c": "capturer une note ou une URL dans le wiki",
    "/q": "poser une question au wiki, réponse synthétisée",
    "/ingest": "lancer l'ingestion des sources en attente",
    "/source": "déléguer une recherche web à Scout",
    "/wikistatus": "connaître l'état de l'ingestion du wiki",
    "/r": "rechercher par mots-clés dans le wiki, sans synthèse",
    "/tags": "lister les tags du wiki",
    "/kbupdate": "mettre à jour la base de connaissances du wiki",
    "/supprimer": "supprimer une page du wiki",
    "/relire": "obtenir la prochaine page du wiki à relire",
    "/verifie": "marquer une page du wiki comme vérifiée",
    "/chercher": "rechercher dans les emails Gmail",
    "/connecter": "démarrer la connexion au compte Google",
    "/inbox": "lister les nouveaux emails",
    "/drive": "rechercher dans Google Drive",
    "/repondre": "répondre à un email",
    "/lire": "lire le contenu d'un email",
    "aucune": "aucune commande ne correspond, message hors sujet",
}

_INSTRUCTIONS = "Quelle commande Tiron ce message déclenche-t-il ?"


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
