#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAQ de faits locaux : parse faits.md et retrouve l'entrée la plus proche
d'un message (single-vector nearest-neighbor sur BGE-M3). Réponse = corps
verbatim de l'entrée, aucun appel LLM."""
import os
from pathlib import Path

FAQ_PATH = Path(os.environ.get(
    "FAQ_PATH", str(Path.home() / "Documents/Arbath/Wiki_LM/faits/faits.md")))
SEUIL_FAQ = float(os.environ.get("FAQ_SEUIL", "0.6"))
FAQ_MAX_ENTREE = int(os.environ.get("FAQ_MAX_ENTREE", "2000"))


def parse_faq(text: str) -> list[dict]:
    """Une entrée = une ou plusieurs lignes '## ...' (formulations) suivies d'un
    corps, jusqu'au prochain '## ' ou la fin. Lignes '# ...' (commentaires/H1)
    ignorées. Entrée sans corps ou dont le corps dépasse FAQ_MAX_ENTREE écartée."""
    entries: list[dict] = []
    questions: list[str] = []
    body: list[str] = []

    def flush() -> None:
        nonlocal questions, body
        answer = "\n".join(body).strip()
        if questions and answer:
            if len(answer) <= FAQ_MAX_ENTREE:
                entries.append({"questions": list(questions), "answer": answer})
            else:
                print(f"[faq] entrée ignorée (> {FAQ_MAX_ENTREE} car.): "
                      f"{questions[0]!r}", flush=True)
        questions = []
        body = []

    for line in text.splitlines():
        if line.startswith("## "):
            if body:            # corps déjà accumulé -> entrée précédente terminée
                flush()
            questions.append(line[3:].strip())
        elif line.startswith("#"):
            continue            # commentaire / titre H1
        elif line.strip() == "" and not body:
            continue            # ligne vide avant le corps
        else:
            body.append(line)
    flush()
    return entries
