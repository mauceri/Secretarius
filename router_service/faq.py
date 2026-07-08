#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAQ de faits locaux : parse faits.md et retrouve l'entrée la plus proche
d'un message (single-vector nearest-neighbor sur BGE-M3). Réponse = corps
verbatim de l'entrée, aucun appel LLM."""
import os
from pathlib import Path
import torch

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


class FaqIndex:
    def __init__(self, embed_fn, path=FAQ_PATH, seuil=SEUIL_FAQ):
        self._embed_fn = embed_fn
        self._path = Path(path)
        self._seuil = seuil
        self._mtime = None
        self._qmat = None       # torch.Tensor [N_questions, D] ou None
        self._entry_of = []     # entrée correspondant à chaque question
        self._reload()

    def _current_mtime(self):
        try:
            return self._path.stat().st_mtime
        except OSError:
            return None

    def _reload(self):
        self._mtime = self._current_mtime()
        if self._mtime is None:
            self._qmat, self._entry_of = None, []
            return
        entries = parse_faq(self._path.read_text(encoding="utf-8"))
        questions, entry_of = [], []
        for e in entries:
            for q in e["questions"]:
                questions.append(q)
                entry_of.append(e)
        self._entry_of = entry_of
        self._qmat = self._embed_fn(questions) if questions else None

    def lookup(self, message: str):
        if self._current_mtime() != self._mtime:
            self._reload()
        if self._qmat is None:
            return None
        e = self._embed_fn([message])                 # [1, D], normalisé
        sims = (e @ self._qmat.T).squeeze(0)          # [N_questions]
        idx = int(sims.argmax())
        if float(sims[idx]) >= self._seuil:
            return self._entry_of[idx]
        return None
