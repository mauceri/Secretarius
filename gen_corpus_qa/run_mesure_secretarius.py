#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harnais de validation locale : matrice de confusion du classifieur +
qualité des réponses phi-4 nu. Écrit RESULTATS_SECRETARIUS.md."""
import random
from pathlib import Path

from labeled_data import build_labeled_data
from classify_secretarius import SecretariusClassifier
from repondre_secretarius import repondre_secretarius
from mesure_secretarius import confusion_matrix, taux_commandes_volees, rappel
from eval_qa import judge_score

TEST_BASE_URL = "http://127.0.0.1:8996"
N_ECHANTILLON_REPONSE = 20


def main():
    data = build_labeled_data(n_centroid=60, seed=42)
    clf = SecretariusClassifier(data["centroid"])

    # 1) Classification du jeu de test complet
    pairs = [(lab, clf.classify(txt)) for txt, lab in data["test"]]
    m = confusion_matrix(pairs)
    voles = taux_commandes_volees(m)
    rap_sec = rappel(m, "secretarius")

    # 2) Qualité des réponses phi-4 nu sur un échantillon secretarius
    sec_txts = [txt for txt, lab in data["test"] if lab == "secretarius"]
    rng = random.Random(42)
    rng.shuffle(sec_txts)
    ech = sec_txts[:N_ECHANTILLON_REPONSE]
    doc = (Path(__file__).resolve().parent / "documents" / "secretarius.md").read_text(encoding="utf-8")
    notes, apercus = [], []
    for q in ech:
        rep = repondre_secretarius(q, base_url=TEST_BASE_URL)
        note = judge_score(doc, q, rep) / 5.0
        notes.append(note)
        apercus.append((q, rep, note))
    note_moy = sum(notes) / len(notes) if notes else 0.0

    # 3) Écriture du verdict
    lignes = ["# Verdict — détection & réponse question-Secretarius", "",
              "## Détection (classifieur centroïde)", "",
              "| vrai \\ prédit | wiki | gog | secretarius | null |",
              "|---|---|---|---|---|"]
    for t in ["wiki", "gog", "secretarius", "null"]:
        r = m[t]
        lignes.append(f"| {t} | {r['wiki']} | {r['gog']} | {r['secretarius']} | {r['null']} |")
    lignes += ["",
               f"- Rappel secretarius : **{rap_sec:.3f}**",
               f"- Taux de commandes wiki/gog détournées vers secretarius : **{voles:.3f}**",
               "",
               "## Réponse (phi-4 nu + document)", "",
               f"- Note moyenne juge DeepSeek sur {len(ech)} questions : **{note_moy:.3f}**",
               "", "### Aperçus", ""]
    for q, rep, note in apercus:
        lignes.append(f"- [{note:.1f}] Q: {q!r}\n  R: {rep[:200]!r}")
    Path(__file__).resolve().parent.joinpath("RESULTATS_SECRETARIUS.md").write_text(
        "\n".join(lignes), encoding="utf-8")
    print(f"rappel_sec={rap_sec:.3f} voles={voles:.3f} note_rep={note_moy:.3f}")


if __name__ == "__main__":
    main()
