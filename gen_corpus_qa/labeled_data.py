#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construit un jeu étiqueté {wiki, gog, secretarius, null} à partir des corpus
existants, avec centroïde et test disjoints pour secretarius."""
import json
import random
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CORPUS_QA = _HERE / "corpus_qa.jsonl"                 # questions secretarius
CORPUS_ROUTEUR = _HERE.parent / "gen_corpus" / "corpus.jsonl"

WIKI_INT = {"wiki_capture", "wiki_ingest", "wiki_status", "wiki_query", "source_read"}
GOG_INT = {"gog_search", "gog_connect", "gog_inbox", "gog_reply", "gog_drive"}
NULL_VAR = {"aide_generale", "conversation_libre"}


def build_labeled_data(n_centroid=60, seed=42, n_par_classe=90, n_null=60):
    rng = random.Random(seed)

    questions = list(dict.fromkeys([json.loads(l)["question"] for l in open(CORPUS_QA, encoding="utf-8")
                                    if l.strip()]))
    rng.shuffle(questions)
    centroid = questions[:n_centroid]
    test_sec = [(q, "secretarius") for q in questions[n_centroid:n_centroid + n_par_classe]]

    wiki, gog, null = [], [], []
    for l in open(CORPUS_ROUTEUR, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        intent, var, txt = r["intention"], r.get("variante"), r["text"]
        if intent in WIKI_INT:
            wiki.append((txt, "wiki"))
        elif intent in GOG_INT:
            gog.append((txt, "gog"))
        elif intent == "out_of_scope" and var in NULL_VAR:
            null.append((txt, "null"))

    rng.shuffle(wiki)
    rng.shuffle(gog)
    rng.shuffle(null)
    test = test_sec + wiki[:n_par_classe] + gog[:n_par_classe] + null[:n_null]
    rng.shuffle(test)
    return {"centroid": centroid, "test": test}
