#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classification BGE-M3 à 3 centroïdes, utilisée uniquement comme portail
de confiance sur les commandes gog (l'extraction de commande se fait par
l'adaptateur unique, pas par ce module)."""
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F

WIKI_CMDS = {"/c", "/q", "/ingest", "/source", "/wikistatus"}
GOG_CMDS = {"/chercher", "/connecter", "/inbox", "/drive", "/repondre"}
# Corpus des centroïdes : relatif au dépôt (router.py est dans router_service/),
# surchargeable par GEN_CORPUS_DIR. Plus de chemin machine en dur.
_GEN_CORPUS = Path(os.environ.get(
    "GEN_CORPUS_DIR", Path(__file__).resolve().parent.parent / "gen_corpus"))
TRAIN_FULL = str(_GEN_CORPUS / "corpus_lora_train.jsonl")
RAW_CORPUS = str(_GEN_CORPUS / "corpus.jsonl")
NULL_VARIANTES = {"action_impossible", "aide_generale"}
T_SOFTMAX = 0.05
SEUIL_GOG = 0.50


def true_domain(cmd):
    if cmd in WIKI_CMDS:
        return "wiki"
    if cmd in GOG_CMDS:
        return "gog"
    return None


class GogGate:
    """Portail de confiance : True si la classification BGE-M3 confirme
    le domaine gog avec une confiance >= SEUIL_GOG."""

    def __init__(self, n_per_class=80):
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self._mdl = AutoModel.from_pretrained("BAAI/bge-m3").eval()

        buckets = {"wiki": [], "gog": []}
        for l in open(TRAIN_FULL):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            cmd = json.loads(r["messages"][-1]["content"]).get("command")
            dom = true_domain(cmd)
            if dom is not None and len(buckets[dom]) < n_per_class:
                buckets[dom].append(r["messages"][-2]["content"])

        null_texts = []
        for l in open(RAW_CORPUS):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("intention") == "out_of_scope" and r.get("variante") in NULL_VARIANTES:
                null_texts.append(r["text"])

        cent_wiki = self._embed(buckets["wiki"]).mean(0, keepdim=True)
        cent_gog = self._embed(buckets["gog"]).mean(0, keepdim=True)
        cent_null = self._embed(null_texts).mean(0, keepdim=True)
        self._cmat = torch.cat([cent_wiki, cent_gog, cent_null], 0)

    def _embed(self, texts):
        enc = self._tok(texts, padding=True, truncation=True, max_length=128,
                        return_tensors="pt")
        with torch.no_grad():
            out = self._mdl(**enc).last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=1)

    def gog_confident(self, message: str) -> bool:
        e = self._embed([message])
        sims = (e @ self._cmat.T).squeeze(0)
        probs = F.softmax(sims / T_SOFTMAX, dim=0)
        gog_is_argmax = sims[1] >= sims[0] and sims[1] >= sims[2]
        return bool(gog_is_argmax and probs[1].item() >= SEUIL_GOG)
