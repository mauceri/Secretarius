#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensembles de commandes du routeur Tiron, et embedding BGE-M3 partagé
(utilisé par la FAQ — cf. router_service/faq.py). Le garde-fou de confiance
gog par centroïdes (GogGate) a été retiré le 2026-09-28 : la confiance vient
désormais du score calibré du classifieur Laya (router_service/laya_classifier.py),
cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md."""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WIKI_CMDS = {"/c", "/q", "/ingest", "/source", "/wikistatus", "/r", "/tags", "/kbupdate", "/supprimer", "/relire", "/verifie"}
GOG_CMDS = {"/chercher", "/connecter", "/inbox", "/drive", "/repondre", "/lire"}

_tok = None
_mdl = None


def _load_bge_m3():
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        _mdl = AutoModel.from_pretrained("BAAI/bge-m3").eval()
    return _tok, _mdl


def embed_bge_m3(texts: list[str]) -> torch.Tensor:
    """Embeddings BGE-M3 normalisés L2 (CLS pooling) — utilisé par la FAQ."""
    tok, mdl = _load_bge_m3()
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state[:, 0]
    return F.normalize(out, p=2, dim=1)
