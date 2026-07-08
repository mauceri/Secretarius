#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classifieur à 4 classes {wiki, gog, secretarius, null} réutilisant le GogGate
BGE-M3 (centroïdes wiki/gog/null) et lui ajoutant un centroïde secretarius.
Ne modifie pas router_service/router.py."""
import torch
from router_service.router import GogGate

SEUIL_SECRETARIUS = 0.5


class SecretariusClassifier:
    def __init__(self, questions_secretarius, gate=None, seuil=SEUIL_SECRETARIUS):
        self.gate = gate if gate is not None else GogGate()
        self.seuil = seuil
        # centroïde secretarius = moyenne normalisée des embeddings des questions
        self._cent_sec = self.gate._embed(questions_secretarius).mean(0, keepdim=True)

    def classify(self, message: str) -> str:
        e = self.gate._embed([message])                       # [1,1024]
        sims = (e @ self.gate._cmat.T).squeeze(0)             # [wiki, gog, null]
        sim_wiki, sim_gog, sim_null = (float(sims[0]), float(sims[1]), float(sims[2]))
        sim_sec = float((e @ self._cent_sec.T).squeeze())
        # priorité commande : secretarius ne gagne que s'il dépasse le seuil ET
        # est strictement supérieur aux similarités wiki et gog
        if sim_sec >= self.seuil and sim_sec > sim_wiki and sim_sec > sim_gog:
            return "secretarius"
        idx = int(torch.tensor([sim_wiki, sim_gog, sim_null]).argmax())
        return ["wiki", "gog", "null"][idx]
