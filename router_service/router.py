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
