#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sélection de passages centraux (PacSum sur embeddings BGE-M3) pour comprimer une
source à un budget de caractères, en vue d'une génération phi-4.
clean_text/split_sentences sont copiés depuis ~/indexation_wiki40b/chunk_data.py."""
import os
import re
from typing import Callable, Optional

import numpy as np
import nltk

_MULTI_NL_RE = re.compile(r"\n{3,}")
_MULTI_SP_RE = re.compile(r"[ \t]{2,}")

LAMBDA1 = float(os.environ.get("PACSUM_LAMBDA1", "-0.2"))  # phrases précédentes (j<i)
LAMBDA2 = float(os.environ.get("PACSUM_LAMBDA2", "1.0"))   # phrases suivantes (j>i), biais position
BETA = float(os.environ.get("PACSUM_BETA", "0.0"))          # seuil soustrait aux similarités


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_NL_RE.sub("\n\n", s)
    s = _MULTI_SP_RE.sub(" ", s)
    return s.strip()


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> list[str]:
    _ensure_punkt()
    try:
        sents = nltk.sent_tokenize(text, language="french")
    except LookupError:
        sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s and s.strip()]


def pacsum_scores(embeddings: np.ndarray, lambda1: float = LAMBDA1,
                  lambda2: float = LAMBDA2, beta: float = BETA) -> np.ndarray:
    sim = embeddings @ embeddings.T            # cosinus (vecteurs normalisés)
    np.fill_diagonal(sim, 0.0)
    e = sim - beta
    n = e.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        back = e[i, :i].sum() if i > 0 else 0.0
        fwd = e[i, i + 1:].sum() if i < n - 1 else 0.0
        scores[i] = lambda1 * back + lambda2 * fwd
    return scores


_prod_model = None


def _bge_m3_embed(sentences: list[str]) -> np.ndarray:
    global _prod_model
    if _prod_model is None:
        from sentence_transformers import SentenceTransformer
        _prod_model = SentenceTransformer("BAAI/bge-m3")
    return _prod_model.encode(sentences, convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)


def select_central_passages(text: str, budget_chars: int = 4000,
                            embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= budget_chars:
        return cleaned
    sentences = split_sentences(cleaned)
    if len(sentences) <= 1:
        return cleaned[:budget_chars]
    embed = embed_fn or _bge_m3_embed
    embeddings = embed(sentences)
    scores = pacsum_scores(embeddings)
    order = np.argsort(-scores)                # score décroissant
    chosen: list[int] = []
    total = 0
    for idx in order:
        s = sentences[int(idx)]
        if chosen and total + len(s) + 1 > budget_chars:
            break
        chosen.append(int(idx))
        total += len(s) + 1
    chosen.sort()                              # restituer l'ordre d'origine
    return " ".join(sentences[i] for i in chosen)
