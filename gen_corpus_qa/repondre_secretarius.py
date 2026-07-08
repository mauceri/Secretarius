#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Répond à une question sur Secretarius avec phi-4 nu (adaptateur à scale 0)
et le document Secretarius injecté en contexte. Cible un llama-server de TEST
(8996 par défaut), jamais la prod 8998."""
from pathlib import Path
from eval_qa import set_lora_scale, infer_llama

_DOC = Path(__file__).resolve().parent / "documents" / "secretarius.md"
TEST_BASE_URL = "http://127.0.0.1:8996"


def repondre_secretarius(question: str, base_url: str = TEST_BASE_URL,
                         doc_path: Path = _DOC) -> str:
    document = Path(doc_path).read_text(encoding="utf-8")
    set_lora_scale(base_url, 0.0)   # désactive l'adaptateur routeur -> phi-4 nu
    return infer_llama(base_url, document, question)
