#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires partagés entre les scripts d'entraînement, d'évaluation et d'inférence.

Les variables d'environnement ROCm/CUDA sont positionnées ici (niveau module),
avant le premier import torch, afin de garantir leur prise en compte quel que
soit le script appelant — à condition d'importer ce module en premier.
"""

import json
import os
from typing import Any, Dict, List

# --- Variables d'environnement GPU (avant tout import torch) -------------------
_ALLOC_CONF = "expandable_segments:True"
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOC_CONF)   # ROCm
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)  # CUDA (inoffensif)
# ROCm : override GFX version pour iGPU AMD (ex. Ryzen 680M / GFX1030).
# Surcharger cette valeur dans le shell si votre GPU est différent.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
from datasets import Dataset, DatasetDict

# --- Prompts par défaut --------------------------------------------------------

SYSTEM_PROMPT_DEFAULT = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
USER_PREFIX_DEFAULT = (
    "Quelles sont les expressions clés contenues à l'identique dans ce texte : "
)

# --- Conversion texte ----------------------------------------------------------

def messages_to_text(ex: Dict[str, Any]) -> Dict[str, str]:
    """Convertit une liste de messages (role/content) en texte brut balisé."""
    msgs = ex.get("messages", [])
    text = "\n".join(
        f"<|{m.get('role', 'user')}|>: {m.get('content', '')}" for m in msgs
    )
    return {"text": text}


def clean_chunk_text(text: str) -> str:
    """Supprime l'artefact b\"...\" produit par certaines sérialisations Python."""
    if text.startswith('b"') and text.endswith('"'):
        return text[2:-1]
    return text


def chunk_to_text(
    chunk_text: str,
    expressions: List[str],
    system_prompt: str,
    user_prefix: str,
) -> str:
    """Formate un chunk + ses expressions en exemple texte pour l'entraînement."""
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": f"{user_prefix}{chunk_text}"},
        {"role": "assistant", "content": json.dumps(expressions, ensure_ascii=False)},
    ]
    return messages_to_text({"messages": messages})["text"]


# --- Construction du dataset texte --------------------------------------------

def _map_dataset(ds, fn, *, batched: bool, remove_columns, num_proc: int):
    if num_proc and num_proc > 1:
        return ds.map(fn, batched=batched, num_proc=num_proc, remove_columns=remove_columns)
    return ds.map(fn, batched=batched, remove_columns=remove_columns)


def build_text_dataset(
    ds: DatasetDict,
    system_prompt: str,
    user_prefix: str,
    num_proc: int,
) -> DatasetDict:
    """
    Accepte deux formats JSONL :
      - colonne ``messages`` (liste role/content)  → transformée directement.
      - colonne ``chunks``   (liste de dicts chunk/expressions_caracteristiques)
        → chaque chunk devient un exemple, en ignorant les chunks sans expression.
    """
    cols = ds["train"].column_names

    if "messages" in cols:
        remove_cols = [c for c in cols if c != "text"]
        return _map_dataset(
            ds, messages_to_text,
            batched=False, remove_columns=remove_cols, num_proc=num_proc,
        )

    if "chunks" in cols:
        def chunks_to_text(batch):
            texts = []
            titres = batch.get("titre", [None] * len(batch["chunks"]))
            for chunks, titre in zip(batch["chunks"], titres):
                for ch in chunks:
                    chunk_text = clean_chunk_text(ch.get("chunk", ""))
                    if titre:
                        chunk_text = f"{titre}\n\n{chunk_text}"
                    expressions = ch.get("expressions_caracteristiques", [])
                    if not expressions:
                        continue
                    texts.append(
                        chunk_to_text(chunk_text, expressions, system_prompt, user_prefix)
                    )
            return {"text": texts}

        return _map_dataset(
            ds, chunks_to_text,
            batched=True, remove_columns=cols, num_proc=num_proc,
        )

    raise ValueError(
        "Format de données inconnu : colonnes attendues 'messages' ou 'chunks'."
    )


# --- Tokenisation + masquage des labels ---------------------------------------

def build_tokenize_and_label_fn(tokenizer, assistant_tag: str, max_len: int):
    """
    Retourne une fonction de tokenisation qui :
      - tronque à ``max_len`` tokens,
      - masque tout (labels = -100) sauf la réponse assistant (après ``assistant_tag``).
    La DERNIÈRE occurrence de ``assistant_tag`` est retenue comme début de cible,
    ce qui permet aux prompts multi-tours de fonctionner correctement.
    """
    tpl_ids: List[int] = tokenizer.encode(assistant_tag, add_special_tokens=False)

    def tok_and_mask(batch):
        t = tokenizer(batch["text"], truncation=True, max_length=max_len)
        labels = []
        has_label = []
        for ids, attn in zip(t["input_ids"], t["attention_mask"]):
            lab = [-100] * len(ids)
            last = -1
            L = len(tpl_ids)
            for j in range(0, len(ids) - L + 1):
                if ids[j : j + L] == tpl_ids:
                    last = j
            if last >= 0:
                start = last + L
                end = max(i for i, a in enumerate(attn) if a == 1) + 1
                lab[start:end] = ids[start:end]
            labels.append(lab)
            has_label.append(any(v != -100 for v in lab))
        t["labels"] = labels
        t["has_label"] = has_label
        return t

    return tok_and_mask


# --- Packing longueur fixe ----------------------------------------------------

def pack_constant_length(ds_split: Dataset, max_len: int) -> Dataset:
    """Concatène tous les tokens puis redécoupe en blocs de ``max_len`` tokens."""
    big_ids, big_mask, big_lab = [], [], []
    for rec in ds_split:
        big_ids.extend(rec["input_ids"])
        big_mask.extend(rec["attention_mask"])
        big_lab.extend(rec["labels"])
    L = min(len(big_ids), len(big_mask), len(big_lab))
    L = (L // max_len) * max_len
    chunks = []
    for i in range(0, L, max_len):
        chunks.append({
            "input_ids":      big_ids [i : i + max_len],
            "attention_mask": big_mask[i : i + max_len],
            "labels":         big_lab [i : i + max_len],
        })
    return Dataset.from_list(chunks)


# --- Collateur ----------------------------------------------------------------

class SimpleCausalCollator:
    """
    Aligne (pad) les champs ``input_ids``, ``attention_mask`` et ``labels``
    à la même longueur dans un batch.
    Les positions de padding sont masquées avec -100 dans ``labels``.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        feats_wo_labels = [
            {k: v for k, v in f.items() if k != "labels"} for f in features
        ]
        batch = self.tok.pad(
            feats_wo_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
