#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue un modèle fusionné (non quantifié) sur un corpus JSONL (data/test.jsonl).
Calcule la loss moyenne et la perplexité via Trainer.evaluate().

Exemple :
    python evaluate_merged.py \\
        --model gguf_out/phi4_merged/merged_hf \\
        --data_file data/test.jsonl \\
        --assistant_tag "<|assistant|>:" \\
        --max_len 512 \\
        --per_device_batch 1
"""

import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import argparse
import logging

import json
import json
from typing import Any, Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


# Logger vers stdout + fichier
LOG_PATH = "evaluate_merged.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"),
    ],
)
log = logging.getLogger("evaluate_merged")

SYSTEM_PROMPT_DEFAULT = (
    "Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un "
    "tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les "
    "expressions, dates et lieux remarquables, évènements, qui apparaissent "
    "à l'identique dans le texte."
)
USER_PREFIX_DEFAULT = "Quelles sont les expressions clés contenues à l'identique dans ce texte : "

def messages_to_text(ex: Dict[str, Any]) -> Dict[str, str]:
    msgs = ex.get("messages", [])
    text = "\n".join(f"<|{m.get('role','user')}|>: {m.get('content','')}" for m in msgs)
    return {"text": text}

def clean_chunk_text(text: str) -> str:
    if text.startswith('b"') and text.endswith('"'):
        return text[2:-1]
    return text

def chunk_to_text(chunk_text: str, expressions: List[str], system_prompt: str, user_prefix: str) -> str:
    user_text = f"{user_prefix}{chunk_text}"
    assistant_text = json.dumps(expressions, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return messages_to_text({"messages": messages})["text"]


def _map_dataset(ds, fn, *, batched: bool, remove_columns, num_proc: int):
    if num_proc and num_proc > 1:
        return ds.map(fn, batched=batched, num_proc=num_proc, remove_columns=remove_columns)
    return ds.map(fn, batched=batched, remove_columns=remove_columns)

def build_text_dataset(ds, system_prompt: str, user_prefix: str, num_proc: int):
    cols = ds["train"].column_names
    if "messages" in cols:
        remove_cols = [c for c in cols if c != "text"]
        return _map_dataset(ds, messages_to_text, batched=False, remove_columns=remove_cols, num_proc=num_proc)
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
                    texts.append(chunk_to_text(chunk_text, expressions, system_prompt, user_prefix))
            return {"text": texts}
        return _map_dataset(ds, chunks_to_text, batched=True, remove_columns=cols, num_proc=num_proc)
    raise ValueError("Format de données inconnu: colonnes attendues 'messages' ou 'chunks'.")


def build_tokenize_and_label_fn(tokenizer, assistant_tag: str, max_len: int):
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
                if ids[j:j+L] == tpl_ids:
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


class SimpleCausalCollator:
    """Pad (input_ids, attention_mask, labels) à la même longueur."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        feats_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
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


def parse_args():
    p = argparse.ArgumentParser("Évaluation d'un modèle fusionné (HF) sur un JSONL.")
    p.add_argument("--model", required=True, help="Répertoire du modèle fusionné (HF).")
    p.add_argument("--data_file", type=str, default="data/test.jsonl")
    p.add_argument("--assistant_tag", type=str, default="<|assistant|>:", help="Début de la réponse.")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=2)
    p.add_argument("--bf16", action="store_true", help="bf16 si supporté, sinon fp16.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--user_prefix", type=str, default=USER_PREFIX_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics_out", type=str, default="", help="Chemin fichier pour sauvegarder les metriques (JSON).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    log.info("=== Évaluation modèle fusionné ===")
    log.info(f"model={args.model} data_file={args.data_file} device={args.device} bf16={args.bf16}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Dataset
    ds = load_dataset("json", data_files=args.data_file)
    ds_txt = build_text_dataset(ds, args.system_prompt, args.user_prefix, args.num_proc)
    tok_and_mask = build_tokenize_and_label_fn(tok, args.assistant_tag, args.max_len)
    remove_cols = [c for c in ds_txt["train"].column_names if c != "text"]
    ds_tok = _map_dataset(ds_txt, tok_and_mask, batched=True, remove_columns=remove_cols, num_proc=args.num_proc)
    eval_dataset = ds_tok["train"].filter(lambda x: x["has_label"]).remove_columns(["has_label"])
    log.info(f"Dataset chargé : {len(eval_dataset)} exemples")

    # dtype et device
    have_cuda = torch.cuda.is_available()
    bf16_supported = False
    if have_cuda:
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False
    use_bf16 = args.bf16 and have_cuda and bf16_supported
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if have_cuda else torch.float32)
    device_map = args.device if args.device != "auto" else ("auto" if have_cuda else "cpu")
    log.info(f"dtype={torch_dtype} device_map={device_map}")

    # Modèle
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.config.use_cache = False

    # Trainer pour eval
    collator = SimpleCausalCollator(tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=os.path.join(args.model, "eval_tmp"),
        per_device_eval_batch_size=args.per_device_batch,
        report_to="none",
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tok,
    )

    log.info("Début de l'évaluation…")
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    if loss is not None:
        ppl = torch.exp(torch.tensor(loss)).item()
        log.info(f"eval_loss={loss:.4f}  perplexity={ppl:.4f}")
    log.info(f"Fin. Détails : {metrics}")

    if args.metrics_out:
        out = {
            "eval_loss": loss,
            "perplexity": float(ppl) if loss is not None else None,
            "eval_runtime": metrics.get("eval_runtime"),
            "eval_samples_per_second": metrics.get("eval_samples_per_second"),
            "eval_steps_per_second": metrics.get("eval_steps_per_second"),
            "eval_samples": len(eval_dataset),
        }
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
