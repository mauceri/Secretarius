#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue un modèle HF fusionné sur un corpus JSONL.
Calcule la loss moyenne et la perplexité via Trainer.evaluate().

Exemple :
    python src/evaluate.py \\
        --model  output/phi4-mini_merged/merged_hf \\
        --data_file data/test.jsonl \\
        --assistant_tag "<|assistant|>:" \\
        --max_len 512
"""

import argparse
import json
import logging
import os

# Positionne les variables d'env ROCm/CUDA avant tout import torch.
from common import (
    SYSTEM_PROMPT_DEFAULT,
    USER_PREFIX_DEFAULT,
    build_text_dataset,
    build_tokenize_and_label_fn,
    SimpleCausalCollator,
    _map_dataset,
)

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
    )
    return logging.getLogger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(
        description="Évaluation loss/perplexité d'un modèle HF fusionné."
    )
    p.add_argument("--model",            required=True,
                   help="Répertoire du modèle fusionné (format HF).")
    p.add_argument("--data_file",        default="data/test.jsonl")
    p.add_argument("--assistant_tag",    default="<|assistant|>:",
                   help="Balise de début de la réponse assistant.")
    p.add_argument("--max_len",          type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--num_proc",         type=int, default=2)
    p.add_argument("--bf16",             action="store_true")
    p.add_argument("--device",           default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--system_prompt",    default=SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--user_prefix",      default=USER_PREFIX_DEFAULT)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--log_file",         default="logs/evaluate.log")
    p.add_argument("--metrics_out",      default="",
                   help="Chemin JSON pour sauvegarder les métriques.")
    return p.parse_args()


def main():
    args = parse_args()
    log  = _setup_logger(args.log_file)
    torch.manual_seed(args.seed)

    log.info("=== Évaluation ===")
    log.info(f"model={args.model}  data={args.data_file}  device={args.device}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True
    )
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Dataset
    ds      = load_dataset("json", data_files=args.data_file)
    ds_txt  = build_text_dataset(ds, args.system_prompt, args.user_prefix, args.num_proc)
    tok_fn  = build_tokenize_and_label_fn(tok, args.assistant_tag, args.max_len)
    rm_cols = [c for c in ds_txt["train"].column_names if c != "text"]
    ds_tok  = _map_dataset(
        ds_txt, tok_fn, batched=True, remove_columns=rm_cols, num_proc=args.num_proc
    )
    eval_dataset = (
        ds_tok["train"].filter(lambda x: x["has_label"]).remove_columns(["has_label"])
    )
    log.info(f"Dataset : {len(eval_dataset)} exemples")

    # dtype / device
    have_cuda   = torch.cuda.is_available()
    bf16_ok     = have_cuda and args.bf16 and _bf16_supported()
    torch_dtype = (
        torch.bfloat16 if bf16_ok
        else torch.float16 if have_cuda
        else torch.float32
    )
    device_map  = args.device if args.device != "auto" else ("auto" if have_cuda else "cpu")
    log.info(f"dtype={torch_dtype}  device_map={device_map}")

    # Modèle
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.config.use_cache = False

    # Évaluation via Trainer
    collator = SimpleCausalCollator(tok, pad_to_multiple_of=8)
    targs    = TrainingArguments(
        output_dir=os.path.join(args.model, "_eval_tmp"),
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
    loss    = metrics.get("eval_loss")
    ppl     = None
    if loss is not None:
        ppl = torch.exp(torch.tensor(loss)).item()
        log.info(f"eval_loss={loss:.4f}  perplexity={ppl:.4f}")
    log.info(f"Détails : {metrics}")

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        out = {
            "eval_loss":               loss,
            "perplexity":              float(ppl) if ppl is not None else None,
            "eval_runtime":            metrics.get("eval_runtime"),
            "eval_samples_per_second": metrics.get("eval_samples_per_second"),
            "eval_steps_per_second":   metrics.get("eval_steps_per_second"),
            "eval_samples":            len(eval_dataset),
        }
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        log.info(f"Métriques sauvegardées dans {args.metrics_out}")


def _bf16_supported() -> bool:
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


if __name__ == "__main__":
    main()
