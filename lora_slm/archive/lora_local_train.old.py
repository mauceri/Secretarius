#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA-local trainer (ROCm-friendly) for chat-style extraction datasets.

- Loads a JSONL with {"messages": [{"role": "...","content":"..."} , ...]}
- Formats conversations with role tags <|system|>, <|user|>, <|assistant|>
- Uses TRL's SFTTrainer + a completion-only collator (assistant-only loss)
- Applies LoRA via PEFT (memory efficient)
- Adds ROCm/HIP safe defaults (disable flash paths, set HIP allocator knobs)

Usage (tmux):
  tmux new -s lora && \
  PYTORCH_HIP_ALLOC_CONF='expandable_segments:True' \
  HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  AMD_SERIALIZE_KERNEL=3 \
  python lora_local_train.py \
    --model-path models/phi4 \
    --data-file data/train.jsonl \
    --output-dir checkpoints/phi4-lora-v1

Requires: transformers, trl (>=0.23), datasets, peft, torch (ROCm build).
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

# --- Set ROCm-friendly env defaults if not already set ---
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
os.environ.setdefault("PYTORCH_SDPA_ENABLE_HEURISTIC", "0")
os.environ.setdefault("PYTORCH_SDPA_ALLOW_MATH", "1")
os.environ.setdefault("PYTORCH_SDPA_ENABLE_FLASH", "0")
os.environ.setdefault("PYTORCH_SDPA_ENABLE_MEM_EFFICIENT", "0")
# HSA_OVERRIDE_GFX_VERSION is system-dependent (680M -> 10.3.0 usually)
# Leave it to the user or kernel.json; uncomment to force here:
# os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch

try:
    from datasets import load_dataset
except Exception as e:
    print("ERROR: `datasets` is required. pip install datasets", file=sys.stderr)
    raise

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        DataCollatorWithPadding,
        PreTrainedTokenizerBase,
    )
except Exception as e:
    print("ERROR: `transformers` is required. pip install transformers", file=sys.stderr)
    raise

try:
    from trl import SFTTrainer
except Exception as e:
    print("ERROR: `trl` is required (>=0.23). pip install -U trl", file=sys.stderr)
    raise

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception as e:
    print("ERROR: `peft` is required. pip install peft", file=sys.stderr)
    raise

# -------------------- Completion-only collator --------------------
from dataclasses import dataclass

@dataclass
class CompletionOnlyCollator:
    tokenizer: PreTrainedTokenizerBase
    response_template: str
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._tpl_ids: List[int] = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        self._base = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self._base(features)
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        labels = input_ids.clone()
        labels[:] = -100  # mask all by default

        tpl = torch.tensor(self._tpl_ids, device=input_ids.device)
        tlen = tpl.numel()

        # Unmask only the part AFTER the LAST occurrence of the response template
        for i in range(input_ids.size(0)):
            row = input_ids[i]
            last = -1
            for j in range(0, row.numel() - tlen + 1):
                if torch.equal(row[j:j+tlen], tpl):
                    last = j
            if last >= 0:
                start = last + tlen
                valid = (attn[i] == 1).nonzero(as_tuple=False).flatten()
                end = valid[-1].item() + 1 if len(valid) else row.numel()
                labels[i, start:end] = row[start:end]

        batch["labels"] = labels
        return batch


def build_argparser():
    p = argparse.ArgumentParser(description="LoRA local trainer (TRL + ROCm safe)")
    p.add_argument("--model-path", type=str, required=True, help="Local HF model directory")
    p.add_argument("--data-file", type=str, required=True, help="Path to JSONL dataset (messages format)")
    p.add_argument("--output-dir", type=str, default="checkpoints/lora-out", help="Where to save checkpoints/adapters")

    # Tokenization / lengths
    p.add_argument("--max-len", type=int, default=768, help="Max sequence length")
    p.add_argument("--packing", action="store_true", help="(ignored on TRL>=0.23; kept for compatibility)")

    # Train hyperparams
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)  # Higher LR OK for LoRA
    p.add_argument("--logging-steps", type=int, default=50)

    # Precision / memory
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.set_defaults(bf16=True)
    p.add_argument("--fp16", action="store_true", help="Use float16 (often unstable on ROCm)")

    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    p.set_defaults(gradient_checkpointing=True)

    # LoRA config
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument("--target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj,fc1,fc2",
                   help="Comma-separated module name fragments to apply LoRA to")

    # Formatting
    p.add_argument("--assistant-tag", type=str, default="<|assistant|>:", help="Response template that precedes the label span")
    p.add_argument("--role-template", type=str, default="<|{role}|>: {content}",
                   help="How to render each message into text")

    # Dataset streaming/sharding (optional)
    p.add_argument("--num-proc", type=int, default=1, help="map num_proc for tokenization")

    return p


def format_messages(example: Dict[str, Any], role_template: str) -> str:
    parts = []
    for m in example["messages"]:
        parts.append(role_template.format(role=m["role"], content=m["content"]))
    return "\n".join(parts)


def main():
    args = build_argparser().parse_args()

    # ---- Load dataset (expects JSONL with "messages") ----
    data_files = args.data_file
    ds = load_dataset("json", data_files=data_files)
    train = ds["train"]

    # ---- Tokenizer & model ----
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None,
    )
    # training-time settings
    base.config.use_cache = False
    try:
        base.config.attn_implementation = "eager"
    except Exception:
        pass
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    # ---- Apply LoRA ----
    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lconf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
    )
    model = get_peft_model(base, lconf)
    model.print_trainable_parameters()

    # ---- Build text field then tokenize ----
    def to_text(batch):
        return {"text": [format_messages(ex, args.role_template) for ex in batch["messages"]]}

    text_ds = train.map(
        lambda ex: {"text": format_messages(ex, args.role_template)},
        remove_columns=train.column_names,
    )

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)
    tok_ds = text_ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=args.num_proc)

    # ---- Collator: assistant-only labels ----
    collator = CompletionOnlyCollator(tokenizer=tok, response_template=args.assistant_tag)

    # ---- TrainingArguments ----
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
#        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        train_dataset=tok_ds,
        args=targs,
        formatting_func=None,          # already tokenized
        processing_class=tok,          # modern TRL API
        data_collator=collator,
    )

    torch.cuda.empty_cache()
    trainer.train()

    # Save only the LoRA adapter (PEFT)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    # Save tokenizer for completeness
    tok.save_pretrained(args.output_dir)
    print("\nDone. LoRA adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
