#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# premier eta 27:17:05, 157.41s/it

from dataclasses import dataclass
from pyexpat import model
from typing import Optional, List, Dict, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch


# --------- PARAMS (les mêmes que votre notebook) ---------
MODEL_PATH = "models/phi4"
DATA_FILE = "data/train.jsonl"
OUTPUT_DIR = "checkpoints/phi4-lora-v1"

# Hyperparams
MAX_LEN = 2048
PER_DEV_BS = 1
GRAD_ACC = 16
EPOCHS = 1
LR = 2e-4
LOG_STEPS = 50
BF16 = True
FP16 = False

# Formatage
ASSISTANT_TAG = "<|assistant|>:"
ROLE_TEMPLATE = "<|{role}|>: {content}"

# LoRA (laisser comme dans le notebook si vous l’y aviez)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # adaptez si besoin
# ---------------------------------------------------------


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
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self._base(features)
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        labels = input_ids.clone()
        labels[:] = -100

        tpl = torch.tensor(self._tpl_ids, device=input_ids.device)
        tlen = tpl.numel()

        # on ne garde en labels QUE ce qui suit la DERNIÈRE occurrence du template
        for i in range(input_ids.size(0)):
            row = input_ids[i]
            last = -1
            for j in range(0, row.numel() - tlen + 1):
                if torch.equal(row[j : j + tlen], tpl):
                    last = j
            if last >= 0:
                start = last + tlen
                valid = (attn[i] == 1).nonzero(as_tuple=False).flatten()
                end = valid[-1].item() + 1 if len(valid) else row.numel()
                labels[i, start:end] = row[start:end]

        batch["labels"] = labels
        return batch


def format_messages(example: Dict[str, Any]) -> str:
    return "\n".join(
        ROLE_TEMPLATE.format(role=m["role"], content=m["content"])
        for m in example["messages"]
    )


def main():
    # Dataset
    ds = load_dataset("json", data_files=DATA_FILE)
    train = ds["train"]

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Modèle de base
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16 if BF16 and torch.cuda.is_available() else None
    )
    # -- Attention SDPA "math only" (ROCm safe)
    base.config.use_cache = False
    try:
        base.config.attn_implementation = "sdpa"  # bascule explicite sur SDPA
    except Exception:
        pass

    
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(
            enable_flash=False,        # pas de FlashAttention
            enable_math=True,          # chemin "math" stable
            enable_mem_efficient=False # pas de mem_efficient
    )  
    print("attn_implementation =", getattr(base.config, "attn_implementation", "default"))
    # Comme dans beaucoup de notebooks : désactiver le cache en training
    base.config.use_cache = False
    # (si votre notebook n’a pas de réglage d’attention, laissez tel quel)

    # LoRA
    lconf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, lconf)
    # Gradient checkpointing (si présent dans le notebook)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Linéarisation -> tokenisation
    text_ds = train.map(lambda ex: {"text": format_messages(ex)}, remove_columns=train.column_names)
    tok_ds = text_ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"],
    )

    # Collator “assistant-only”
    collator = CompletionOnlyCollator(tokenizer=tok, response_template=ASSISTANT_TAG)

    # --- Warm-up pour compiler les kernels JIT avant le timing d'entraînement ---
    device = torch.device("cuda")
    model = model.to(device)  # 'model' est déjà PEFT-isé
    model.train()             # on reste en mode train
    # prenez un batch minuscule depuis le dataset tokenisé
    sample = tok_ds[0]
    import numpy as np
    def to_tensor(s):
        return torch.tensor(np.array([s]), device=device)

    with torch.no_grad():
        _ = model(
            input_ids=to_tensor(sample["input_ids"]),
            attention_mask=to_tensor(sample["attention_mask"]),
        )
    # >>>>>>>>>>>>>>>>> EXACTEMENT comme votre notebook <<<<<<<<<<<<<<<<<
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEV_BS,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        bf16=BF16,
        fp16=FP16,
        optim="adamw_torch",
        max_grad_norm=0.0,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    trainer = SFTTrainer(
        model=model,
        train_dataset=tok_ds,
        args=args,
        formatting_func=None,   # déjà tokenisé
        processing_class=tok,   # API TRL moderne
        data_collator=collator,
    )

    torch.cuda.empty_cache()
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print("Saved LoRA adapter to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
