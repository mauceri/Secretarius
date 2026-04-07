#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement d'un adaptateur LoRA sur un modèle causal (ROCm/CUDA/CPU).

Le script est agnostique au modèle de base : passez simplement le chemin
via ``--model_path``. Les modules cibles du LoRA sont configurables via
``--target_modules`` (liste séparée par des virgules).

Exemple minimal :
    python src/train.py \\
        --model_path models/phi4-mini \\
        --data_file  data/train.jsonl \\
        --output_dir checkpoints/phi4-mini-lora-v1

Exemple avec plusieurs options :
    python src/train.py \\
        --model_path  models/mistral-7b \\
        --data_file   data/train.jsonl \\
        --output_dir  checkpoints/mistral-lora-v1 \\
        --assistant_tag "<|im_start|>assistant" \\
        --target_modules q_proj,k_proj,v_proj,o_proj \\
        --max_len 1024 --epochs 3 --bf16
"""

import os
import argparse

# common.py positionne les variables d'env ROCm/CUDA avant tout import torch.
from common import (
    SYSTEM_PROMPT_DEFAULT,
    USER_PREFIX_DEFAULT,
    build_text_dataset,
    build_tokenize_and_label_fn,
    pack_constant_length,
    SimpleCausalCollator,
    _map_dataset,
)

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


# --- Callback de log de la perte ----------------------------------------------

class LossLoggerCallback(TrainerCallback):
    """Écrit la perte (et le LR) dans stdout et dans un fichier texte."""

    def __init__(self, log_path: str, every: int = 1):
        self.log_path = log_path
        self.every = max(1, every)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        line = f"step={state.global_step} loss={logs['loss']:.4f}"
        if "learning_rate" in logs:
            line += f" lr={logs['learning_rate']:.2e}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fh.close()
        except Exception:
            pass


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraînement LoRA — agnostique au modèle, ROCm/CUDA/CPU."
    )

    # Chemins
    p.add_argument("--model_path",  required=True,
                   help="Répertoire local du modèle de base (HF format).")
    p.add_argument("--data_file",   default="data/train.jsonl",
                   help="JSONL d'entraînement (format 'messages' ou 'chunks').")
    p.add_argument("--output_dir",  default="checkpoints/lora-run",
                   help="Répertoire de sortie pour l'adaptateur et les checkpoints.")
    p.add_argument("--log_file",    default="logs/training.log",
                   help="Fichier texte pour les pertes pas-à-pas.")

    # Format de données / prompt
    p.add_argument("--assistant_tag", default="<|assistant|>:",
                   help="Balise de début de la réponse assistant dans le texte.")
    p.add_argument("--system_prompt", default=SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--user_prefix",   default=USER_PREFIX_DEFAULT)

    # Tokenisation
    p.add_argument("--max_len",   type=int, default=512)
    p.add_argument("--num_proc",  type=int, default=4,
                   help="Parallélisme pour le mapping dataset (0 = désactivé).")
    p.add_argument("--packing",    action="store_true",  dest="packing",
                   help="Active le constant-length packing.")
    p.add_argument("--no-packing", action="store_false", dest="packing",
                   help="Désactive le constant-length packing.")
    p.set_defaults(packing=True)

    # Entraînement
    p.add_argument("--epochs",           type=int,   default=1)
    p.add_argument("--per_device_batch", type=int,   default=1)
    p.add_argument("--grad_accum",       type=int,   default=16)
    p.add_argument("--lr",               type=float, default=2e-5)
    p.add_argument("--optimizer",        default="adamw_torch")
    p.add_argument("--save_strategy",    default="epoch",
                   choices=["epoch", "steps", "no"])
    p.add_argument("--save_steps",       type=int, default=0)
    p.add_argument("--log_every",        type=int, default=1,
                   help="Nombre de steps entre deux logs de perte.")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--resume_from",      default=None,
                   help="Reprendre depuis un checkpoint existant.")

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Modules cibles du LoRA, séparés par des virgules. "
             "Adapter selon l'architecture du modèle.",
    )

    # Dispositif / dtype
    p.add_argument("--bf16",        action="store_true", default=False,
                   help="bfloat16 si supporté, sinon bascule en float16.")
    p.add_argument("--gpu_mem_gib", type=int, default=12,
                   help="Budget VRAM (GiB) pour device_map=auto.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device",      default="auto", choices=["auto", "cuda", "cpu"])

    return p.parse_args()


# --- Main ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    have_cuda = torch.cuda.is_available()
    on_hip    = getattr(torch.version, "hip", None) is not None
    if have_cuda:
        print("GPU : CUDA disponible")
    elif on_hip:
        print("GPU : ROCm disponible")
    else:
        print("GPU : aucun accélérateur détecté — entraînement sur CPU")

    # 1) Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, local_files_only=True
    )
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # 2) Dataset → texte → tokenisation + labels → (packing optionnel)
    ds      = load_dataset("json", data_files=args.data_file)
    ds_txt  = build_text_dataset(ds, args.system_prompt, args.user_prefix, args.num_proc)
    tok_fn  = build_tokenize_and_label_fn(tok, args.assistant_tag, args.max_len)
    rm_cols = [c for c in ds_txt["train"].column_names if c != "text"]
    ds_tok  = DatasetDict({
        k: _map_dataset(ds_txt[k], tok_fn, batched=True,
                        remove_columns=rm_cols, num_proc=args.num_proc)
        for k in ds_txt.keys()
    })
    ds_tok = DatasetDict({
        k: ds_tok[k].filter(lambda x: x["has_label"]).remove_columns(["has_label"])
        for k in ds_tok.keys()
    })

    if args.packing:
        train_dataset = pack_constant_length(ds_tok["train"], args.max_len)
    else:
        train_dataset = ds_tok["train"]

    # 3) Modèle
    device_map = args.device if args.device != "auto" else "auto"
    if args.device == "cuda" and not have_cuda:
        print("CUDA demandé mais indisponible — bascule en CPU.")
        device_map = "cpu"
        have_cuda  = False

    bf16_ok   = have_cuda and args.bf16 and _bf16_supported()
    torch_dtype = (
        torch.bfloat16 if bf16_ok
        else torch.float16 if have_cuda
        else torch.float32
    )
    if args.bf16 and have_cuda and not bf16_ok:
        print("bf16 demandé mais non supporté — bascule en float16.")

    max_memory = {0: f"{args.gpu_mem_gib}GiB", "cpu": "256GiB"} if have_cuda else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        max_memory=max_memory,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    model.config.use_cache = False

    # 4) LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5) Entraînement
    collator    = SimpleCausalCollator(tok, pad_to_multiple_of=8)
    targs_kw    = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.log_every,
        logging_strategy="steps",
        save_strategy=args.save_strategy,
        report_to="none",
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        optim=args.optimizer,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        seed=args.seed,
    )
    if args.save_strategy == "steps":
        targs_kw["save_steps"] = args.save_steps
    # Compat transformers ancienne/nouvelle API
    try:
        targs_kw["evaluation_strategy"] = "no"
        targs = TrainingArguments(**targs_kw)
    except TypeError:
        targs_kw.pop("evaluation_strategy", None)
        targs_kw["eval_strategy"] = "no"
        targs = TrainingArguments(**targs_kw)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tok,
        callbacks=[LossLoggerCallback(args.log_file, args.log_every)],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from)
    print(result)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)


def _bf16_supported() -> bool:
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


if __name__ == "__main__":
    main()
