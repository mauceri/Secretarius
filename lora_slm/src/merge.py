#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusionne un adaptateur LoRA avec le modèle de base, puis (optionnellement)
produit des fichiers GGUF quantifiés via llama.cpp.

Exemple — fusion seule :
    python src/merge.py \\
        --base  models/phi4-mini \\
        --lora  checkpoints/phi4-mini-lora-v1 \\
        --out   output/phi4-mini_merged/merged_hf

Exemple — fusion + GGUF :
    python src/merge.py \\
        --base           models/phi4-mini \\
        --lora           checkpoints/phi4-mini-lora-v1 \\
        --out            output/phi4-mini_merged/merged_hf \\
        --gguf-dir       output/phi4-mini_merged/gguf \\
        --llama-cpp      /chemin/vers/llama.cpp \\
        --quantize-types Q4_K_M Q5_K_M Q6_K
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    return logging.getLogger("merge")


def merge_lora(base: str, lora: str, out: str, dtype: str = "float16") -> None:
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    log.info(f"[merge] Chargement base={base}  dtype={dtype}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch_dtype,
        device_map="cpu",
        local_files_only=True,
    )
    log.info(f"[merge] Chargement LoRA depuis {lora}")
    lora_model = PeftModel.from_pretrained(base_model, lora, local_files_only=True)
    merged = lora_model.merge_and_unload()

    os.makedirs(out, exist_ok=True)
    log.info(f"[merge] Sauvegarde dans {out}")
    merged.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=True)
    tok.save_pretrained(out)
    log.info("[merge] Terminé.")


def _run(cmd: List[str], cwd: str = None) -> None:
    log.info("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def quantize_gguf(
    merged_dir: str, gguf_dir: str, llama_cpp: str, qtypes: List[str]
) -> None:
    llama_cpp      = llama_cpp.rstrip("/")
    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    quant_bin      = os.path.join(llama_cpp, "build/bin/llama-quantize")

    if not os.path.isfile(convert_script):
        raise FileNotFoundError(f"{convert_script} introuvable dans {llama_cpp}")
    if not os.path.isfile(quant_bin):
        raise FileNotFoundError(
            f"{quant_bin} introuvable. Compilez llama.cpp d'abord."
        )

    os.makedirs(gguf_dir, exist_ok=True)
    f16_path = os.path.join(gguf_dir, "model-f16.gguf")

    log.info(f"[gguf] Conversion HF → GGUF (f16) → {f16_path}")
    _run(["python3", convert_script, "--outtype", "f16", "--outfile", f16_path, merged_dir])

    for q in qtypes:
        out_path = os.path.join(gguf_dir, f"model-{q}.gguf")
        log.info(f"[gguf] Quantization {q} → {out_path}")
        _run([quant_bin, f16_path, out_path, q])
    log.info("[gguf] Terminé.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fusion LoRA + quantization GGUF optionnelle."
    )
    p.add_argument("--base",  required=True, help="Modèle de base (HF local).")
    p.add_argument("--lora",  required=True, help="Checkpoint LoRA (sortie Trainer).")
    p.add_argument("--out",   required=True, help="Répertoire HF fusionné en sortie.")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--log-file", default="logs/merge.log")
    p.add_argument("--gguf-dir",
                   help="Répertoire cible pour les GGUF. Absent = pas de quantization.")
    p.add_argument("--llama-cpp",
                   help="Racine de llama.cpp (avec convert_hf_to_gguf.py et llama-quantize).")
    p.add_argument("--quantize-types", nargs="+", default=[],
                   help="Ex : Q4_K_M Q5_K_M Q6_K. Requiert --gguf-dir et --llama-cpp.")
    return p.parse_args()


# Logger global (initialisé dans main après lecture des args)
log: logging.Logger = None  # type: ignore


def main():
    global log
    args = parse_args()
    log  = _setup_logger(args.log_file)

    merge_lora(args.base, args.lora, args.out, args.dtype)

    if args.gguf_dir:
        if not args.llama_cpp:
            raise ValueError("--llama-cpp est requis pour produire des GGUF.")
        quantize_gguf(args.out, args.gguf_dir, args.llama_cpp, args.quantize_types)
    else:
        log.info("[info] Pas de --gguf-dir fourni, aucune quantization réalisée.")


if __name__ == "__main__":
    main()
