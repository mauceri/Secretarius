# Pipeline de fine-tuning — Phi-4-mini extracteur d'expressions

Ce document décrit la chaîne complète pour reconstruire le modèle de production
à partir de zéro.

---

## Modèle de production

**Fichier actif :** `test_wikipedia_gguf/model-Q6_K.gguf`
**Chargé par :** llama.cpp server (port 8989, user service systemd)
**Rôle :** extraction d'expressions caractéristiques (pipeline Secretarius)

---

## Prérequis

```bash
cd ~/Secretarius/lora_local
source llenv/bin/activate

export HF_HOME=~/Secretarius/lora_local/.hf_cache
export HF_DATASETS_CACHE=~/Secretarius/lora_local/.hf_cache/datasets
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # iGPU AMD GFX900
```

---

## Étape 1 — Fine-tuning sur corpus synthétique

> Produit : `checkpoints_phi4_lora/` (phi4-lora-v1)

```bash
python lora_local_train.py \
  --model_path models/phi4 \
  --data_file data/corpus_synth_indexed_1000.jsonl \
  --output_dir checkpoints_phi4_lora \
  --epochs 3 \
  --log_file checkpoints_phi4_lora/training.log
```

---

## Étape 2 — Fusion base + LoRA v1

> Produit : `gguf_out.précieux/phi4_merged/merged_hf/` (point de départ pour étape 3)

```bash
python merge_and_quantize.py \
  --base models/phi4 \
  --lora checkpoints_phi4_lora \
  --out gguf_out.précieux/phi4_merged/merged_hf
```

---

## Étape 3 — Fine-tuning sur Wikipedia FR

> Produit : `test_wikipedia/` (checkpoint de production)

```bash
python lora_local_train.py \
  --model_path gguf_out.précieux/phi4_merged/merged_hf \
  --data_file data/corpus_wiki40b_fr_indexed_1000.jsonl \
  --output_dir test_wikipedia \
  --epochs 3 \
  --num_proc 0 \
  --log_file test_wikipedia/training.log
```

**Résultats mesurés :** loss finale ~0.95, PPL 1.30–1.45 sur données Wikipedia

---

## Étape 4 — Fusion + Quantization GGUF

> Produit : `test_wikipedia_gguf/` (modèles GGUF quantifiés)

```bash
python merge_and_quantize.py \
  --base gguf_out.précieux/phi4_merged/merged_hf \
  --lora test_wikipedia \
  --out test_wikipedia_merged \
  --gguf-dir test_wikipedia_gguf \
  --llama-cpp ~/llama.cpp \
  --quantize-types Q4_K_M Q5_K_M Q6_K
```

---

## Étape 5 — Évaluation

```bash
python evaluate_merged.py \
  --model test_wikipedia \
  --data_file data/test.jsonl \
  --assistant_tag "<|assistant|>:" \
  --max_len 512 \
  --per_device_batch 1
```

---

## Étape 6 — Déploiement

```bash
# Mettre à jour le service llama.cpp avec le nouveau GGUF
# Éditer : ~/Secretarius/deploy/systemd-user/llama_cpp.service
# Puis :
systemctl --user daemon-reload
systemctl --user restart llama_cpp
```

---

## Prompt d'entraînement (ne pas modifier)

**System prompt :**
```
Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les expressions, dates et lieux remarquables, évènements, qui apparaissent à l'identique dans le texte.
```

**User prefix :**
```
Quelles sont les expressions clés contenues à l'identique dans ce texte :
```

> ⚠️ Le modèle a été conditionné sur ce prompt exact. Toute modification dégrade les résultats.

---

## Artefacts à préserver

| Artefact | Rôle |
|----------|------|
| `models/phi4/` | Modèle de base Phi-4-mini (point de départ absolu) |
| `checkpoints_phi4_lora/` | LoRA v1 (entraîné sur corpus synthétique) |
| `gguf_out.précieux/phi4_merged/merged_hf/` | Base fusionnée (entrée de l'étape 3) |
| `test_wikipedia/` | LoRA de production (checkpoint précieux) |
| `test_wikipedia_gguf/model-Q6_K.gguf` | GGUF actif en production |
| `data/corpus_synth_indexed_1000.jsonl` | Corpus étape 1 |
| `data/corpus_wiki40b_fr_indexed_1000.jsonl` | Corpus étape 3 |
