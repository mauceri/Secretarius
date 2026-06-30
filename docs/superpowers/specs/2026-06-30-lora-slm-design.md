# Design — lora_slm : pipeline LoRA par agent pour Tiron

**Date :** 2026-06-30
**Statut :** approuvé
**But :** créer `Secretarius/lora_slm/`, un pipeline automatisé train → merge → GGUF associant un adaptateur LoRA à chaque agent Tiron. Premier cas : agent routeur Tiron (phi-4-mini-instruct).

---

## 1. Contexte et finalité

Chaque agent Tiron sera associé à un adaptateur LoRA fine-tuné sur son propre corpus. Le corpus du routeur Tiron (2000 exemples ChatML, produit par `gen_corpus/`) est prêt. `lora_slm/` orchestre l'entraînement sur sanroque (AMD iGPU gfx900 / ROCm) et exporte un GGUF pour llama.cpp.

Base : `lora_local_AMD` (dépôt existant) — `lora_local_train.py` et `merge_and_quantize.py` ont été validés sur sanroque pour l'extracteur phi-4-mini. Ils sont repris avec adaptations minimales.

---

## 2. Structure

```
lora_slm/
├── lora_train.py              # lora_local_train.py renommé, constantes extracteur retirées
├── merge_and_quantize.py      # copié tel quel de lora_local_AMD
├── requirements-rocm.txt      # copié tel quel (torch 2.5.1 ROCm 6.2)
├── run.sh                     # orchestration : train → merge → gguf (arg : AGENT)
└── agents/
    └── tiron/
        └── config.env         # chemins + hyperparamètres spécifiques Tiron
```

Artefacts exclus du dépôt (`.gitignore`) : `checkpoints/`, `merged/`, `*.gguf`.

---

## 3. Composants

### `lora_train.py`

Repris de `lora_local_train.py` (lora_local_AMD). Adaptations :
- Retrait de `SYSTEM_PROMPT_DEFAULT` et `USER_PREFIX_DEFAULT` (spécifiques à l'extracteur — inutiles quand le corpus a la colonne `messages`)
- Renommage du fichier

Le format `messages` (ChatML) est déjà géré par `build_text_dataset` → `messages_to_text`. Le masquage des labels cible la séquence après `<|assistant|>:`. Aucune modification de logique.

Variables d'environnement ROCm posées au démarrage :
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`

### `merge_and_quantize.py`

Copié tel quel. Fusionne le checkpoint LoRA avec le modèle de base, exporte en GGUF via `convert_hf_to_gguf.py` + `llama-quantize`.

### `agents/tiron/config.env`

```bash
# Chemins
MODEL_PATH=$HOME/Modèles/phi4
CORPUS_TRAIN=$HOME/Secretarius/gen_corpus/corpus_lora_train.jsonl
OUTPUT_DIR=$HOME/lora_slm/checkpoints/tiron
MERGED_DIR=$HOME/lora_slm/merged/tiron
GGUF_DIR=$HOME/Modèles
LLAMA_CPP_PATH=$HOME/llama.cpp
QUANTIZE_TYPES="Q6_K"

# Hyperparamètres
MAX_LEN=256
EPOCHS=3
LORA_R=8
LORA_ALPHA=16
LR=2e-5
PER_DEVICE_BATCH=1
GRAD_ACCUM=16
```

Justifications :
- `MAX_LEN=256` : exemples courts (~80 tokens prompt, ~20 tokens réponse)
- `LORA_R=8` : tâche de routing plus contrainte que l'extraction, rang réduit suffit
- `EPOCHS=3` : convergence attendue rapide sur 1800 exemples

### `run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

AGENT=${1:?Usage: ./run.sh <agent>}
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/agents/$AGENT/config.env"

echo "=== [1/2] Entraînement LoRA — $AGENT ==="
python "$DIR/lora_train.py" \
  --model_path       "$MODEL_PATH" \
  --data_file        "$CORPUS_TRAIN" \
  --output_dir       "$OUTPUT_DIR" \
  --max_len          "${MAX_LEN}" \
  --epochs           "${EPOCHS}" \
  --lora_r           "${LORA_R}" \
  --lora_alpha       "${LORA_ALPHA}" \
  --lr               "${LR}" \
  --per_device_batch "${PER_DEVICE_BATCH}" \
  --grad_accum       "${GRAD_ACCUM}" \
  --log_file         "$OUTPUT_DIR/training.log"

echo "=== [2/2] Fusion + export GGUF — $AGENT ==="
python "$DIR/merge_and_quantize.py" \
  --base             "$MODEL_PATH" \
  --lora             "$OUTPUT_DIR" \
  --out              "$MERGED_DIR" \
  --gguf-dir         "$GGUF_DIR" \
  --llama-cpp        "$LLAMA_CPP_PATH" \
  --quantize-types   $QUANTIZE_TYPES

mv "$GGUF_DIR/model-Q6_K.gguf" "$GGUF_DIR/$AGENT-router-Q6_K.gguf"
echo "=== GGUF produit : $GGUF_DIR/$AGENT-router-Q6_K.gguf ==="
```

Lancement : `./run.sh tiron`

Pour GPU externe : seuls `PER_DEVICE_BATCH`, `GRAD_ACCUM`, `LORA_R` changent dans `config.env`.

---

## 4. Dépendances et environnement

- Python 3.9+, venv dédié (à créer depuis `requirements-rocm.txt`)
- `requirements-rocm.txt` consolidé : PyTorch 2.5.1 ROCm 6.2 + `transformers`, `peft`, `datasets`, `accelerate` (le `requirements.txt` de lora_local_AMD est un pip freeze complet incluant des packages CUDA inutiles — non repris)
- Modèle de base : `~/Modèles/phi4/` (poids HF phi-4-mini-instruct)
- llama.cpp : `~/llama.cpp/` (`llama-quantize` compilé, `convert_hf_to_gguf.py` présent)

---

## 5. Extensibilité

Pour un nouvel agent : créer `agents/<agent>/config.env` avec ses propres chemins et hyperparamètres. `run.sh` ne change pas. Le GGUF produit est nommé `<agent>-router-Q6_K.gguf`.
