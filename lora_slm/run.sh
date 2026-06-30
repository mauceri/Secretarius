#!/usr/bin/env bash
set -euo pipefail

AGENT=${1:?Usage: ./run.sh <agent>}
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/agents/$AGENT/config.env"

mkdir -p "$OUTPUT_DIR" "$MERGED_DIR"

echo "=== [1/2] Entraînement LoRA — $AGENT ==="
"$DIR/lenv/bin/python" "$DIR/lora_train.py" \
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
"$DIR/lenv/bin/python" "$DIR/merge_and_quantize.py" \
  --base           "$MODEL_PATH" \
  --lora           "$OUTPUT_DIR" \
  --out            "$MERGED_DIR" \
  --gguf-dir       "$GGUF_DIR" \
  --llama-cpp      "$LLAMA_CPP_PATH" \
  --quantize-types $QUANTIZE_TYPES

mv "$GGUF_DIR/model-Q6_K.gguf" "$GGUF_DIR/$AGENT-router-Q6_K.gguf"
echo "=== GGUF produit : $GGUF_DIR/$AGENT-router-Q6_K.gguf ==="
