#!/usr/bin/env bash
# openclaw-config/install.sh — Génère ~/.openclaw/ depuis les templates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../install.conf"

OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
FORCE="${FORCE:-false}"

TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*"; }

mkdir -p "$OPENCLAW_PATH"

# openclaw.json
TARGET="${OPENCLAW_PATH}/openclaw.json"
if [[ -f "$TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw.json existe déjà — ignoré (utilisez --force pour écraser)"
else
  export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND}' \
    < "${SCRIPT_DIR}/openclaw.json.template" \
    > "$TARGET"
  info "openclaw.json généré dans ${OPENCLAW_PATH}"
fi

# gateway.systemd.env
ENV_TARGET="${OPENCLAW_PATH}/gateway.systemd.env"
if [[ -f "$ENV_TARGET" && "$FORCE" != "true" ]]; then
  info "gateway.systemd.env existe déjà — ignoré"
else
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — gateway.systemd.env généré avec des valeurs vides"
  fi
  export TELEGRAM_BOT_TOKEN GATEWAY_TOKEN GATEWAY_PASSWORD
  envsubst '${TELEGRAM_BOT_TOKEN} ${GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env généré (600)"
fi

# Service systemd user
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
SERVICE_TARGET="${SYSTEMD_USER_DIR}/openclaw-gateway.service"
if [[ -f "$SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw-gateway.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/openclaw-gateway.service" "$SERVICE_TARGET"
  info "Service systemd installé dans ${SYSTEMD_USER_DIR}"
fi
