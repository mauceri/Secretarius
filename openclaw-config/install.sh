#!/usr/bin/env bash
# openclaw-config/install.sh — Génère ~/.openclaw/ depuis les templates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source la config par défaut
source "${SCRIPT_DIR}/../install.conf"

# Variables d'environnement ou valeurs par défaut
OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"

# Secrets (doivent être définis)
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"

if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
  echo "ERREUR: TELEGRAM_BOT_TOKEN non défini" >&2
  echo "Définissez-le via --env-file ou variable d'environnement" >&2
  exit 1
fi

echo "→ Génération de ${OPENCLAW_PATH}..."

mkdir -p "$OPENCLAW_PATH"

# Générer openclaw.json
envsubst '${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND}' \
  < "${SCRIPT_DIR}/openclaw.json.template" \
  > "${OPENCLAW_PATH}/openclaw.json"

# Générer gateway.systemd.env (permissions 600)
envsubst '${TELEGRAM_BOT_TOKEN} ${GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
  < "${SCRIPT_DIR}/gateway.systemd.env.template" \
  > "${OPENCLAW_PATH}/gateway.systemd.env"
chmod 600 "${OPENCLAW_PATH}/gateway.systemd.env"

# Installer le service systemd user
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
cp "${SCRIPT_DIR}/openclaw-gateway.service" "$SYSTEMD_USER_DIR/"

echo "→ Configuration OpenClaw générée dans ${OPENCLAW_PATH}"
echo "→ Pour activer le service :"
echo "   systemctl --user daemon-reload"
echo "   systemctl --user enable --now openclaw-gateway.service"
