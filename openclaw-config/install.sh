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
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

# Si gateway.systemd.env existe déjà (réinstallation), lire les secrets
EXISTING_ENV="${OPENCLAW_PATH}/gateway.systemd.env"
if [[ -f "$EXISTING_ENV" ]]; then
  while IFS='=' read -r key val; do
    key="${key#"${key%%[! ]*}"}"   # strip leading spaces
    key="${key%"${key##*[! ]}"}"   # strip trailing spaces
    key="${key%$'\r'}"             # strip CR (CRLF)
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -z "${!key:-}" ]] && declare "$key=$val"
  done < <(grep -v '^#' "$EXISTING_ENV")
fi

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*"; }

# Détecter le binaire openclaw (NVM, npm global, ou chemin par défaut)
OPENCLAW_BIN=$(command -v openclaw 2>/dev/null || true)
if [[ -z "$OPENCLAW_BIN" ]]; then
  NPM_PREFIX=$(npm prefix -g 2>/dev/null || true)
  [[ -n "$NPM_PREFIX" ]] && OPENCLAW_BIN="${NPM_PREFIX}/bin/openclaw"
fi
OPENCLAW_BIN="${OPENCLAW_BIN:-/usr/bin/openclaw}"
export OPENCLAW_BIN

# Migration GATEWAY_TOKEN → OPENCLAW_GATEWAY_TOKEN
if [[ -z "${OPENCLAW_GATEWAY_TOKEN:-}" && -n "${GATEWAY_TOKEN:-}" ]]; then
  warn "GATEWAY_TOKEN est déprécié — utiliser OPENCLAW_GATEWAY_TOKEN dans gateway.systemd.env"
  OPENCLAW_GATEWAY_TOKEN="$GATEWAY_TOKEN"
fi

# Générer OPENCLAW_GATEWAY_TOKEN si absent (token HTTP du gateway, pas lié à Telegram)
if [[ -z "${OPENCLAW_GATEWAY_TOKEN:-}" ]]; then
  OPENCLAW_GATEWAY_TOKEN="$(openssl rand -hex 32)"
  info "OPENCLAW_GATEWAY_TOKEN généré automatiquement"
fi

mkdir -p "$OPENCLAW_PATH"

# openclaw.json
TARGET="${OPENCLAW_PATH}/openclaw.json"
if [[ -f "$TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw.json existe déjà — ignoré (utilisez --force pour écraser)"
else
  export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND DEEPSEEK_API_KEY OPENCLAW_GATEWAY_TOKEN
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND} ${DEEPSEEK_API_KEY} ${OPENCLAW_GATEWAY_TOKEN}' \
    < "${SCRIPT_DIR}/openclaw.json.template" \
    > "$TARGET"
  info "openclaw.json généré dans ${OPENCLAW_PATH}"
fi

# gateway.systemd.env
ENV_TARGET="${OPENCLAW_PATH}/gateway.systemd.env"
if [[ -f "$ENV_TARGET" && "$FORCE" != "true" ]]; then
  info "gateway.systemd.env existe déjà — ignoré"
elif [[ "$FORCE" != "true" ]]; then
  # Premier passage : TELEGRAM_BOT_TOKEN vide (à renseigner), OPENCLAW_GATEWAY_TOKEN auto-généré
  TELEGRAM_BOT_TOKEN="" GATEWAY_PASSWORD="" \
    envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env généré (600) — renseigner TELEGRAM_BOT_TOKEN et DEEPSEEK_API_KEY avant de démarrer le service"
else
  # Passage --force : injecter les secrets (depuis le fichier existant ou --env-file)
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — à renseigner dans ${ENV_TARGET}"
  fi
  export TELEGRAM_BOT_TOKEN OPENCLAW_GATEWAY_TOKEN GATEWAY_PASSWORD
  envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env mis à jour (600)"
fi

# Service systemd user
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
SERVICE_TARGET="${SYSTEMD_USER_DIR}/openclaw-gateway.service"
if [[ -f "$SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw-gateway.service existe déjà — ignoré"
else
  envsubst '${OPENCLAW_BIN}' \
    < "${SCRIPT_DIR}/openclaw-gateway.service" \
    > "$SERVICE_TARGET"
  info "Service systemd installé dans ${SYSTEMD_USER_DIR} (${OPENCLAW_BIN})"
fi

# Workspace .md et skills
WORKSPACE_SRC="${SCRIPT_DIR}/workspace"
WORKSPACE_DST="${OPENCLAW_PATH}/workspace"
export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME
SUBST_VARS='${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME}'

while IFS= read -r -d '' src; do
  rel="${src#${WORKSPACE_SRC}/}"
  dst="${WORKSPACE_DST}/${rel}"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$FORCE" != "true" ]]; then
    info "${rel} existe déjà — ignoré"
  else
    envsubst "$SUBST_VARS" < "$src" > "$dst"
    info "${rel} installé"
  fi
done < <(find "$WORKSPACE_SRC" -name "*.md" -print0)
