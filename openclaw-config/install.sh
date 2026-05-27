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
for _arg in "$@"; do
  [[ "$_arg" == "--force" ]] && FORCE="true"
done

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
    envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD} ${OPENCLAW_BIN} ${HOME}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env généré (600) — renseigner TELEGRAM_BOT_TOKEN et DEEPSEEK_API_KEY avant de démarrer le service"
else
  # Passage --force : injecter les secrets (depuis le fichier existant ou --env-file)
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — à renseigner dans ${ENV_TARGET}"
  fi
  export TELEGRAM_BOT_TOKEN OPENCLAW_GATEWAY_TOKEN GATEWAY_PASSWORD OPENCLAW_BIN
  envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD} ${OPENCLAW_BIN} ${HOME}' \
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

# Service scout (watcher)
SCOUT_SERVICE_TARGET="${SYSTEMD_USER_DIR}/openclaw-scout.service"
SCOUT_WATCHER_TARGET="${HOME}/.local/bin/scout-watcher"
mkdir -p "${HOME}/.local/bin"
if [[ -f "$SCOUT_SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw-scout.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/openclaw-scout.service" "$SCOUT_SERVICE_TARGET"
  info "openclaw-scout.service installé dans ${SYSTEMD_USER_DIR}"
fi
if [[ -f "$SCOUT_WATCHER_TARGET" && "$FORCE" != "true" ]]; then
  info "scout-watcher existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/scout-watcher" "$SCOUT_WATCHER_TARGET"
  chmod +x "$SCOUT_WATCHER_TARGET"
  info "scout-watcher installé dans ${HOME}/.local/bin"
fi

# Injection guard
GUARD_TARGET="${HOME}/.local/bin/injection_guard.py"
GUARD_PROCESS_TARGET="${HOME}/.local/bin/scout_process.py"
GUARD_SERVICE_TARGET="${SYSTEMD_USER_DIR}/openclaw-injection-guard.service"

if [[ -f "$GUARD_TARGET" && "$FORCE" != "true" ]]; then
  info "injection_guard.py existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/injection_guard.py" "$GUARD_TARGET"
  chmod +x "$GUARD_TARGET"
  info "injection_guard.py installé dans ${HOME}/.local/bin"
fi

if [[ -f "$GUARD_PROCESS_TARGET" && "$FORCE" != "true" ]]; then
  info "scout_process.py existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/scout_process.py" "$GUARD_PROCESS_TARGET"
  chmod +x "$GUARD_PROCESS_TARGET"
  info "scout_process.py installé dans ${HOME}/.local/bin"
fi

if [[ -f "$GUARD_SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw-injection-guard.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/openclaw-injection-guard.service" "$GUARD_SERVICE_TARGET"
  info "openclaw-injection-guard.service installé dans ${SYSTEMD_USER_DIR}"
fi

# Dépendances Python pour injection-guard
if python3 -c "import flask, bs4, requests" &>/dev/null; then
  info "flask, beautifulsoup4, requests déjà installés"
else
  info "Installation des dépendances Python (flask, beautifulsoup4, requests)..."
  pip install --user --quiet flask beautifulsoup4 requests || \
    warn "pip install échoué — relancer manuellement : pip install flask beautifulsoup4 requests"
fi

# transformers et torch (optionnel — slow download, ~1GB)
if python3 -c "import transformers" &>/dev/null; then
  info "transformers déjà installé"
else
  info "Installation de transformers et torch (peut prendre plusieurs minutes)..."
  pip install --user --quiet transformers torch || \
    warn "pip install transformers échoué — DeBERTa désactivé, mode regex-only"
fi

# fastmcp (dépendance du serveur MCP Wiki_LM)
if "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3" -c "import fastmcp" &>/dev/null; then
  info "fastmcp déjà installé dans le venv Wiki_LM"
else
  info "Installation de fastmcp dans le venv Wiki_LM..."
  "${HOME}/Secretarius/Wiki_LM/.venv/bin/pip" install --quiet fastmcp || \
    warn "pip install fastmcp échoué — relancer manuellement dans le venv Wiki_LM"
fi

# Recharger et démarrer injection-guard si TELEGRAM_BOT_TOKEN est renseigné
if [[ -n "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now openclaw-injection-guard.service 2>/dev/null && \
    info "openclaw-injection-guard.service démarré" || \
    warn "Démarrage automatique échoué — lancer manuellement : systemctl --user start openclaw-injection-guard.service"
else
  info "TELEGRAM_BOT_TOKEN absent — openclaw-injection-guard.service non démarré automatiquement"
  info "Démarrer avec : systemctl --user start openclaw-injection-guard.service"
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

# Workspace scout (isolé dans agents/scout/workspace/)
SCOUT_WORKSPACE_SRC="${SCRIPT_DIR}/agents/scout/workspace"
SCOUT_WORKSPACE_DST="${OPENCLAW_PATH}/agents/scout/workspace"
mkdir -p "${SCOUT_WORKSPACE_DST}/tasks/pending" "${SCOUT_WORKSPACE_DST}/tasks/done" "${SCOUT_WORKSPACE_DST}/results"

while IFS= read -r -d '' src; do
  rel="${src#${SCOUT_WORKSPACE_SRC}/}"
  dst="${SCOUT_WORKSPACE_DST}/${rel}"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$FORCE" != "true" ]]; then
    info "scout/${rel} existe déjà — ignoré"
  else
    envsubst "$SUBST_VARS" < "$src" > "$dst"
    info "scout/${rel} installé"
  fi
done < <(find "$SCOUT_WORKSPACE_SRC" -name "*.md" -print0)

# Image Docker sandbox
SANDBOX_IMAGE="openclaw-sandbox:bookworm-slim"
SANDBOX_DOCKERFILE="${SCRIPT_DIR}/sandbox/Dockerfile"
if docker ps &>/dev/null 2>&1; then
  if docker image inspect "$SANDBOX_IMAGE" &>/dev/null 2>&1; then
    info "Image Docker ${SANDBOX_IMAGE} déjà présente — ignorée"
  else
    info "Construction de l'image Docker ${SANDBOX_IMAGE} (peut prendre 1-2 minutes)..."
    SANDBOX_DIR="$(dirname "$SANDBOX_DOCKERFILE")"
    if docker build -t "$SANDBOX_IMAGE" "$SANDBOX_DIR"; then
      info "Image ${SANDBOX_IMAGE} construite ✓"
    else
      warn "Échec de la construction de l'image Docker — relancer manuellement :"
      warn "  docker build -t ${SANDBOX_IMAGE} ${SANDBOX_DIR}"
    fi
  fi
else
  info "Docker inaccessible — image sandbox non construite (relancer install.sh après reconnexion)"
fi
