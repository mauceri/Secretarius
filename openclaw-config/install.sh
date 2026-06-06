#!/usr/bin/env bash
# openclaw-config/install.sh — Génère ~/.openclaw/ depuis les templates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../install.conf"

OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
FORCE="${FORCE:-false}"
PROFILE="${PROFILE:-prod}"
_i=0; _args=("$@")
while [[ $_i -lt ${#_args[@]} ]]; do
  case "${_args[$_i]}" in
    --force) FORCE="true" ;;
    --profile) _i=$((_i+1)); PROFILE="${_args[$_i]:-prod}" ;;
  esac
  _i=$((_i+1))
done
unset _i _args
if [[ "$PROFILE" == "slm" ]]; then
  OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw-slm}"
else
  OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
fi

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

# Sourcer NVM si nécessaire (sessions SSH non-interactives ne chargent pas .bashrc)
if ! command -v openclaw &>/dev/null; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh" 2>/dev/null || true
fi

# Détecter le binaire openclaw (NVM, npm global, ou chemin par défaut)
OPENCLAW_BIN=$(command -v openclaw 2>/dev/null || true)
if [[ -z "$OPENCLAW_BIN" ]]; then
  NPM_PREFIX=$(npm prefix -g 2>/dev/null || true)
  [[ -n "$NPM_PREFIX" ]] && OPENCLAW_BIN="${NPM_PREFIX}/bin/openclaw"
fi
OPENCLAW_BIN="${OPENCLAW_BIN:-/usr/bin/openclaw}"
export OPENCLAW_BIN
# Instance slm : installer openclaw@2026.5.12 localement (n'affecte pas la prod)
if [[ "$PROFILE" == "slm" ]]; then
  mkdir -p "${OPENCLAW_PATH}/npm"
  info "Installation d'openclaw@2026.5.12 dans ${OPENCLAW_PATH}/npm..."
  npm install --prefix "${OPENCLAW_PATH}/npm" "openclaw@2026.5.12" --silent 2>&1 || \
    warn "npm install openclaw@2026.5.12 échoué — vérifier npm registry"
  SLM_BIN="${OPENCLAW_PATH}/npm/node_modules/.bin/openclaw"
  if [[ -x "$SLM_BIN" ]]; then
    OPENCLAW_BIN="$SLM_BIN"
    export OPENCLAW_BIN
    info "OpenClaw SLM : ${OPENCLAW_BIN}"
  else
    warn "Binaire openclaw@2026.5.12 non trouvé dans ${OPENCLAW_PATH}/npm"
  fi
fi

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

# auth-profiles.json de l'agent principal — injecter les clés API des providers configurés
AUTH_PROFILES="${OPENCLAW_PATH}/agents/main/agent/auth-profiles.json"
if [[ -f "$AUTH_PROFILES" ]] && [[ -n "${EURIA_API_KEY:-}" ]]; then
  python3 - <<PYEOF
import json, os
path = os.environ.get('AUTH_PROFILES', '$AUTH_PROFILES')
try:
    with open(path) as f:
        d = json.load(f)
    d.setdefault('profiles', {})
    if isinstance(d['profiles'], list):
        d['profiles'] = {p: {} for p in d['profiles']}
    d['profiles']['euria:default'] = {
        'type': 'api_key',
        'provider': 'euria',
        'key': '${EURIA_API_KEY}'
    }
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    print('[INFO] Profil euria:default écrit dans auth-profiles.json')
except Exception as e:
    print(f'[WARN] auth-profiles.json non mis à jour : {e}')
PYEOF
fi

# openclaw.json
TARGET="${OPENCLAW_PATH}/openclaw.json"
if [[ -f "$TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw.json existe déjà — ignoré (utilisez --force pour écraser)"
else
  export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND DEEPSEEK_API_KEY OPENCLAW_GATEWAY_TOKEN EURIA_API_KEY EURIA_PRODUCT_ID
  _json_tpl="${SCRIPT_DIR}/openclaw.json.template"
  [[ "$PROFILE" == "slm" ]] && _json_tpl="${SCRIPT_DIR}/openclaw-slm.json.template"
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND} ${DEEPSEEK_API_KEY} ${OPENCLAW_GATEWAY_TOKEN} ${EURIA_API_KEY} ${EURIA_PRODUCT_ID}' \
    < "$_json_tpl" > "$TARGET"
  unset _json_tpl
  # Sync .bak pour éviter que le gateway détecte notre écriture comme un "clobber"
  # et restaure silencieusement l'ancienne config au démarrage suivant.
  cp "$TARGET" "${OPENCLAW_PATH}/openclaw.json.bak" 2>/dev/null || true
  info "openclaw.json généré dans ${OPENCLAW_PATH}"
fi

# gateway.systemd.env
_env_tpl="${SCRIPT_DIR}/gateway.systemd.env.template"
[[ "$PROFILE" == "slm" ]] && _env_tpl="${SCRIPT_DIR}/gateway-slm.systemd.env.template"
ENV_TARGET="${OPENCLAW_PATH}/gateway.systemd.env"
if [[ -f "$ENV_TARGET" && "$FORCE" != "true" ]]; then
  info "gateway.systemd.env existe déjà — ignoré"
elif [[ "$FORCE" != "true" ]]; then
  # Premier passage : TELEGRAM_BOT_TOKEN vide (à renseigner), OPENCLAW_GATEWAY_TOKEN auto-généré
  TELEGRAM_BOT_TOKEN="" GATEWAY_PASSWORD="" \
    envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD} ${OPENCLAW_BIN} ${HOME}' \
    < "$_env_tpl" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env généré (600) — renseigner TELEGRAM_BOT_TOKEN et DEEPSEEK_API_KEY avant de démarrer le service"
else
  # Passage --force : injecter les secrets (depuis le fichier existant ou --env-file)
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — à renseigner dans ${ENV_TARGET}"
  fi
  export TELEGRAM_BOT_TOKEN OPENCLAW_GATEWAY_TOKEN GATEWAY_PASSWORD OPENCLAW_BIN GOG_ACCOUNT
  envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD} ${OPENCLAW_BIN} ${HOME} ${GOG_ACCOUNT}' \
    < "$_env_tpl" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env mis à jour (600)"
fi
unset _env_tpl

# Service systemd user
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
_gw_svc="openclaw-gateway.service"
[[ "$PROFILE" == "slm" ]] && _gw_svc="openclaw-gateway-slm.service"
SERVICE_TARGET="${SYSTEMD_USER_DIR}/${_gw_svc}"
if [[ -f "$SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "${_gw_svc} existe déjà — ignoré"
else
  envsubst '${OPENCLAW_BIN}' \
    < "${SCRIPT_DIR}/${_gw_svc}" \
    > "$SERVICE_TARGET"
  info "Service systemd installé dans ${SYSTEMD_USER_DIR} (${OPENCLAW_BIN})"
fi
unset _gw_svc

# switch-model
SWITCH_MODEL_TARGET="${HOME}/.local/bin/switch-model"
mkdir -p "${HOME}/.local/bin"
cp "${SCRIPT_DIR}/switch-model" "$SWITCH_MODEL_TARGET"
chmod +x "$SWITCH_MODEL_TARGET"
info "switch-model installé dans ${HOME}/.local/bin"

if [[ "$PROFILE" != "slm" ]]; then
# Services MCP SSE
for _svc in wiki-lm-mcp gog-mcp; do
  _svc_src="${SCRIPT_DIR}/${_svc}.service"
  _svc_dst="${SYSTEMD_USER_DIR}/${_svc}.service"
  if [[ -f "$_svc_dst" && "$FORCE" != "true" ]]; then
    info "${_svc}.service existe déjà — ignoré"
  else
    cp "$_svc_src" "$_svc_dst"
    info "${_svc}.service installé dans ${SYSTEMD_USER_DIR}"
  fi
done

# Service router-mcp (routeur d'intention EmbedRouter BGE-M3)
ROUTER_SVC_DST="${SYSTEMD_USER_DIR}/router-mcp.service"
if [[ -f "$ROUTER_SVC_DST" && "$FORCE" != "true" ]]; then
  info "router-mcp.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/router-mcp.service" "$ROUTER_SVC_DST"
  info "router-mcp.service installé dans ${SYSTEMD_USER_DIR}"
fi
ROUTER_BIN="${HOME}/Secretarius/Wiki_LM/routing/routing_mcp.py"
if [[ -f "$ROUTER_BIN" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable router-mcp.service 2>/dev/null && \
    info "router-mcp.service activé au boot" || \
    warn "Activation de router-mcp.service échouée"
fi

# Service SLM local (llama.cpp Phi-4-mini, cerveau de Tiron)
# Installé partout, mais activé/démarré seulement si le binaire et le modèle existent
SLM_SVC_DST="${SYSTEMD_USER_DIR}/slm-llama-cpp.service"
if [[ -f "$SLM_SVC_DST" && "$FORCE" != "true" ]]; then
  info "slm-llama-cpp.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/slm-llama-cpp.service" "$SLM_SVC_DST"
  info "slm-llama-cpp.service installé dans ${SYSTEMD_USER_DIR}"
fi
SLM_BIN="${HOME}/llama.cpp/build/bin/llama-server"
SLM_MODEL="${HOME}/Modèles/Phi-4-mini-instruct-Q6_K.gguf"
if [[ -x "$SLM_BIN" && -f "$SLM_MODEL" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable slm-llama-cpp.service 2>/dev/null && \
    info "slm-llama-cpp.service activé au boot" || \
    warn "Activation de slm-llama-cpp.service échouée"
else
  info "llama-server ou modèle Phi-4-mini absent — slm-llama-cpp.service installé mais non activé"
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

# Dépendances Python pour injection-guard (vérifiées dans le venv Wiki_LM)
VENV_PY="${SCRIPT_DIR}/../Wiki_LM/.venv/bin/python3"
if [[ -x "$VENV_PY" ]]; then
  if "$VENV_PY" -c "import flask, bs4, requests" &>/dev/null 2>&1; then
    info "flask, beautifulsoup4, requests déjà installés dans le venv"
  else
    info "Installation de flask, beautifulsoup4, requests dans le venv..."
    "${SCRIPT_DIR}/../Wiki_LM/.venv/bin/pip" install --quiet flask beautifulsoup4 requests || \
      warn "pip install échoué dans le venv"
  fi
else
  warn "Venv Wiki_LM non encore créé — flask/bs4/requests installés à l'étape suivante"
fi

# transformers et torch (optionnel — slow download, ~1GB)
if python3 -c "import transformers" &>/dev/null; then
  info "transformers déjà installé"
else
  info "Installation de transformers et torch (peut prendre plusieurs minutes)..."
  pip install --user --quiet transformers torch || \
    warn "pip install transformers échoué — DeBERTa désactivé, mode regex-only"
fi

# openclaw-mcp-adapter — pré-installation des dépendances npm uniquement.
# L'appel à openclaw plugins install est dans install.sh (racine), APRÈS le
# redémarrage du gateway : plugins.installs n'est écrit que quand le gateway répond.
ADAPTER_SRC="${SCRIPT_DIR}/openclaw-mcp-adapter"
if ! [[ -d "${ADAPTER_SRC}/node_modules/@modelcontextprotocol" ]]; then
  info "Pré-installation des dépendances npm de openclaw-mcp-adapter..."
  (cd "${ADAPTER_SRC}" && npm install --omit=dev --silent) || \
    warn "npm install pré-install échoué dans ${ADAPTER_SRC}"
fi

# Compiler les sources TypeScript de l'adaptateur
if [[ -f "${ADAPTER_SRC}/tsconfig.json" ]]; then
  info "Compilation TypeScript de openclaw-mcp-adapter..."
  (cd "${ADAPTER_SRC}" && node_modules/.bin/tsc --noEmit false 2>&1) || \
    warn "tsc échoué — l'adaptateur utilisera les .js existants"
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

fi  # end guard 1 (services MCP, Scout, injection-guard, Python deps)

# Workspace .md et skills
WORKSPACE_SRC="${SCRIPT_DIR}/workspace"
[[ "$PROFILE" == "slm" ]] && WORKSPACE_SRC="${SCRIPT_DIR}/workspace-slm"
WORKSPACE_DST="${OPENCLAW_PATH}/workspace"
export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME
SUBST_VARS='${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME}'

while IFS= read -r -d '' src; do
  rel="${src#${WORKSPACE_SRC}/}"
  dst="${WORKSPACE_DST}/${rel}"
  mkdir -p "$(dirname "$dst")"
  # AGENTS.md est de la configuration — toujours mis à jour
  if [[ "$rel" == "AGENTS.md" || "$FORCE" == "true" || ! -f "$dst" ]]; then
    envsubst "$SUBST_VARS" < "$src" > "$dst"
    info "${rel} installé"
  else
    info "${rel} existe déjà — ignoré"
  fi
done < <(find "$WORKSPACE_SRC" -name "*.md" -print0)

if [[ "$PROFILE" != "slm" ]]; then
# Workspace scout (isolé dans agents/scout/workspace/)
SCOUT_WORKSPACE_SRC="${SCRIPT_DIR}/agents/scout/workspace"
SCOUT_WORKSPACE_DST="${OPENCLAW_PATH}/agents/scout/workspace"
mkdir -p "${SCOUT_WORKSPACE_DST}/tasks/pending" "${SCOUT_WORKSPACE_DST}/tasks/done" "${SCOUT_WORKSPACE_DST}/results"

while IFS= read -r -d '' src; do
  rel="${src#${SCOUT_WORKSPACE_SRC}/}"
  dst="${SCOUT_WORKSPACE_DST}/${rel}"
  mkdir -p "$(dirname "$dst")"
  if [[ "$rel" == "AGENTS.md" || "$FORCE" == "true" || ! -f "$dst" ]]; then
    envsubst "$SUBST_VARS" < "$src" > "$dst"
    info "scout/${rel} installé"
  else
    info "scout/${rel} existe déjà — ignoré"
  fi
done < <(find "$SCOUT_WORKSPACE_SRC" -name "*.md" -print0)

# Workspace wikilm (isolé dans agents/wikilm/workspace/)
WIKILM_WORKSPACE_SRC="${SCRIPT_DIR}/agents/wikilm/workspace"
WIKILM_WORKSPACE_DST="${OPENCLAW_PATH}/agents/wikilm/workspace"
mkdir -p "${WIKILM_WORKSPACE_DST}"

while IFS= read -r -d '' src; do
  rel="${src#${WIKILM_WORKSPACE_SRC}/}"
  dst="${WIKILM_WORKSPACE_DST}/${rel}"
  mkdir -p "$(dirname "$dst")"
  if [[ "$rel" == "AGENTS.md" || "$FORCE" == "true" || ! -f "$dst" ]]; then
    envsubst "$SUBST_VARS" < "$src" > "$dst"
    info "wikilm/${rel} installé"
  else
    info "wikilm/${rel} existe déjà — ignoré"
  fi
done < <(find "$WIKILM_WORKSPACE_SRC" -name "*.md" -print0)

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
fi  # end guard 2 (Scout workspace, wikilm workspace, Docker image)

# Finalisation instance slm
if [[ "$PROFILE" == "slm" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable openclaw-gateway-slm.service 2>/dev/null && \
    info "openclaw-gateway-slm.service activé" || \
    warn "Activation échouée — lancer manuellement"
  info "Installation slm terminée."
  info "Démarrer : systemctl --user start openclaw-gateway-slm.service"
  info "UI : http://localhost:18790"
fi
