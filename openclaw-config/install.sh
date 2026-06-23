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
  OPENCLAW_PATH="$HOME/.openclaw-slm"
else
  OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
fi

OPENCLAW_DIR=".openclaw"
OPENCLAW_PORT=18789
[[ "$PROFILE" == "slm" ]] && OPENCLAW_DIR=".openclaw-slm"
[[ "$PROFILE" == "slm" ]] && OPENCLAW_PORT=18790
export OPENCLAW_DIR OPENCLAW_PORT

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

# Détecter le binaire openclaw (PATH, versions NVM, npm global)
OPENCLAW_BIN=$(command -v openclaw 2>/dev/null || true)
# Ignorer notre propre lien ~/.local/bin/openclaw : on doit résoudre le binaire
# RÉEL (NVM). Sinon, si ce lien est en tête du PATH, le symlink créé plus bas
# pointerait vers lui-même → "Too many levels of symbolic links".
# (Couvre aussi le cas où ce lien est déjà cassé : -x échoue → on re-détecte.)
if [[ "$OPENCLAW_BIN" == "${HOME}/.local/bin/openclaw" || ! -x "$OPENCLAW_BIN" ]]; then
  OPENCLAW_BIN=""
fi
# Fallback : sourcer nvm.sh ne met pas toujours node dans le PATH (pas de nvm use) —
# chercher directement dans les versions NVM installées (la plus récente).
if [[ -z "$OPENCLAW_BIN" ]]; then
  for _cand in "$HOME"/.nvm/versions/node/*/bin/openclaw; do
    [[ -x "$_cand" ]] && OPENCLAW_BIN="$_cand"
  done
  unset _cand
fi
if [[ -z "$OPENCLAW_BIN" ]]; then
  NPM_PREFIX=$(npm prefix -g 2>/dev/null || true)
  [[ -n "$NPM_PREFIX" ]] && OPENCLAW_BIN="${NPM_PREFIX}/bin/openclaw"
fi
OPENCLAW_BIN="${OPENCLAW_BIN:-/usr/bin/openclaw}"
if [[ ! -x "$OPENCLAW_BIN" ]]; then
  warn "openclaw introuvable. Installer dans votre Node (NVM) :"
  echo "    npm install -g openclaw"
  warn "(NE PAS utiliser install-cli.sh : openclaw irait dans ~/.openclaw, collision.)"
  warn "puis relancer ce script."
  exit 1
fi
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

# Fichier secret clé Euria (monté en lecture seule dans le conteneur wiki)
mkdir -p "${OPENCLAW_PATH}/secrets"
chmod 700 "${OPENCLAW_PATH}/secrets"
if [[ -n "${EURIA_API_KEY:-}" ]]; then
  printf '%s' "$EURIA_API_KEY" > "${OPENCLAW_PATH}/secrets/euria-key"
  chmod 600 "${OPENCLAW_PATH}/secrets/euria-key"
  info "Fichier secrets/euria-key créé (600)"
else
  warn "EURIA_API_KEY absent — secrets/euria-key non créé (l'agent wiki échouera)"
fi

# Note : l'auth par agent (clés API) est configurée plus bas, APRÈS la génération
# d'openclaw.json, via `openclaw models auth paste-api-key` (openclaw 2026.6.x lit
# son propre store et ignore un auth-profiles.json écrit à la main).

# openclaw.json
TARGET="${OPENCLAW_PATH}/openclaw.json"
if [[ -f "$TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw.json existe déjà — ignoré (utilisez --force pour écraser)"
else
  export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME OPENCLAW_GATEWAY_TOKEN EURIA_API_KEY EURIA_PRODUCT_ID OPENCLAW_DIR OPENCLAW_PORT
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${OPENCLAW_GATEWAY_TOKEN} ${EURIA_API_KEY} ${EURIA_PRODUCT_ID} ${OPENCLAW_DIR} ${OPENCLAW_PORT}' \
    < "${SCRIPT_DIR}/openclaw-slm.json.template" > "$TARGET"
  # Sync .bak pour éviter que le gateway détecte notre écriture comme un "clobber"
  # et restaure silencieusement l'ancienne config au démarrage suivant.
  cp "$TARGET" "${OPENCLAW_PATH}/openclaw.json.bak" 2>/dev/null || true
  info "openclaw.json généré dans ${OPENCLAW_PATH}"
fi

# Réconcilier le token gateway (même quand openclaw.json est "ignoré").
# Le gateway (systemd) s'authentifie via OPENCLAW_GATEWAY_TOKEN (env) ; le CLI et la
# Control UI lisent gateway.{auth,remote}.token du fichier. S'ils divergent →
# "unauthorized: gateway token mismatch". On force auth=remote=OPENCLAW_GATEWAY_TOKEN
# dans openclaw.json + .bak + .last-good (sinon un ancien token persiste et casse l'UI).
if [[ -f "$TARGET" && -n "${OPENCLAW_GATEWAY_TOKEN:-}" ]]; then
  OPENCLAW_GATEWAY_TOKEN="$OPENCLAW_GATEWAY_TOKEN" python3 - "$OPENCLAW_PATH" <<'PYEOF'
import json, os, sys
base, tok = sys.argv[1], os.environ['OPENCLAW_GATEWAY_TOKEN']
for name in ('openclaw.json', 'openclaw.json.bak', 'openclaw.json.last-good'):
    p = os.path.join(base, name)
    if not os.path.exists(p):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    g = d.setdefault('gateway', {})
    g.setdefault('auth', {}).update({'mode': 'token', 'token': tok})
    g.setdefault('remote', {})['token'] = tok
    json.dump(d, open(p, 'w'), indent=1)
print("[INFO] token gateway réconcilié (auth=remote) dans openclaw.json + .bak/.last-good")
PYEOF
fi

# Auth par agent (clés API). openclaw 2026.6.x stocke l'auth dans un store par
# agent et IGNORE un auth-profiles.json écrit à la main → "No API key found for
# provider". La commande officielle `models auth paste-api-key` écrit le profil ET
# met à jour la config. La clé est passée sur stdin. Chaque agent a son store isolé.
if [[ -f "$TARGET" && ( -n "${EURIA_API_KEY:-}" || -n "${DEEPSEEK_API_KEY:-}" ) ]]; then
  export OPENCLAW_GATEWAY_TOKEN   # éviter que la commande régénère le token gateway
  _set_auth() {  # $1=agent  $2=provider  $3=clé
    [[ -n "$3" ]] || return 0
    if printf '%s\n' "$3" | "$OPENCLAW_BIN" models auth --agent "$1" \
         paste-api-key --provider "$2" --profile-id "$2:default" >/dev/null 2>&1; then
      info "auth ${2} → agent ${1}"
    else
      warn "auth ${2} → agent ${1} échouée"
    fi
  }
  _set_auth main  euria    "${EURIA_API_KEY:-}"
  _set_auth wiki  euria    "${EURIA_API_KEY:-}"
  _set_auth gog   euria    "${EURIA_API_KEY:-}"
  _set_auth main  deepseek "${DEEPSEEK_API_KEY:-}"
  _set_auth scout deepseek "${DEEPSEEK_API_KEY:-}"
  unset -f _set_auth
  # paste-api-key réécrit openclaw.json (et un .bak décalé) à chaque appel :
  # resync .bak = openclaw.json pour éviter la restauration anti-clobber au démarrage.
  cp "$TARGET" "${OPENCLAW_PATH}/openclaw.json.bak" 2>/dev/null || true
fi

# gateway.systemd.env
_env_tpl="${SCRIPT_DIR}/gateway-slm.systemd.env.template"
ENV_TARGET="${OPENCLAW_PATH}/gateway.systemd.env"
if [[ -f "$ENV_TARGET" && "$FORCE" != "true" ]]; then
  info "gateway.systemd.env existe déjà — ignoré (utilisez --force pour régénérer)"
else
  # Génère gateway.systemd.env avec les secrets fournis (env/--env-file).
  # Variables manquantes = chaîne vide ; on avertit pour les indispensables.
  [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]] && warn "TELEGRAM_BOT_TOKEN non défini — à renseigner dans ${ENV_TARGET} avant de démarrer"
  [[ -z "${EURIA_API_KEY:-}" ]] && warn "EURIA_API_KEY non défini — à renseigner dans ${ENV_TARGET} avant de démarrer"
  export TELEGRAM_BOT_TOKEN OPENCLAW_GATEWAY_TOKEN GATEWAY_PASSWORD OPENCLAW_BIN GOG_ACCOUNT EURIA_API_KEY EURIA_PRODUCT_ID DEEPSEEK_API_KEY
  envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD} ${OPENCLAW_BIN} ${HOME} ${GOG_ACCOUNT} ${EURIA_API_KEY} ${EURIA_PRODUCT_ID} ${DEEPSEEK_API_KEY}' \
    < "$_env_tpl" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env généré (600)"
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

# Wrapper openclaw dans ~/.local/bin (au lieu d'un symlink) :
#  1) NVM n'est pas chargé dans les sessions SSH → openclaw introuvable. ~/.local/bin
#     est dans le PATH par défaut (Ubuntu, via ~/.profile).
#  2) SURTOUT : toute commande `openclaw` lancée SANS OPENCLAW_GATEWAY_TOKEN dans
#     l'environnement RÉGÉNÈRE le token gateway dans openclaw.json → désynchronise
#     le service en cours et casse l'auth (CLI + Control UI). Le wrapper charge le
#     token depuis gateway.systemd.env avant d'appeler le vrai binaire.
if [[ -x "$OPENCLAW_BIN" && "$OPENCLAW_BIN" != "${HOME}/.local/bin/openclaw" ]]; then
  cat > "${HOME}/.local/bin/openclaw" <<WRAP
#!/usr/bin/env bash
# Généré par Secretarius install.sh — ne pas éditer.
_envf="\${HOME}/.openclaw/gateway.systemd.env"
if [[ -f "\$_envf" ]]; then
  _t=\$(grep '^OPENCLAW_GATEWAY_TOKEN=' "\$_envf" | cut -d= -f2-)
  [[ -n "\$_t" ]] && export OPENCLAW_GATEWAY_TOKEN="\$_t"
fi
exec "${OPENCLAW_BIN}" "\$@"
WRAP
  chmod +x "${HOME}/.local/bin/openclaw"
  info "wrapper openclaw installé dans ${HOME}/.local/bin (→ ${OPENCLAW_BIN}, token chargé)"
fi

# Service Wiki_LM Query Server (Flask :5051, pour le template de requête Obsidian).
# Installé quel que soit le profil ; activé seulement si le venv et server.py existent.
WIKI_SERVER_DST="${SYSTEMD_USER_DIR}/wiki-lm-server.service"
if [[ -f "$WIKI_SERVER_DST" && "$FORCE" != "true" ]]; then
  info "wiki-lm-server.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/wiki-lm-server.service" "$WIKI_SERVER_DST"
  info "wiki-lm-server.service installé dans ${SYSTEMD_USER_DIR}"
fi
if [[ -x "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3" && -f "${HOME}/Secretarius/Wiki_LM/tools/server.py" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable wiki-lm-server.service 2>/dev/null && \
    info "wiki-lm-server.service activé au boot" || \
    warn "Activation de wiki-lm-server.service échouée"
fi

# Timer Wiki_LM embeddings : recalcul incrémental périodique (+ reload du serveur).
for _wl in wiki-lm-embed.service wiki-lm-embed.timer; do
  _wl_dst="${SYSTEMD_USER_DIR}/${_wl}"
  if [[ -f "$_wl_dst" && "$FORCE" != "true" ]]; then
    info "${_wl} existe déjà — ignoré"
  else
    cp "${SCRIPT_DIR}/${_wl}" "$_wl_dst"
    info "${_wl} installé dans ${SYSTEMD_USER_DIR}"
  fi
done
if [[ -x "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3" && -f "${HOME}/Secretarius/Wiki_LM/tools/embed.py" ]]; then
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now wiki-lm-embed.timer 2>/dev/null && \
    info "wiki-lm-embed.timer activé (toutes les 6 h)" || \
    warn "Activation de wiki-lm-embed.timer échouée"
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
  # AGENTS.md est de la configuration — toujours mis à jour
  if [[ "$rel" == "AGENTS.md" || "$rel" == */SKILL.md || "$FORCE" == "true" || ! -f "$dst" ]]; then
    envsubst "$SUBST_VARS" < "$src" > "$dst"
    info "${rel} installé"
  else
    info "${rel} existe déjà — ignoré"
  fi
done < <(find "$WORKSPACE_SRC" -name "*.md" -print0)

# Workspaces sous-agents (wiki, scout, gog)
for _pair in "workspace-wiki:workspace-wiki" "workspace-scout:workspace-scout" "workspace-gog:workspace-gog"; do
  _wa_src="${SCRIPT_DIR}/${_pair%%:*}"
  _wa_dst="${OPENCLAW_PATH}/${_pair##*:}"
  if [[ ! -d "$_wa_src" ]]; then
    info "Source ${_pair%%:*} absente — ignorée"
    continue
  fi
  mkdir -p "$_wa_dst"
  while IFS= read -r -d '' src; do
    rel="${src#${_wa_src}/}"
    dst="${_wa_dst}/${rel}"
    mkdir -p "$(dirname "$dst")"
    if [[ "$rel" == "AGENTS.md" || "$rel" == */SKILL.md || "$FORCE" == "true" || ! -f "$dst" ]]; then
      envsubst "$SUBST_VARS" < "$src" > "$dst"
      info "${_pair##*:}/${rel} installé"
    else
      info "${_pair##*:}/${rel} existe déjà — ignoré"
    fi
  done < <(find "$_wa_src" -name "*.md" -print0)
done
unset _pair _wa_src _wa_dst

# Répertoire tasks/done/results pour l'agent scout + .gog-config pour l'agent gog
mkdir -p "${OPENCLAW_PATH}/workspace-scout/tasks/pending" \
         "${OPENCLAW_PATH}/workspace-scout/tasks/done" \
         "${OPENCLAW_PATH}/workspace-scout/results"
mkdir -p "${OPENCLAW_PATH}/workspace/.gog-config"
info "Répertoires scout (tasks/results) et .gog-config créés"

# Finalisation
_gw_svc_final="openclaw-gateway.service"
_gw_port_final=18789
[[ "$PROFILE" == "slm" ]] && _gw_svc_final="openclaw-gateway-slm.service"
[[ "$PROFILE" == "slm" ]] && _gw_port_final=18790
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user enable "${_gw_svc_final}" 2>/dev/null && \
  info "${_gw_svc_final} activé" || \
  warn "Activation échouée — lancer manuellement"
info "Installation terminée."
info "Modèles Euria disponibles (agent main) :"
info "  - mistralai/Mistral-Small-4-119B-2603  [défaut si Qwen397 indispo]"
info "  - Qwen/Qwen3.5-397B-A17B-FP8           [recommandé — meilleur routage wiki]"
info "  - Qwen/Qwen3.5-122B-A10B-FP8"
info "  - google/gemma-4-31B-it"
info "  - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
info "Agent scout : deepseek/deepseek-chat (DEEPSEEK_API_KEY requis)"
info ""
info "Post-install — plugin derisk-deleg (NVM : copie manuelle requise) :"
echo "  SRC=\${HOME}/Secretarius/derisk-deleg"
echo "  DST=\${HOME}/.openclaw/extensions/derisk-deleg"
echo "  mkdir -p \"\$DST\" && cp -r \"\$SRC/dist\" \"\$SRC/node_modules\" \"\$SRC/openclaw.plugin.json\" \"\$SRC/package.json\" \"\$DST/\""
info "  Puis dans l'UI gateway : activer le plugin + hooks:allowConversationAccess=true"
info "  ⚠️  Après --force : plugins.entries est réinitialisé — rajouter l'entrée manuellement"
info ""
info "Démarrer : systemctl --user start ${_gw_svc_final}"
info "Control UI : ouvrir depuis un poste réseau, mode jeton ="
info "  grep OPENCLAW_GATEWAY_TOKEN ${OPENCLAW_PATH}/gateway.systemd.env"
unset _gw_svc_final _gw_port_final
