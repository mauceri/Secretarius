#!/usr/bin/env bash
# install.sh — Installation idempotente de Secretarius
# Usage: ./install.sh [--obsidian-path PATH] [--assistant-name NAME] [--llm BACKEND]
#                     [--openclaw-path PATH] [--env-file FILE] [--interactive] [--force] [--help]
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SECRETARIUS_ROOT}/install.conf"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

[[ -t 0 ]] && INTERACTIVE=true || INTERACTIVE=false
FORCE=false
ENV_FILE=""

usage() {
  cat << 'EOF'
Usage: ./install.sh [options]

  --obsidian-path PATH    Chemin du coffre Obsidian (défaut: ~/Documents/Obsidian)
  --assistant-name NAME   Nom de l'assistant (défaut: Tiron)
  --llm BACKEND           deepseek | ollama | claude (défaut: deepseek)
  --openclaw-path PATH    Chemin config OpenClaw (défaut: ~/.openclaw)
  --env-file FILE         Fichier de secrets (API keys, tokens)
  --interactive           Pose les questions une par une
  --force                 Écrase les fichiers existants
  --help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --obsidian-path)  OBSIDIAN_PATH="$2"; shift 2 ;;
    --assistant-name) ASSISTANT_NAME="$2"; shift 2 ;;
    --llm)            LLM_BACKEND="$2"; shift 2 ;;
    --openclaw-path)  OPENCLAW_PATH="$2"; shift 2 ;;
    --env-file)       ENV_FILE="$2"; shift 2 ;;
    --interactive)    INTERACTIVE=true; shift ;;
    --force)          FORCE=true; shift ;;
    --help)           usage ;;
    *) error "Option inconnue: $1"; usage ;;
  esac
done

# Étape 1 — Prérequis (avant toute question interactive)
info "Vérification des prérequis..."
WARNINGS=()

# --- Bloquants ---
if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
  info "Python $(python3 --version | cut -d' ' -f2) ✓"
else
  error "Python 3.11+ requis"
  error "  Ubuntu/Debian : sudo apt install python3.11 python3.11-venv"
  error "  macOS         : brew install python@3.11"
  exit 1
fi

if command -v git &>/dev/null; then
  info "git ✓"
else
  error "git requis"
  error "  Ubuntu/Debian : sudo apt install git"
  error "  macOS         : brew install git"
  exit 1
fi

if command -v envsubst &>/dev/null; then
  info "envsubst ✓"
else
  error "envsubst requis (paquet gettext)"
  error "  Ubuntu/Debian : sudo apt install gettext"
  error "  macOS         : brew install gettext && brew link gettext --force"
  exit 1
fi

if command -v openclaw &>/dev/null; then
  info "openclaw $(openclaw --version 2>/dev/null | head -1 || echo '?') ✓"
else
  error "openclaw requis — installer Node.js 22+ via NVM puis openclaw :"
  echo ""
  echo "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash"
  echo "    source ~/.bashrc"
  echo "    nvm install 22 && nvm use 22"
  echo "    npm install -g openclaw"
  echo ""
  echo "Puis relancer : ./install.sh"
  exit 1
fi

# Mode interactif
if [[ "$INTERACTIVE" == true ]]; then
  echo "=== Installation Secretarius ==="
  read -rp "Coffre Obsidian [${OBSIDIAN_PATH}]: " v; OBSIDIAN_PATH="${v:-$OBSIDIAN_PATH}"
  read -rp "Nom de l'assistant [${ASSISTANT_NAME}]: " v; ASSISTANT_NAME="${v:-$ASSISTANT_NAME}"
  read -rp "LLM (deepseek|ollama|claude) [${LLM_BACKEND}]: " v; LLM_BACKEND="${v:-$LLM_BACKEND}"
  read -rp "Config OpenClaw [${OPENCLAW_PATH}]: " v; OPENCLAW_PATH="${v:-$OPENCLAW_PATH}"
fi

# Charger les secrets
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { error "Fichier introuvable: $ENV_FILE"; exit 1; }
  source "$ENV_FILE"
fi

export OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND OPENCLAW_PATH FORCE

if ! systemctl --user status &>/dev/null 2>&1; then
  WARNINGS+=("systemd user non disponible (WSL ou macOS ?) — démarrer openclaw manuellement\n    openclaw start")
fi

DOCKER_OK=false
if command -v docker &>/dev/null; then
  if docker ps &>/dev/null 2>&1; then
    info "docker $(docker --version | cut -d' ' -f3 | tr -d ',') ✓"
    DOCKER_OK=true
  else
    info "docker $(docker --version | cut -d' ' -f3 | tr -d ',') — accès refusé au socket"
    WARNINGS+=("utilisateur non dans le groupe docker — requis pour Milvus\n    sudo usermod -aG docker \$USER && newgrp docker")
  fi
else
  WARNINGS+=("docker non trouvé — requis pour Milvus\n    Ubuntu/Debian : sudo apt install docker.io docker-compose-plugin\n    Puis : sudo usermod -aG docker \$USER && newgrp docker")
fi

# Étape 2 — Coffre Obsidian
info "Validation du coffre Obsidian: ${OBSIDIAN_PATH}"
OBSIDIAN_PATH="${OBSIDIAN_PATH/#\~/$HOME}"
if [[ ! -d "$OBSIDIAN_PATH" ]]; then
  read -rp "Répertoire absent. Créer ? [y/N] " c
  [[ "$c" =~ ^[Yy] ]] && mkdir -p "$OBSIDIAN_PATH" && info "Créé: $OBSIDIAN_PATH" \
    || { error "Coffre Obsidian requis. Annulation."; exit 1; }
fi
info "Coffre Obsidian ✓"

# Étape 3 — Config OpenClaw
info "Génération de la configuration OpenClaw..."
OPENCLAW_PATH="${OPENCLAW_PATH/#\~/$HOME}"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
export GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
bash "${SECRETARIUS_ROOT}/openclaw-config/install.sh"

# Étape 4 — Wiki_LM/.env
info "Configuration de Wiki_LM/.env..."
WIKI_ENV="${SECRETARIUS_ROOT}/Wiki_LM/.env"
WIKI_ENV_TEMPLATE="${SECRETARIUS_ROOT}/Wiki_LM/.env.template"
WIKI_PATH="${OBSIDIAN_PATH}/Wiki_LM"

if [[ ! -f "$WIKI_ENV" || "$FORCE" == "true" ]]; then
  cp "$WIKI_ENV_TEMPLATE" "$WIKI_ENV"
  sed -i "s|^WIKI_PATH=.*|WIKI_PATH=${WIKI_PATH}|" "$WIKI_ENV"
  case "$LLM_BACKEND" in
    deepseek) sed -i "s|^WIKI_LLM_BACKEND=.*|WIKI_LLM_BACKEND=openai|" "$WIKI_ENV" ;;
    ollama)
      sed -i "s|^WIKI_LLM_BACKEND=.*|WIKI_LLM_BACKEND=ollama|" "$WIKI_ENV"
      sed -i "s|^OPENAI_BASE_URL=.*|# OPENAI_BASE_URL=https://api.deepseek.com/v1|" "$WIKI_ENV"
      ;;
    claude) sed -i "s|^WIKI_LLM_BACKEND=.*|WIKI_LLM_BACKEND=claude|" "$WIKI_ENV" ;;
  esac
  info "Wiki_LM/.env créé"
else
  sed -i "s|^WIKI_PATH=.*|WIKI_PATH=${WIKI_PATH}|" "$WIKI_ENV"
  info "Wiki_LM/.env : WIKI_PATH mis à jour (${WIKI_PATH})"
fi

# Étape 5 — Dépendances Python (venv)
info "Installation des dépendances Python..."
WIKI_LM_PATH="${SECRETARIUS_ROOT}/Wiki_LM"
VENV_PATH="${WIKI_LM_PATH}/.venv"
if python3 -m venv "$VENV_PATH" 2>/dev/null; then
  if "${VENV_PATH}/bin/pip" install -r "${WIKI_LM_PATH}/requirements.txt" --quiet 2>/dev/null; then
    info "Dépendances Python ✓ (venv: ${VENV_PATH})"
  else
    WARNINGS+=("dépendances Python non installées dans le venv\n    ${VENV_PATH}/bin/pip install -r ${WIKI_LM_PATH}/requirements.txt")
  fi
else
  WARNINGS+=("impossible de créer le venv Python\n    sudo apt install python3-venv\n    Puis relancer install.sh")
fi

# Vérification google-auth si Gmail configuré
if [[ -n "${GMAIL_CLIENT_ID:-}" ]]; then
  if ! python3 -c "import google.auth" 2>/dev/null; then
    WARNINGS+=("google-auth non installé (requis pour Gmail OAuth2)\n    pip3 install google-auth google-api-python-client")
  fi
fi

# Activer et démarrer les services OpenClaw si systemd est disponible
SYSTEMD_OK=false
if systemctl --user daemon-reload &>/dev/null 2>&1 && command -v openclaw &>/dev/null; then
  systemctl --user daemon-reload
  systemctl --user enable openclaw-gateway.service openclaw-scout.service 2>/dev/null || true
  SYSTEMD_OK=true
  # Démarrer seulement si les secrets sont renseignés
  if grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null; then
    systemctl --user restart openclaw-gateway.service openclaw-scout.service
    info "Services OpenClaw démarrés ✓"
  else
    info "Services OpenClaw activés (démarreront après renseignement des secrets)"
  fi
fi

# Résumé
echo ""
info "=== Installation terminée ==="
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  echo ""
  warn "Points d'attention :"
  for w in "${WARNINGS[@]}"; do
    echo -e "  - ${w}"
  done
fi
echo ""
echo "Prochaines étapes :"
echo ""


if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null; then
  echo "  1. Renseigner les secrets dans ${OPENCLAW_PATH}/gateway.systemd.env :"
  echo ""
  echo "       TELEGRAM_BOT_TOKEN=<token BotFather>"
  echo "       DEEPSEEK_API_KEY=<clé API DeepSeek>"
  echo "       GATEWAY_PASSWORD=<mot de passe optionnel pour l'interface web>"
  echo ""
  echo "       (OPENCLAW_GATEWAY_TOKEN est généré automatiquement)"
  echo ""
  echo "       nano ${OPENCLAW_PATH}/gateway.systemd.env"
  echo ""
  if [[ "$SYSTEMD_OK" == true ]]; then
    echo "       Puis démarrer les services :"
    echo "       systemctl --user start openclaw-gateway.service openclaw-scout.service"
    echo ""
  fi
fi

echo "  2. Appairer Telegram : envoyer /start au bot, puis :"
echo "       openclaw pairing approve telegram <CODE>"
echo "       (la connexion Telegram est interrompue — le gateway restart ci-dessous la rétablit)"
echo ""
echo "       Puis redémarrer le gateway pour prendre en compte le pairing :"
echo "       openclaw gateway restart"
echo ""
echo "  3. Tester Wiki_LM :"
echo "       cd ${WIKI_LM_PATH} && .venv/bin/python -m pytest tests/"

# Si docker inaccessible, rappeler la correction avant Milvus
if [[ "$DOCKER_OK" != true ]]; then
  echo ""
  echo "  Nota : Docker inaccessible — requis avant de démarrer Milvus :"
  if ! command -v docker &>/dev/null; then
    echo "       sudo apt install docker.io docker-compose-plugin"
  fi
  echo "       sudo usermod -aG docker \$USER"
  echo "       Puis fermer la session SSH et se reconnecter"
  echo "       (le groupe docker n'est pris en compte qu'à la prochaine connexion)"
fi
