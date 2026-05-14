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

INTERACTIVE=true
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

# Mode interactif
if [[ "$INTERACTIVE" == true ]]; then
  echo "=== Installation Secretarius ==="
  read -rp "Coffre Obsidian [${OBSIDIAN_PATH}]: " v; OBSIDIAN_PATH="${v:-$OBSIDIAN_PATH}"
  read -rp "Nom de l'assistant [${ASSISTANT_NAME}]: " v; ASSISTANT_NAME="${v:-$ASSISTANT_NAME}"
  read -rp "LLM (deepseek|ollama|claude) [${LLM_BACKEND}]: " v; LLM_BACKEND="${v:-$LLM_BACKEND}"
  read -rp "Config OpenClaw [${OPENCLAW_PATH}]: " v; OPENCLAW_PATH="${v:-$OPENCLAW_PATH}"
  read -rp "Fichier de secrets (optionnel): " v; ENV_FILE="${v:-$ENV_FILE}"
fi

# Charger les secrets
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { error "Fichier introuvable: $ENV_FILE"; exit 1; }
  source "$ENV_FILE"
fi

export OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND OPENCLAW_PATH FORCE

# Étape 1 — Prérequis
info "Vérification des prérequis..."

python3 -c "import sys; assert sys.version_info >= (3,11), f'Python 3.11+ requis (trouvé {sys.version})'" \
  && info "Python $(python3 --version | cut -d' ' -f2) ✓" \
  || { error "Python 3.11+ requis"; exit 1; }

command -v git &>/dev/null && info "git ✓" || { error "git requis"; exit 1; }
command -v envsubst &>/dev/null && info "envsubst ✓" \
  || { error "envsubst requis (apt install gettext / brew install gettext)"; exit 1; }
command -v openclaw &>/dev/null \
  && info "openclaw $(openclaw --version 2>/dev/null || echo '?') ✓" \
  || warn "openclaw non trouvé — config générée mais service inactif"

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
export GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
export GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
bash "${SECRETARIUS_ROOT}/openclaw-config/install.sh"

# Étape 4 — Wiki_LM/.env
info "Configuration de Wiki_LM/.env..."
WIKI_ENV="${SECRETARIUS_ROOT}/Wiki_LM/.env"
WIKI_ENV_TEMPLATE="${SECRETARIUS_ROOT}/Wiki_LM/.env.template"
if [[ -f "$WIKI_ENV" && "$FORCE" != "true" ]]; then
  info "Wiki_LM/.env existe déjà — ignoré"
else
  cp "$WIKI_ENV_TEMPLATE" "$WIKI_ENV"
  WIKI_PATH="${OBSIDIAN_PATH}/Wiki_LM"
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
fi

# Étape 5 — Dépendances Python
info "Installation des dépendances Python..."
WIKI_LM_PATH="${SECRETARIUS_ROOT}/Wiki_LM"
if command -v pip3 &>/dev/null; then
  if pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet 2>/dev/null; then
    info "Dépendances Python ✓"
  elif pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet --break-system-packages 2>/dev/null; then
    info "Dépendances Python ✓ (--break-system-packages)"
  else
    warn "pip3 a échoué — installez dans un venv : python -m venv .venv && .venv/bin/pip install -r Wiki_LM/requirements.txt"
  fi
else
  warn "pip3 non trouvé — installez manuellement: pip install -r Wiki_LM/requirements.txt"
fi

# Résumé
echo ""
info "=== Installation terminée ==="
echo ""
echo "Prochaines étapes :"
echo ""
echo "  1. Renseigner les secrets dans ${OPENCLAW_PATH}/gateway.systemd.env :"
echo ""
echo "       TELEGRAM_BOT_TOKEN=<token BotFather de ce serveur>"
echo "       GATEWAY_TOKEN=<votre identifiant Telegram (numérique)>"
echo "       OPENCLAW_GATEWAY_TOKEN=<même valeur que GATEWAY_TOKEN>"
echo "       GATEWAY_PASSWORD=<mot de passe gateway>"
echo "       DEEPSEEK_API_KEY=<clé API DeepSeek>"
echo ""
echo "       nano ${OPENCLAW_PATH}/gateway.systemd.env"
echo ""
echo "  2. Activer le service OpenClaw :"
echo "       systemctl --user daemon-reload"
echo "       systemctl --user enable --now openclaw-gateway.service"
echo ""
echo "  3. Appairer Telegram (envoyer /start au bot, puis) :"
echo "       openclaw pairing approve telegram <CODE>"
echo ""
echo "  4. Tester Wiki_LM :"
echo "       cd ${WIKI_LM_PATH} && python -m pytest tests/"
echo ""
echo "  IMPORTANT : ne lancer aucune commande openclaw avant l'étape 1,"
echo "  sinon les fichiers de configuration seront écrasés par les défauts."
