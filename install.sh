#!/usr/bin/env bash
# install.sh — Installation idempotente de Secretarius
# Usage: ./install.sh [--obsidian-path PATH] [--assistant-name NAME] [--llm BACKEND] \
#                      [--openclaw-path PATH] [--env-file FILE] [--interactive] [--help]
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Couleurs ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

# ── Defaults depuis install.conf ──
source "${SECRETARIUS_ROOT}/install.conf"

# ── Parse arguments ──
INTERACTIVE=false
ENV_FILE=""

usage() {
  cat << 'EOF'
Usage: ./install.sh [options]

Options:
  --obsidian-path PATH    Chemin du coffre Obsidian (défaut: ~/Documents/Obsidian)
  --assistant-name NAME   Nom de l'assistant OpenClaw (défaut: Tiron)
  --llm BACKEND           LLM par défaut: deepseek | ollama | local (défaut: deepseek)
  --openclaw-path PATH    Où installer la config OpenClaw (défaut: ~/.openclaw)
  --env-file FILE         Fichier contenant les secrets (API keys, tokens)
  --interactive           Mode interactif: pose les questions une par une
  --help                  Affiche l'aide
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --obsidian-path)   OBSIDIAN_PATH="$2"; shift 2 ;;
    --assistant-name)  ASSISTANT_NAME="$2"; shift 2 ;;
    --llm)             LLM_BACKEND="$2"; shift 2 ;;
    --openclaw-path)   OPENCLAW_PATH="$2"; shift 2 ;;
    --env-file)        ENV_FILE="$2"; shift 2 ;;
    --interactive)     INTERACTIVE=true; shift ;;
    --help)            usage ;;
    *)                 error "Option inconnue: $1"; usage ;;
  esac
done

# ── Mode interactif ──
if [[ "$INTERACTIVE" == true ]]; then
  echo "=== Installation Secretarius (mode interactif) ==="
  echo ""
  read -rp "Chemin du coffre Obsidian [${OBSIDIAN_PATH}]: " val
  OBSIDIAN_PATH="${val:-$OBSIDIAN_PATH}"
  read -rp "Nom de l'assistant [${ASSISTANT_NAME}]: " val
  ASSISTANT_NAME="${val:-$ASSISTANT_NAME}"
  read -rp "LLM backend (deepseek|ollama|local) [${LLM_BACKEND}]: " val
  LLM_BACKEND="${val:-$LLM_BACKEND}"
  read -rp "Chemin config OpenClaw [${OPENCLAW_PATH}]: " val
  OPENCLAW_PATH="${val:-$OPENCLAW_PATH}"
  read -rp "Fichier de secrets (optionnel): " val
  ENV_FILE="${val:-$ENV_FILE}"
fi

# ── Charger les secrets depuis --env-file ──
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    error "Fichier de secrets introuvable: $ENV_FILE"
    exit 1
  fi
  source "$ENV_FILE"
fi

# ── Étape 1: Vérification des prérequis ──
info "Vérification des prérequis..."

# Python 3.11+
if ! command -v python3 &>/dev/null; then
  error "Python3 non installé"
  exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
  error "Python 3.11+ requis (trouvé: $PYTHON_VERSION)"
  exit 1
fi
info "Python $PYTHON_VERSION ✓"

# Git
if ! command -v git &>/dev/null; then
  error "Git non installé"
  exit 1
fi
info "Git ✓"

# OpenClaw (version minimale)
if command -v openclaw &>/dev/null; then
  OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "0.0.0")
  info "OpenClaw $OPENCLAW_VERSION détecté"
  if [[ "$OPENCLAW_VERSION" < "$OPENCLAW_MIN_VERSION" ]]; then
    warn "OpenClaw >= $OPENCLAW_MIN_VERSION recommandé (trouvé: $OPENCLAW_VERSION)"
  fi
else
  warn "OpenClaw non installé — la config sera générée mais le service ne démarrera pas"
fi

# ── Étape 2: Validation du coffre Obsidian ──
info "Validation du coffre Obsidian..."
if [[ ! -d "$OBSIDIAN_PATH" ]]; then
  warn "Coffre Obsidian introuvable: $OBSIDIAN_PATH"
  read -rp "Créer le répertoire ? [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy] ]]; then
    mkdir -p "$OBSIDIAN_PATH"
    info "Répertoire créé: $OBSIDIAN_PATH"
  else
    error "Coffre Obsidian requis. Annulation."
    exit 1
  fi
else
  info "Coffre Obsidian trouvé: $OBSIDIAN_PATH ✓"
fi

# ── Étape 3: Génération de ~/.openclaw/ ──
info "Génération de la configuration OpenClaw..."
export OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND OPENCLAW_PATH
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
export GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"

bash "${OPENCLAW_CONFIG_PATH}/install.sh"

# ── Étape 4: Configuration Wiki_LM/.env ──
info "Configuration de Wiki_LM/.env..."
WIKI_ENV_FILE="${SECRETARIUS_ROOT}/.env"
if [[ ! -f "$WIKI_ENV_FILE" ]]; then
  cp "${WIKI_LM_PATH}/.env.template" "$WIKI_ENV_FILE"
  WIKI_DATA_PATH="${OBSIDIAN_PATH}/Wiki_LM"
  sed -i "s|^WIKI_PATH=.*|WIKI_PATH=${WIKI_DATA_PATH}|" "$WIKI_ENV_FILE"
  info "Wiki_LM/.env créé depuis le template"
else
  info "Wiki_LM/.env existe déjà — inchangé"
fi

# ── Étape 5: Création des répertoires de données ──
info "Création des répertoires de données..."
DATA_ROOT="${SECRETARIUS_ROOT}/data"
mkdir -p "${DATA_ROOT}/raw" "${DATA_ROOT}/wiki"
info "Répertoires créés: ${DATA_ROOT}/{raw,wiki}"

# ── Étape 6: Installation des dépendances Python ──
info "Installation des dépendances Python..."
if command -v pip3 &>/dev/null; then
  pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet
  info "Dépendances Python installées ✓"
else
  warn "pip3 non trouvé — installez manuellement: pip install -r Wiki_LM/requirements.txt"
fi

# ── Résumé ──
echo ""
info "=== Installation terminée ==="
echo ""
echo "Prochaines étapes :"
echo "  1. Activer le service OpenClaw :"
echo "     systemctl --user daemon-reload"
echo "     systemctl --user enable --now openclaw-gateway.service"
echo ""
echo "  2. Tester Wiki_LM :"
echo "     cd ${WIKI_LM_PATH}"
echo "     python -m pytest tests/"
echo ""
echo "  3. Ingérer une source :"
echo "     python tools/ingest.py https://example.com"
