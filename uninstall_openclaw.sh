#!/usr/bin/env bash
# uninstall_openclaw.sh — Désinstallation d'OpenClaw et de sa configuration
# Usage: ./uninstall_openclaw.sh [--force] [--openclaw-path PATH] [--help]
#
# Supprime :
#   - Le service systemd openclaw-gateway.service
#   - Le répertoire de configuration OpenClaw (~/.openclaw par défaut)
#   - Le fichier Wiki_LM/.env généré
#
# Ne supprime PAS :
#   - Le coffre Obsidian (données utilisateur)
#   - Le dépôt Secretarius lui-même
#   - NVM / Node.js / npm (optionnel, voir --nvm)
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SECRETARIUS_ROOT}/install.conf"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

FORCE=false
REMOVE_NVM=false

usage() {
  cat << 'EOF'
Usage: ./uninstall_openclaw.sh [options]

  --openclaw-path PATH   Répertoire config OpenClaw (défaut: ~/.openclaw)
  --force                Pas de confirmation interactive
  --nvm                  Supprimer aussi NVM, Node.js et npm global
  --help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openclaw-path) OPENCLAW_PATH="$2"; shift 2 ;;
    --force)         FORCE=true; shift ;;
    --nvm)           REMOVE_NVM=true; shift ;;
    --help)          usage ;;
    *) error "Option inconnue: $1"; usage ;;
  esac
done

OPENCLAW_PATH="${OPENCLAW_PATH/#\~/$HOME}"

confirm() {
  local msg="$1"
  if [[ "$FORCE" == true ]]; then return 0; fi
  read -rp "${msg} [y/N] " c
  [[ "$c" =~ ^[Yy] ]]
}

echo ""
echo "=== Désinstallation OpenClaw / Secretarius ==="
echo ""
echo "Éléments qui seront supprimés :"
echo "  - Service systemd  : ~/.config/systemd/user/openclaw-gateway.service"
echo "  - Config OpenClaw  : ${OPENCLAW_PATH}"
echo "  - Wiki_LM/.env     : ${SECRETARIUS_ROOT}/Wiki_LM/.env"
if [[ "$REMOVE_NVM" == true ]]; then
  echo "  - NVM + Node.js    : ~/.nvm"
fi
echo ""

if [[ "$FORCE" != true ]]; then
  read -rp "Continuer ? [y/N] " c
  [[ "$c" =~ ^[Yy] ]] || { echo "Annulé."; exit 0; }
fi

# 1 — Arrêter et désactiver le service
SERVICE_FILE="$HOME/.config/systemd/user/openclaw-gateway.service"
if systemctl --user is-active openclaw-gateway.service &>/dev/null 2>&1; then
  info "Arrêt du service openclaw-gateway..."
  systemctl --user stop openclaw-gateway.service || warn "Impossible d'arrêter le service"
fi
if systemctl --user is-enabled openclaw-gateway.service &>/dev/null 2>&1; then
  systemctl --user disable openclaw-gateway.service 2>/dev/null || true
fi
if [[ -f "$SERVICE_FILE" ]]; then
  rm -f "$SERVICE_FILE"
  systemctl --user daemon-reload 2>/dev/null || true
  info "Service systemd supprimé"
else
  info "Service systemd absent — ignoré"
fi

# 2 — Supprimer la configuration OpenClaw
if [[ -d "$OPENCLAW_PATH" ]]; then
  rm -rf "$OPENCLAW_PATH"
  info "Répertoire supprimé : ${OPENCLAW_PATH}"
else
  info "Répertoire absent : ${OPENCLAW_PATH} — ignoré"
fi

# 3 — Supprimer Wiki_LM/.env
WIKI_ENV="${SECRETARIUS_ROOT}/Wiki_LM/.env"
if [[ -f "$WIKI_ENV" ]]; then
  rm -f "$WIKI_ENV"
  info "Supprimé : ${WIKI_ENV}"
else
  info "Absent : ${WIKI_ENV} — ignoré"
fi

# 4 — NVM (optionnel)
if [[ "$REMOVE_NVM" == true ]]; then
  if [[ -d "$HOME/.nvm" ]]; then
    rm -rf "$HOME/.nvm"
    info "NVM supprimé : ~/.nvm"
    warn "Pensez à retirer les lignes NVM de ~/.bashrc / ~/.zshrc / ~/.profile"
  else
    info "NVM absent — ignoré"
  fi
fi

echo ""
info "=== Désinstallation terminée ==="
echo ""
echo "Pour réinstaller : cd ${SECRETARIUS_ROOT} && ./install.sh"
