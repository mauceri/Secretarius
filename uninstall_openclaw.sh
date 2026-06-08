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
REMOVE_OPENCLAW=false

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
echo "  - Services systemd : openclaw-gateway.service, openclaw-scout.service"
echo "  - scout-watcher    : ~/.local/bin/scout-watcher"
echo "  - Config OpenClaw  : ${OPENCLAW_PATH}"
echo "  - Wiki_LM/.env     : ${SECRETARIUS_ROOT}/Wiki_LM/.env"
if [[ "$REMOVE_NVM" == true ]]; then
  echo "  - NVM + Node.js    : ~/.nvm"
fi
echo ""

if [[ "$FORCE" != true ]]; then
  read -rp "Continuer ? [y/N] " c
  [[ "$c" =~ ^[Yy] ]] || { echo "Annulé."; exit 0; }
  echo ""
  read -rp "Désinstaller aussi le paquet npm openclaw ? [y/N] " c
  [[ "$c" =~ ^[Yy] ]] && REMOVE_OPENCLAW=true
fi

# 1 — Arrêter et désactiver les services
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

for svc in openclaw-gateway openclaw-scout; do
  if systemctl --user is-active "${svc}.service" &>/dev/null 2>&1; then
    info "Arrêt de ${svc}..."
    systemctl --user stop "${svc}.service" || warn "Impossible d'arrêter ${svc}"
  fi
  if systemctl --user is-enabled "${svc}.service" &>/dev/null 2>&1; then
    systemctl --user disable "${svc}.service" 2>/dev/null || true
  fi
  SERVICE_FILE="${SYSTEMD_USER_DIR}/${svc}.service"
  if [[ -f "$SERVICE_FILE" ]]; then
    rm -f "$SERVICE_FILE"
    info "Service ${svc}.service supprimé"
  else
    info "Service ${svc}.service absent — ignoré"
  fi
done
systemctl --user daemon-reload 2>/dev/null || true

# scout-watcher
for bin in scout-watcher; do
  TARGET="$HOME/.local/bin/$bin"
  if [[ -f "$TARGET" ]]; then
    rm -f "$TARGET"
    info "$bin supprimé"
  else
    info "$bin absent — ignoré"
  fi
done

# 2 — Supprimer la configuration OpenClaw
if [[ -d "$OPENCLAW_PATH" ]]; then
  rm -rf "$OPENCLAW_PATH"
  info "Répertoire supprimé : ${OPENCLAW_PATH}"
else
  info "Répertoire absent : ${OPENCLAW_PATH} — ignoré"
fi

# 3 — Désinstaller le paquet openclaw (optionnel)
if [[ "$REMOVE_OPENCLAW" == true ]]; then
  if command -v openclaw &>/dev/null; then
    info "Désinstallation du paquet openclaw..."
    npm uninstall -g openclaw 2>/dev/null \
      || npm uninstall -g openclaw --prefix "$(npm prefix -g 2>/dev/null)" 2>/dev/null \
      || warn "Impossible de désinstaller openclaw via npm — supprimez-le manuellement"
  else
    info "Paquet openclaw absent — ignoré"
  fi
else
  info "Paquet openclaw conservé"
fi

# 4 — Supprimer Wiki_LM/.env
WIKI_ENV="${SECRETARIUS_ROOT}/Wiki_LM/.env"
if [[ -f "$WIKI_ENV" ]]; then
  rm -f "$WIKI_ENV"
  info "Supprimé : ${WIKI_ENV}"
else
  info "Absent : ${WIKI_ENV} — ignoré"
fi

# 5 — NVM (optionnel)
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
warn "Si des secrets sont encore exportés dans ce shell, purgez-les avant de réinstaller :"
echo "  unset TELEGRAM_BOT_TOKEN OPENCLAW_GATEWAY_TOKEN GATEWAY_PASSWORD DEEPSEEK_API_KEY"
echo ""
echo "Pour réinstaller : cd ${SECRETARIUS_ROOT} && ./install.sh"
