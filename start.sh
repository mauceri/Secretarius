#!/usr/bin/env bash
# start.sh — Démarre les services Secretarius sans réinstaller
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SECRETARIUS_ROOT}/install.conf"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

# Source NVM (sessions SSH non-interactives ne chargent pas .bashrc)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh" 2>/dev/null || true

OPENCLAW_PATH="${OPENCLAW_PATH/#\~/$HOME}"

if ! systemctl --user status &>/dev/null 2>&1; then
  error "systemd user non disponible"
  exit 1
fi

# Vérifier le binaire openclaw dans le service
SERVICE_FILE="${HOME}/.config/systemd/user/openclaw-gateway.service"
if [[ -f "$SERVICE_FILE" ]]; then
  EXEC_BIN=$(grep '^ExecStart=' "$SERVICE_FILE" | sed 's/ExecStart=\([^ ]*\).*/\1/')
  if [[ -n "$EXEC_BIN" && ! -x "$EXEC_BIN" ]]; then
    error "Binaire openclaw introuvable : $EXEC_BIN"
    error "Relancer : cd openclaw-config && bash install.sh --force"
    exit 1
  fi
fi

if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null; then
  error "TELEGRAM_BOT_TOKEN absent de ${OPENCLAW_PATH}/gateway.systemd.env"
  exit 1
fi

systemctl --user daemon-reload

# Gateway (obligatoire)
systemctl --user restart openclaw-gateway.service
info "openclaw-gateway démarré ✓"

# Wiki-LM server (optionnel — si le venv et server.py sont présents)
if systemctl --user is-enabled wiki-lm-server.service &>/dev/null 2>&1; then
  systemctl --user restart wiki-lm-server.service
  info "wiki-lm-server démarré ✓"
else
  warn "wiki-lm-server.service non activé (normal si Wiki_LM non installé)"
fi

echo ""
info "=== Services Secretarius ==="
systemctl --user status openclaw-gateway.service --no-pager -l 2>&1 \
  | grep -E "(Active|●)" || true
