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
  error "systemd user non disponible — impossible de démarrer les services"
  exit 1
fi

systemctl --user daemon-reload

# injection-guard
if systemctl --user is-enabled openclaw-injection-guard.service &>/dev/null 2>&1; then
  systemctl --user restart openclaw-injection-guard.service
  info "openclaw-injection-guard démarré ✓"
else
  warn "openclaw-injection-guard.service non activé — relancer install.sh pour l'installer"
fi

# gateway + scout
if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null; then
  error "TELEGRAM_BOT_TOKEN absent de ${OPENCLAW_PATH}/gateway.systemd.env"
  error "Renseigner les secrets puis relancer start.sh"
  exit 1
fi

systemctl --user restart openclaw-gateway.service openclaw-scout.service
info "openclaw-gateway + openclaw-scout démarrés ✓"

echo ""
info "Attente de l'enregistrement des outils MCP (~60s)..."
DEADLINE=$((SECONDS + 90))
while [[ $SECONDS -lt $DEADLINE ]]; do
  COUNT=$(journalctl --user -u openclaw-gateway.service --since "1 minute ago" 2>/dev/null \
    | grep -c "Registered: wiki_" || true)
  if [[ "$COUNT" -ge 6 ]]; then
    info "6 outils wiki_* enregistrés ✓"
    break
  fi
  sleep 5
done

if [[ $SECONDS -ge $DEADLINE ]]; then
  warn "Outils MCP non encore enregistrés après 90s — vérifier :"
  warn "  journalctl --user -u openclaw-gateway.service -n 50"
fi

echo ""
info "=== Services Secretarius opérationnels ==="
systemctl --user status openclaw-gateway.service openclaw-scout.service \
  openclaw-injection-guard.service --no-pager -l 2>&1 | grep -E "(Active|●)" || true
