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

# Vérifier que le binaire openclaw dans le service systemd est accessible
SERVICE_FILE="${HOME}/.config/systemd/user/openclaw-gateway.service"
if [[ -f "$SERVICE_FILE" ]]; then
  EXEC_BIN=$(grep '^ExecStart=' "$SERVICE_FILE" | sed 's/ExecStart=\([^ ]*\).*/\1/')
  if [[ -n "$EXEC_BIN" && ! -x "$EXEC_BIN" ]]; then
    error "Le binaire dans ExecStart est introuvable : $EXEC_BIN"
    error "Le fichier service a été généré sans NVM dans le PATH."
    error "Relancer : ./install.sh --force"
    exit 1
  fi
fi

# injection-guard
if systemctl --user is-enabled openclaw-injection-guard.service &>/dev/null 2>&1; then
  systemctl --user restart openclaw-injection-guard.service
  info "openclaw-injection-guard démarré ✓"
else
  warn "openclaw-injection-guard.service non activé — relancer install.sh pour l'installer"
fi

# Services MCP SSE (doivent être prêts avant le gateway)
for _svc in wiki-lm-mcp gog-mcp; do
  if systemctl --user is-enabled "${_svc}.service" &>/dev/null 2>&1; then
    systemctl --user restart "${_svc}.service"
    info "${_svc} démarré ✓"
  else
    warn "${_svc}.service non activé — relancer install.sh"
  fi
done
sleep 2

# gateway + scout
if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null; then
  error "TELEGRAM_BOT_TOKEN absent de ${OPENCLAW_PATH}/gateway.systemd.env"
  error "Renseigner les secrets puis relancer start.sh"
  exit 1
fi

# Propager DEEPSEEK_API_KEY vers Wiki_LM/.env si absent
WIKI_ENV="${SECRETARIUS_ROOT}/Wiki_LM/.env"
if [[ -f "$WIKI_ENV" ]] && grep -q "^DEEPSEEK_API_KEY=$" "$WIKI_ENV" 2>/dev/null; then
  GW_KEY=$(grep "^DEEPSEEK_API_KEY=" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null \
    | cut -d'=' -f2- | tr -d '"' || true)
  if [[ -n "$GW_KEY" ]]; then
    sed -i "s|^DEEPSEEK_API_KEY=.*|DEEPSEEK_API_KEY=${GW_KEY}|" "$WIKI_ENV"
    info "DEEPSEEK_API_KEY propagé vers Wiki_LM/.env ✓"
  fi
fi

# Étape 1 : démarrer le gateway
systemctl --user restart openclaw-gateway.service openclaw-scout.service
info "openclaw-gateway + openclaw-scout démarrés ✓"

sleep 5

# Vérifier que les services MCP répondent (transport streamable-http, endpoint /mcp)
for _port in 8901 8902; do
  _resp=$(curl -s -X POST --max-time 3 "http://127.0.0.1:${_port}/mcp" \
    -H 'Content-Type: application/json' -d '{}' 2>/dev/null | head -c 200)
  if [[ -n "$_resp" ]]; then
    info "MCP port ${_port} ✓"
  else
    warn "MCP port ${_port} ne répond pas — vérifier : journalctl --user -u wiki-lm-mcp -u gog-mcp -n 20"
  fi
done

echo ""
info "=== Services Secretarius opérationnels ==="
systemctl --user status openclaw-gateway.service openclaw-scout.service \
  openclaw-injection-guard.service wiki-lm-mcp.service gog-mcp.service \
  --no-pager -l 2>&1 | grep -E "(Active|●)" || true
