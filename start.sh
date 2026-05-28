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

# gateway + scout
if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+" "${OPENCLAW_PATH}/gateway.systemd.env" 2>/dev/null; then
  error "TELEGRAM_BOT_TOKEN absent de ${OPENCLAW_PATH}/gateway.systemd.env"
  error "Renseigner les secrets puis relancer start.sh"
  exit 1
fi

# Étape 1 : démarrer le gateway
systemctl --user restart openclaw-gateway.service openclaw-scout.service
info "openclaw-gateway + openclaw-scout démarrés ✓"

# Étape 2 : charger le plugin MCP adapter (OpenClaw ne le charge pas au démarrage
# depuis extensions/ ; il faut déclencher plugins install sur gateway vivant).
# plugins install écrit plugins.installs et cause un "supervisor restart" (exit 0).
sleep 5
ADAPTER_SRC="${SECRETARIUS_ROOT}/openclaw-config/openclaw-mcp-adapter"
if [[ -d "$ADAPTER_SRC" ]] && command -v openclaw &>/dev/null; then
  info "Chargement de openclaw-mcp-adapter..."
  openclaw plugins install --force "${ADAPTER_SRC}" 2>&1 | grep -E "(Installed|failed|WARN|warn)" || true
  # Re-synchroniser .bak pour éviter l'anti-clobber au prochain démarrage
  cp "${OPENCLAW_PATH}/openclaw.json" "${OPENCLAW_PATH}/openclaw.json.bak" 2>/dev/null || true
  # Étape 3 : relancer après le supervisor restart déclenché par plugins install
  systemctl --user restart openclaw-gateway.service openclaw-scout.service
  info "Gateway redémarré avec l'adaptateur MCP ✓"
else
  warn "openclaw-mcp-adapter introuvable ou openclaw absent — outils wiki_* non chargés"
fi

# Attendre l'enregistrement des outils dans le log fichier d'OpenClaw
echo ""
info "Attente de l'enregistrement des outils MCP (~60s)..."
OPENCLAW_LOG="/tmp/openclaw/openclaw-$(date +%Y-%m-%d).log"
LOG_BASELINE=$(wc -l < "$OPENCLAW_LOG" 2>/dev/null || echo 0)
DEADLINE=$((SECONDS + 90))
while [[ $SECONDS -lt $DEADLINE ]]; do
  COUNT=$(tail -n +"$((LOG_BASELINE + 1))" "$OPENCLAW_LOG" 2>/dev/null \
    | grep -c "Registered: wiki_" || echo 0)
  if [[ "${COUNT:-0}" -ge 6 ]]; then
    info "6 outils wiki_* enregistrés ✓"
    break
  fi
  sleep 5
done

if [[ $SECONDS -ge $DEADLINE ]]; then
  warn "Outils MCP non encore enregistrés après 90s — vérifier :"
  warn "  grep 'Registered: wiki_' $OPENCLAW_LOG"
fi

echo ""
info "=== Services Secretarius opérationnels ==="
systemctl --user status openclaw-gateway.service openclaw-scout.service \
  openclaw-injection-guard.service --no-pager -l 2>&1 | grep -E "(Active|●)" || true
