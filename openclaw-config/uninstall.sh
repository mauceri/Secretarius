#!/usr/bin/env bash
# openclaw-config/uninstall.sh — Désinstalle ce qu'install.sh a posé
set -euo pipefail

FORCE="${FORCE:-false}"
PROFILE="${PROFILE:-prod}"
_i=0; _args=("$@")
while [[ $_i -lt ${#_args[@]} ]]; do
  case "${_args[$_i]}" in
    --yes)     FORCE="true" ;;
    --profile) _i=$((_i+1)); PROFILE="${_args[$_i]:-prod}" ;;
  esac
  _i=$((_i+1))
done
unset _i _args

if [[ "$PROFILE" == "slm" ]]; then
  OPENCLAW_PATH="$HOME/.openclaw-slm"
  GW_SVC="openclaw-gateway-slm.service"
else
  OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
  GW_SVC="openclaw-gateway.service"
fi

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

SERVICES=(
  "$GW_SVC"
  "wiki-lm-server.service"
  "wiki-lm-embed.timer"
  "wiki-lm-embed.service"
)

echo ""
echo "Profil   : $PROFILE"
echo "Cible    : $OPENCLAW_PATH"
echo "Services : ${SERVICES[*]}"
echo ""

if [[ "$FORCE" != "true" ]]; then
  read -r -p "Confirmer la désinstallation ? [y/N] " _ans
  [[ "$_ans" =~ ^[Yy]$ ]] || { echo "Annulé."; exit 0; }
  unset _ans
fi

# Arrêt et désactivation des services
for _svc in "${SERVICES[@]}"; do
  if systemctl --user is-active "$_svc" &>/dev/null; then
    systemctl --user stop "$_svc" 2>/dev/null && info "$_svc arrêté" || warn "Arrêt de $_svc échoué"
  fi
  if systemctl --user is-enabled "$_svc" &>/dev/null; then
    systemctl --user disable "$_svc" 2>/dev/null && info "$_svc désactivé" || warn "Désactivation de $_svc échouée"
  fi
  _svc_file="${SYSTEMD_USER_DIR}/${_svc}"
  if [[ -f "$_svc_file" ]]; then
    rm -f "$_svc_file"
    info "$_svc_file supprimé"
  fi
done
systemctl --user daemon-reload 2>/dev/null || true

# Répertoire de configuration du gateway
if [[ -d "$OPENCLAW_PATH" ]]; then
  rm -rf "$OPENCLAW_PATH"
  info "${OPENCLAW_PATH} supprimé"
else
  info "${OPENCLAW_PATH} absent — ignoré"
fi

# switch-model
if [[ -f "${HOME}/.local/bin/switch-model" ]]; then
  rm -f "${HOME}/.local/bin/switch-model"
  info "switch-model supprimé"
fi

info "Désinstallation terminée."
