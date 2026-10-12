#!/usr/bin/env bash
# openclaw-config/uninstall.sh — Désinstalle ce qu'install.sh a posé
set -euo pipefail

FORCE="${FORCE:-false}"
_i=0; _args=("$@")
while [[ $_i -lt ${#_args[@]} ]]; do
  case "${_args[$_i]}" in
    --yes) FORCE="true" ;;
  esac
  _i=$((_i+1))
done
unset _i _args

OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
GW_SVC="openclaw-gateway.service"

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

# Toutes les unités que les modules peuvent poser (openclaw-config/modules.sh) ;
# celles d'un module absent du profil sont simplement ignorées plus bas.
SERVICES=(
  "$GW_SVC"
  "wiki-lm-server.service"
  "wiki-lm-embed.timer"
  "wiki-lm-embed.service"
  "obfuscator-proxy.service"
  "openclaw-injection-guard.service"
  "openclaw-scout.service"
  "tiron-router.service"
  "tiron-mail-releve.timer"
  "tiron-mail-releve.service"
)
# Instances de synchronisation du coffre (obsidian-sync@<coffre>).
while read -r _inst _; do
  [[ -n "$_inst" ]] && SERVICES+=("$_inst")
done < <(systemctl --user list-units 'obsidian-sync@*' --all --plain --no-legend 2>/dev/null)
unset _inst

echo ""
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
# Unité modèle des instances obsidian-sync@ (ne contient pas de coffre).
rm -f "${SYSTEMD_USER_DIR}/obsidian-sync@.service"
systemctl --user daemon-reload 2>/dev/null || true
# Le coffre, ~/obfuscator et les fichiers ZIM ne sont jamais supprimés.

# Répertoire de configuration du gateway
if [[ -d "$OPENCLAW_PATH" ]]; then
  rm -rf "$OPENCLAW_PATH"
  info "${OPENCLAW_PATH} supprimé"
else
  info "${OPENCLAW_PATH} absent — ignoré"
fi

if [[ -f "${HOME}/.local/bin/switch-model" ]]; then
  rm -f "${HOME}/.local/bin/switch-model"
  info "switch-model supprimé"
fi

info "Désinstallation terminée."
