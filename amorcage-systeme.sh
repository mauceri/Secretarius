#!/usr/bin/env bash
# amorcage-systeme.sh — prérequis système de Secretarius, une fois par machine
# (spec 2026-10-12). Seul script qui demande sudo ; install.sh, lui, ne fait
# que vérifier. Idempotent : relancé, il ne refait que ce qui manque.
# Usage : sudo ./amorcage-systeme.sh        (Ubuntu 24.04 ou plus récent)
set -euo pipefail

if [[ $EUID -ne 0 || -z "${SUDO_USER:-}" ]]; then
  echo "À lancer avec sudo depuis le compte qui fera tourner Secretarius :" >&2
  echo "  sudo ./amorcage-systeme.sh" >&2
  exit 1
fi
UTILISATEUR="$SUDO_USER"
MAISON="$(getent passwd "$UTILISATEUR" | cut -d: -f6)"
etape() { echo ""; echo "==> $*"; }

etape "Paquets système (git, Python, gettext, curl, Docker)"
apt-get update -q
apt-get install -y -q git python3 python3-venv python3-pip gettext curl docker.io

etape "Groupe docker pour ${UTILISATEUR}"
usermod -aG docker "$UTILISATEUR"

if [[ -z "$(swapon --show)" ]]; then
  etape "Swap de 4 Go (/swapfile)"
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
else
  etape "Swap déjà présent : inchangé"
fi

if ! command -v tailscale >/dev/null; then
  etape "Tailscale (dépôt officiel)"
  curl -fsSL https://tailscale.com/install.sh | sh
else
  etape "Tailscale déjà installé"
fi

etape "NVM, Node LTS, openclaw et obsidian-headless pour ${UTILISATEUR}"
sudo -u "$UTILISATEUR" -H bash -c '
  set -euo pipefail
  export NVM_DIR="$HOME/.nvm"
  if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  fi
  source "$NVM_DIR/nvm.sh"
  nvm install --lts
  npm install -g openclaw obsidian-headless
'

etape "Services utilisateur sans session ouverte (linger)"
loginctl enable-linger "$UTILISATEUR"

cat <<EOF

Amorçage terminé. Reste à faire, en tant que ${UTILISATEUR} :
  1. Se déconnecter puis se reconnecter (le groupe docker n'est pris en compte
     qu'à la prochaine connexion).
  2. Relier la machine au tailnet :
       sudo tailscale up --hostname=<nom> --ssh
     puis, dans la console Tailscale, désactiver l'expiration de la clé de
     cette machine (sinon elle se déconnecte périodiquement).
  3. Lancer l'installation :
       cd ${MAISON}/Secretarius && ./install.sh --profil leger --env-file ~/.config/secrets.env
EOF
