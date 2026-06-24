#!/usr/bin/env bash
# gog-connect.sh — (Ré)autorise le compte Google de l'agent gog, en shell.
#
# Plus fiable que la commande Telegram /connecter (qui passe par un pont FIFO
# fragile). Lance le flux OAuth « manual » de gog dans le conteneur gog, avec
# --force-consent : chaque machine obtient ainsi SON PROPRE token (ne pas
# partager un même .gog-config entre machines, sinon Google les invalide
# mutuellement).
#
# Usage : ./gog-connect.sh [email]   (à défaut, lit GOG_ACCOUNT de l'environnement)
set -euo pipefail

GOG_CONFIG="${HOME}/.openclaw/workspace/.gog-config"
ACCOUNT="${1:-${GOG_ACCOUNT:-}}"

if [[ -z "$ACCOUNT" ]]; then
  echo "Usage: ./gog-connect.sh <email>   (ou définir GOG_ACCOUNT)" >&2
  exit 1
fi
if [[ ! -d "$GOG_CONFIG" ]]; then
  echo "[ERREUR] ${GOG_CONFIG} introuvable — installer Secretarius d'abord." >&2
  exit 1
fi
if ! docker image inspect secretarius-gog:latest &>/dev/null; then
  echo "[ERREUR] image secretarius-gog:latest absente — la construire d'abord." >&2
  exit 1
fi

echo "Autorisation de ${ACCOUNT} (Gmail + Drive + Agenda)…"
echo "  1. Ouvrez l'URL affichée ci-dessous dans un navigateur"
echo "  2. Connectez-vous et autorisez le compte"
echo "  3. Google redirige vers une URL http://localhost:1/... qui NE CHARGE PAS (normal)"
echo "  4. Copiez cette URL complète et collez-la au prompt 'Paste redirect URL'"
echo ""

exec docker run --rm -it \
  -v "${GOG_CONFIG}:/gog-config:rw" \
  -e XDG_CONFIG_HOME=/gog-config \
  -e GOG_KEYRING_BACKEND=file \
  -e GOG_ACCOUNT="$ACCOUNT" \
  secretarius-gog:latest \
  gog auth add "$ACCOUNT" --manual --force-consent --services gmail,drive,calendar
