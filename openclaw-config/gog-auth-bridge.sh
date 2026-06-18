#!/bin/sh
# Pont OAuth pour /connecter : lance `gog auth add --manual`, expose l'URL
# d'autorisation dans un fichier, attend l'URL de redirection recollée par
# l'utilisateur, l'injecte sur stdin de gog, puis signale la fin.
# Tout passe par $XDG_CONFIG_HOME (bind /gog-config), lu/écrit par le plugin.
set -e
CFG="${XDG_CONFIG_HOME:-/gog-config}"
EMAIL="${1:?usage: gog-auth-bridge.sh <email>}"
RAW="$CFG/auth_raw"; URL="$CFG/auth_url"; RESP="$CFG/auth_response"; DONE="$CFG/auth_done"
rm -f "$RAW" "$URL" "$RESP" "$DONE"
FIFO="$CFG/.auth_fifo"; rm -f "$FIFO"; mkfifo "$FIFO"

# gog lit l'URL de redirection sur stdin (FIFO), écrit ses prompts dans RAW.
gog auth add "$EMAIL" --manual --force-consent \
  --services gmail,drive,calendar < "$FIFO" > "$RAW" 2>&1 &
GOGPID=$!

# Attendre l'URL d'autorisation Google, puis l'exposer.
i=0
while ! grep -qE 'https://accounts\.google\.com/[^ ]+' "$RAW" 2>/dev/null; do
  sleep 0.5; i=$((i+1)); [ $i -gt 120 ] && { echo "timeout url" > "$DONE"; kill $GOGPID 2>/dev/null; exit 1; }
done
grep -oE 'https://accounts\.google\.com/[^ ]+' "$RAW" | head -1 > "$URL"

# Attendre la réponse de l'utilisateur (déposée par le plugin), puis l'injecter.
i=0
while [ ! -f "$RESP" ]; do
  sleep 1; i=$((i+1)); [ $i -gt 600 ] && { echo "timeout response" > "$DONE"; kill $GOGPID 2>/dev/null; exit 1; }
done
cat "$RESP" > "$FIFO"
if wait $GOGPID; then echo ok > "$DONE"; else echo error > "$DONE"; fi
rm -f "$FIFO"
