#!/usr/bin/env bash
# Point d'entrée du conteneur : serveur wiki Publication (127.0.0.1:5051 interne)
# puis porte (PORTE_HOTE:5052). Si l'un meurt, l'autre est arrêté et le
# conteneur sort avec son code : systemd relance (Restart=always).
set -uo pipefail
: "${VENV:?}" "${TOOLS:?}"

(cd "$TOOLS" && exec "$VENV/bin/python3" server.py --no-public --port 5051) &
wiki=$!
(cd "$(dirname "$TOOLS")" && exec "$VENV/bin/python3" -m porte.serveur) &
porte=$!

trap 'kill "$wiki" "$porte" 2>/dev/null' EXIT TERM INT
wait -n "$wiki" "$porte"
code=$?
exit "$code"
