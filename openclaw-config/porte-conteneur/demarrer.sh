#!/usr/bin/env bash
# Point d'entrée du conteneur : la porte seule (PORTE_HOTE:5052). Le wiki
# Publication n'y tourne plus depuis le 2026-11-25 (plus d'outils wiki).
set -euo pipefail
: "${VENV:?}" "${PORTE_DIR:?}"
cd "$(dirname "$PORTE_DIR")"
exec "$VENV/bin/python3" -m porte.serveur
