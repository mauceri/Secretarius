#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/mauceric/Secretarius"
PROTOTYPE_DIR="$REPO_ROOT/Prototype"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

cd "$PROTOTYPE_DIR"

"$VENV_PYTHON" session_resume.py snapshot-auto \
  --title "Auto snapshot VSCode" \
  --summary "Snapshot periodique du contexte Prototype pour reprise locale." \
  --next-step "Lancer make resume-last avant de reprendre le travail." \
  --notes "Snapshot cree automatiquement par timer systemd --user." \
  --min-interval-seconds 1800
