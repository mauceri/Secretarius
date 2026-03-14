#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.config/systemd/user"

cp "$HOME/Secretarius/deploy/systemd-user/secretarius_ollama.service" \
  "$HOME/.config/systemd/user/"

cp "$HOME/Secretarius/deploy/systemd-user/secretarius_server.service" \
  "$HOME/.config/systemd/user/"

systemctl --user daemon-reload
systemctl --user enable --now secretarius_ollama.service
systemctl --user enable --now secretarius_server.service
systemctl --user status --no-pager secretarius_ollama.service
systemctl --user status --no-pager secretarius_server.service
