#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.config/secretarius"
mkdir -p "$HOME/.config/systemd/user"

cp "$HOME/Secretarius/deploy/env/openwebui-api.env.example" \
  "$HOME/.config/secretarius/openwebui-api.env"

cp "$HOME/Secretarius/deploy/systemd-user/secretarius-openwebui-api.service" \
  "$HOME/.config/systemd/user/"

systemctl --user daemon-reload
systemctl --user enable --now secretarius-openwebui-api.service
systemctl --user status --no-pager secretarius-openwebui-api.service
