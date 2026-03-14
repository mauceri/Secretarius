#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.config/systemd/user"

cp "$HOME/Secretarius/deploy/systemd-user/milvus-compose.service" \
  "$HOME/.config/systemd/user/"

systemctl --user daemon-reload
systemctl --user enable --now milvus-compose.service
systemctl --user status --no-pager milvus-compose.service
