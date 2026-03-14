#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.config/systemd/user"

cp "$HOME/Secretarius/deploy/systemd-user/open-webui.service" \
  "$HOME/.config/systemd/user/"

systemctl --user daemon-reload
systemctl --user enable --now open-webui.service
systemctl --user status --no-pager open-webui.service
