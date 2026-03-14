#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.config/systemd/user"

cp "$HOME/Secretarius/deploy/systemd-user/secretarius-prototype-session-snapshot.service" \
  "$HOME/.config/systemd/user/"

cp "$HOME/Secretarius/deploy/systemd-user/secretarius-prototype-session-snapshot.timer" \
  "$HOME/.config/systemd/user/"

systemctl --user daemon-reload
systemctl --user enable --now secretarius-prototype-session-snapshot.timer
systemctl --user status --no-pager secretarius-prototype-session-snapshot.timer
