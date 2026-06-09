#!/bin/sh
KEYRING_PASSWORD_FILE="${XDG_CONFIG_HOME:-/workspace/.gog-config}/keyring-password"
if [ -f "$KEYRING_PASSWORD_FILE" ]; then
  export GOG_KEYRING_PASSWORD="$(cat "$KEYRING_PASSWORD_FILE")"
fi
exec /usr/local/bin/gog-bin "$@"
