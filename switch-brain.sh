#!/usr/bin/env bash
# switch-brain — bascule le cerveau LLM de Tiron (provider + routeur) vers un
# endpoint nommé du registre ~/.openclaw/brains.env, puis redémarre.
set -euo pipefail

BRAINS_ENV="${BRAINS_ENV:-$HOME/.openclaw/brains.env}"
OPENCLAW_JSON="${OPENCLAW_JSON:-$HOME/.openclaw/openclaw.json}"
ROUTER_ENV="${ROUTER_ENV:-$HOME/.openclaw/tiron-router.env}"

usage() { echo "Usage: switch-brain <sanroque|modal>" >&2; exit 1; }
[ $# -eq 1 ] || usage
NAME="$1"; UP="$(printf '%s' "$NAME" | tr '[:lower:]' '[:upper:]')"

[ -f "$BRAINS_ENV" ] || { echo "brains.env introuvable : $BRAINS_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$BRAINS_ENV"

url_var="BRAIN_${UP}_URL"; key_var="BRAIN_${UP}_KEY"; keyfile_var="BRAIN_${UP}_KEY_FILE"
URL="${!url_var:-}"
[ -n "$URL" ] || { echo "Cerveau inconnu : $NAME" >&2; usage; }
KEY="${!key_var:-}"
KEYFILE="${!keyfile_var:-}"
if [ -z "$KEY" ] && [ -n "$KEYFILE" ]; then
  KEYFILE="${KEYFILE/#\~/$HOME}"
  [ -f "$KEYFILE" ] || { echo "Clé du cerveau $NAME absente : $KEYFILE" >&2; exit 1; }
  KEY="$(cat "$KEYFILE")"
fi
APIKEY="${KEY:-local}"

# 1. provider openclaw.json (python = robuste) + sync .bak
OPENCLAW_JSON="$OPENCLAW_JSON" URL="$URL" APIKEY="$APIKEY" python3 - <<'PY'
import json, os, shutil
p = os.environ["OPENCLAW_JSON"]
c = json.load(open(p))
prov = c["models"]["providers"]["tiron-llm"]
prov["baseUrl"] = os.environ["URL"].rstrip("/") + "/v1"
prov["apiKey"] = os.environ["APIKEY"]
json.dump(c, open(p, "w"), indent=1, ensure_ascii=False)
open(p, "a").write("\n")
shutil.copy(p, p + ".bak")
PY

# 2. routeur : maj des 2 vars, préserve le reste (WIKI_PATH…)
touch "$ROUTER_ENV"
_set() {  # fichier clé valeur
  local f="$1" k="$2" v="$3" tmp
  tmp="$(mktemp)"
  awk -v k="$k" -v v="$v" '
    $0 ~ "^" k "=" { print k "=" v; done=1; next }
    { print }
    END { if (!done) print k "=" v }
  ' "$f" > "$tmp"
  mv "$tmp" "$f"
}
_set "$ROUTER_ENV" TIRON_LLAMA_BASE "$URL"
_set "$ROUTER_ENV" TIRON_LLAMA_KEY "$KEY"

echo "Cerveau actif : $NAME ($URL)"

# 3. restart (sautable en test)
if [ "${SWITCH_BRAIN_NO_RESTART:-}" != "1" ]; then
  systemctl --user restart openclaw-gateway tiron-router 2>/dev/null || \
    echo "Redémarrez manuellement : systemctl --user restart openclaw-gateway tiron-router" >&2
fi
