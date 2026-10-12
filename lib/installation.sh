# Fonctions de l'installateur, sourcées par install.sh (testées par
# tests_installation/test_installation_lib.py).

charger_profil() {  # $1 = leger | lourd
  local f="${SECRETARIUS_ROOT}/profils/$1.conf"
  [[ -f "$f" ]] || { echo "Profil inconnu : $1 (leger | lourd)" >&2; return 1; }
  # shellcheck source=/dev/null
  source "$f"
}

# Dossier des ZIM de Wikipédia, fournis par l'utilisateur (spec 2026-10-12).
resoudre_zim() {
  if [[ -n "${WIKI_ZIM_DIR:-}" ]]; then
    if compgen -G "${WIKI_ZIM_DIR}/*.zim" >/dev/null; then echo "$WIKI_ZIM_DIR"; return 0; fi
    echo "WIKI_ZIM_DIR=${WIKI_ZIM_DIR} : dossier absent ou sans fichier .zim" >&2
    return 1
  fi
  local defaut="${SECRETARIUS_ROOT}/Wiki_LM/zim"
  if compgen -G "${defaut}/*.zim" >/dev/null; then echo "$defaut"; return 0; fi
  if [[ "${INTERACTIVE:-false}" == true ]]; then
    local v; read -rp "Dossier des fichiers ZIM de Wikipédia (vide = aucun) : " v || true
    if [[ -n "$v" ]]; then WIKI_ZIM_DIR="$v" resoudre_zim; return $?; fi
  fi
  echo ""
}

# Pose KEY=VALUE dans un fichier .env (remplace la ligne, ou l'ajoute).
_poser_env() {  # $1 = fichier, $2 = clé, $3 = valeur
  if grep -q "^$2=" "$1"; then
    sed -i "s|^$2=.*|$2=$3|" "$1"
  else
    printf '%s=%s\n' "$2" "$3" >> "$1"
  fi
}

# Wiki_LM/.env selon le profil (spec 2026-10-12) : chemins, branche web sur
# DeepSeek, puis modèle du wiki appliqué par le code existant (op_switch_model).
ecrire_env_wiki() {
  local env="${SECRETARIUS_ROOT}/Wiki_LM/.env"
  if [[ ! -f "$env" || "${FORCE:-false}" == "true" ]]; then
    cp "${SECRETARIUS_ROOT}/Wiki_LM/.env.template" "$env"
  fi
  _poser_env "$env" WIKI_PATH "$WIKI_PATH"
  _poser_env "$env" WIKI_ZIM_DIR "$WIKI_ZIM_DIR"
  _poser_env "$env" WIKI_LOOKUP_OFFLINE 1
  _poser_env "$env" WIKI_VAULT_MIRRORS "${WIKI_VAULT_MIRRORS:-}"
  _poser_env "$env" WIKI_INGEST_WEB_LLM_BACKEND openai
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] && _poser_env "$env" DEEPSEEK_API_KEY "$DEEPSEEK_API_KEY"
  if [[ " ${MODULES} " == *" obfuscateur "* ]]; then
    export OBFUSQUE_BASE_URL="http://172.17.0.1:8001/v1"
    _poser_env "$env" OBFUSQUE_BASE_URL "$OBFUSQUE_BASE_URL"
  fi
  (cd "${SECRETARIUS_ROOT}/Wiki_LM/tools" && ../.venv/bin/python -c \
    "import sys, json, wiki; r = wiki.op_switch_model(sys.argv[1]); print(json.dumps(r)); sys.exit('error' in r)" \
    "$WIKI_LLM") >/dev/null
}
