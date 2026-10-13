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
    # Valeur passée par l'environnement : ni « | » ni « & » ne cassent le remplacement.
    VALEUR="$3" CLE="$2" python3 - "$1" <<'PYEOF'
import os, re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(rf"^{re.escape(os.environ['CLE'])}=.*$", lambda m: f"{os.environ['CLE']}={os.environ['VALEUR']}", t, flags=re.M)
open(p, "w").write(t)
PYEOF
  else
    printf '%s=%s\n' "$2" "$3" >> "$1"
  fi
}

# N'ajoute KEY=VALUE que si la clé est absente (réinstallation sans --force).
_poser_env_si_absente() {  # $1 = fichier, $2 = clé, $3 = valeur
  grep -q "^$2=" "$1" || printf '%s=%s\n' "$2" "$3" >> "$1"
}

# Wiki_LM/.env selon le profil (spec 2026-10-12) : chemins, branche web sur
# DeepSeek, puis modèle du wiki appliqué par le code existant (op_switch_model).
# Sur un .env existant sans --force, on n'ajoute que les clés manquantes : une
# réinstallation ne doit ni effacer un miroir de coffre ni défaire un
# /switch-wiki-model.
ecrire_env_wiki() {
  local env="${SECRETARIUS_ROOT}/Wiki_LM/.env" poser=_poser_env neuf=false
  if [[ ! -f "$env" || "${FORCE:-false}" == "true" ]]; then
    cp "${SECRETARIUS_ROOT}/Wiki_LM/.env.template" "$env"
    neuf=true
  else
    poser=_poser_env_si_absente
  fi
  $poser "$env" WIKI_PATH "$WIKI_PATH"
  $poser "$env" WIKI_ZIM_DIR "$WIKI_ZIM_DIR"
  $poser "$env" WIKI_LOOKUP_OFFLINE 1
  $poser "$env" WIKI_VAULT_MIRRORS "${WIKI_VAULT_MIRRORS:-}"
  $poser "$env" WIKI_INGEST_WEB_LLM_BACKEND openai
  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then $poser "$env" DEEPSEEK_API_KEY "$DEEPSEEK_API_KEY"; fi
  if [[ " ${MODULES} " == *" obfuscateur "* ]]; then
    export OBFUSQUE_BASE_URL="http://172.17.0.1:8001/v1"
    $poser "$env" OBFUSQUE_BASE_URL "$OBFUSQUE_BASE_URL"
  fi
  [[ "$neuf" == true ]] || return 0
  local sortie
  sortie=$(cd "${SECRETARIUS_ROOT}/Wiki_LM/tools" && ../.venv/bin/python -c \
    "import sys, json, wiki; r = wiki.op_switch_model(sys.argv[1]); print(json.dumps(r)); sys.exit('error' in r)" \
    "$WIKI_LLM") || { echo "$sortie" >&2; return 1; }
}

# Une instance = un robot Telegram. Si la gateway est déjà installée avec un
# autre jeton, le changer exige --force (sinon l'ancien resterait en place).
verifier_jeton_telegram() {  # $1 = gateway.systemd.env
  [[ -f "$1" && "${FORCE:-false}" != "true" ]] || return 0
  local actuel
  actuel=$(grep '^TELEGRAM_BOT_TOKEN=' "$1" | cut -d= -f2-)
  if [[ -n "$actuel" && "$actuel" != "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    echo "Le jeton Telegram de --env-file diffère de celui déjà installé ($1)." >&2
    echo "Deux instances sur un même robot se volent les messages : vérifiez le jeton," >&2
    echo "puis relancez avec --force pour remplacer l'ancien." >&2
    return 1
  fi
}

# Ancienne installation (sanroque) : deux proxys, dont obfuscator-proxy-docker
# sur 172.17.0.1:8001. Le module installerait un second processus sur la même
# adresse : on s'arrête avec la marche à suivre.
verifier_proxy_obfuscateur() {
  local ancien="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}/obfuscator-proxy-docker.service"
  if [[ -f "$ancien" ]]; then
    echo "Ancien proxy présent : ${ancien} (déjà sur 172.17.0.1:8001)." >&2
    echo "Migration : désactiver obfuscator-proxy-docker et obfuscator-proxy, supprimer leurs" >&2
    echo "unités, faire pointer le fournisseur obfusque d'openclaw.json sur 172.17.0.1:8001," >&2
    echo "puis relancer install.sh (voir openclaw-config/INSTALL.md, « Migration »)." >&2
    return 1
  fi
}

# Vérification finale (spec 2026-10-12) : chaque entrée "unite|url" doit être
# active et, si une URL est donnée, répondre (trois essais : le proxy peut
# attendre le réveil de Modal). SYSTEMCTL, CURL et PAUSE sont remplaçables
# par les tests.
verifier_services() {
  local systemctl_cmd="${SYSTEMCTL:-systemctl --user}" curl_cmd="${CURL:-curl -sf --max-time 10 -o /dev/null}"
  local muets=() entree unite url ok
  for entree in "${SERVICES_A_VERIFIER[@]}"; do
    unite="${entree%%|*}"; url=""
    [[ "$entree" == *"|"* ]] && url="${entree#*|}"
    if ! $systemctl_cmd is-active --quiet "$unite"; then muets+=("$unite (inactif)"); continue; fi
    [[ -n "$url" ]] || continue
    ok=false
    for _ in 1 2 3; do
      $curl_cmd "$url" && { ok=true; break; }
      sleep "${PAUSE:-10}"
    done
    [[ "$ok" == true ]] || muets+=("$unite (ne répond pas sur $url)")
  done
  if [[ ${#muets[@]} -gt 0 ]]; then
    echo "Services muets :" >&2
    printf '  - %s\n' "${muets[@]}" >&2
    return 1
  fi
  echo "Tous les services répondent (${#SERVICES_A_VERIFIER[@]})."
}

# Ubuntu 26.04 monte /tmp en tmpfs à quota par utilisateur : le torch CUDA que
# PyPI sert par défaut (plusieurs Go de nvidia-*) l'épuise. Aucune machine
# Secretarius n'a de carte NVIDIA : torch CPU d'abord, et pip travaille sur disque.
installer_dependances_python() {  # $1 = pip du venv, $2 = requirements.txt
  local tmp="${HOME}/.cache/pip-tmp"
  mkdir -p "$tmp"
  TMPDIR="$tmp" "$1" install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
    && TMPDIR="$tmp" "$1" install --quiet -r "$2"
}
