#!/usr/bin/env bash
# install.sh — Installation idempotente de Secretarius, par profil (spec 2026-10-12)
# Usage: ./install.sh --profil leger|lourd [--env-file FILE] [--obsidian-path PATH]
#                     [--assistant-name NAME] [--openclaw-path PATH] [--interactive] [--force] [--help]
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SECRETARIUS_ROOT}/install.conf"
source "${SECRETARIUS_ROOT}/lib/installation.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

[[ -t 0 ]] && INTERACTIVE=true || INTERACTIVE=false
FORCE=false
ENV_FILE=""
PROFIL=""

usage() {  # $1 = code de sortie
  cat << 'EOF'
Usage: ./install.sh --profil leger|lourd [options]

  --profil NOM            leger (modèles déportés, santiago) | lourd (type sanroque) — obligatoire
  --env-file FILE         Secrets et réglages de la machine (clés, jeton Telegram, WIKI_ZIM_DIR…)
  --obsidian-path PATH    Coffre Obsidian (défaut : ~/Documents/<coffre du profil>)
  --assistant-name NAME   Nom de l'assistant (défaut: Tiron)
  --openclaw-path PATH    Chemin config OpenClaw (défaut: ~/.openclaw)
  --interactive           Pose les questions une par une
  --force                 Écrase les fichiers existants
  --help
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profil)         PROFIL="$2"; shift 2 ;;
    --obsidian-path)  OBSIDIAN_PATH="$2"; shift 2 ;;
    --assistant-name) ASSISTANT_NAME="$2"; shift 2 ;;
    --openclaw-path)  OPENCLAW_PATH="$2"; shift 2 ;;
    --env-file)       ENV_FILE="$2"; shift 2 ;;
    --interactive)    INTERACTIVE=true; shift ;;
    --force)          FORCE=true; shift ;;
    --help)           usage 0 ;;
    *) error "Option inconnue: $1"; usage 2 ;;
  esac
done

[[ -n "$PROFIL" ]] || { error "--profil est obligatoire"; usage 2; }

# Charger les secrets AVANT le profil : le profil n'écrit que des défauts
# (${VAR:-…}), une valeur explicite du fichier garde donc la priorité.
# set -a pour EXPORTER toutes les variables (sous-scripts).
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { error "Fichier introuvable: $ENV_FILE"; exit 1; }
  set -a; source "$ENV_FILE"; set +a
fi
charger_profil "$PROFIL" || exit 1
info "Profil ${PROFIL} : modules ${MODULES}"

# Étape 1 — Prérequis (avant toute question interactive). Tous bloquants :
# ils s'installent avec amorcage-systeme.sh (sudo, une fois).
info "Vérification des prérequis..."
WARNINGS=()

if [[ "$SECRETARIUS_ROOT" != "$HOME/Secretarius" ]]; then
  error "Le dépôt doit se trouver dans ~/Secretarius (trouvé : ${SECRETARIUS_ROOT})"
  exit 1
fi

if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
  info "Python $(python3 --version | cut -d' ' -f2) ✓"
else
  error "Python 3.11+ requis — lancer : sudo ./amorcage-systeme.sh"
  exit 1
fi

for _cmd in git envsubst curl; do
  command -v "$_cmd" &>/dev/null || { error "$_cmd requis — lancer : sudo ./amorcage-systeme.sh"; exit 1; }
done
unset _cmd
info "git, envsubst, curl ✓"

# Sourcer NVM si nécessaire (sessions SSH non-interactives ne chargent pas .bashrc)
if ! command -v openclaw &>/dev/null; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh" 2>/dev/null || true
fi

# Tester openclaw avec un timeout : un binaire résiduel (paquet npm désinstallé)
# reste dans le PATH mais part en boucle sur --version → l'install se figeait.
_oc_ver="$(timeout 15 openclaw --version 2>/dev/null | head -1 || true)"
if [[ -n "$_oc_ver" ]]; then
  info "openclaw ${_oc_ver} ✓"
else
  error "openclaw absent ou non fonctionnel — lancer : sudo ./amorcage-systeme.sh"
  echo "  (NE PAS utiliser l'installeur curl install-cli.sh : il place openclaw"
  echo "   dans ~/.openclaw, en collision avec la config Secretarius.)"
  exit 1
fi
unset _oc_ver

if ! command -v docker &>/dev/null || ! docker ps &>/dev/null; then
  error "Docker inaccessible (absent, ou utilisateur hors du groupe docker)."
  error "  Lancer : sudo ./amorcage-systeme.sh, puis se reconnecter."
  exit 1
fi
info "docker $(docker --version | cut -d' ' -f3 | tr -d ',') ✓"

if ! systemctl --user status &>/dev/null; then
  error "systemd utilisateur indisponible — requis pour les services"
  exit 1
fi

for _var in TELEGRAM_BOT_TOKEN TELEGRAM_OWNER_ID; do
  if [[ -z "${!_var:-}" ]]; then
    error "${_var} manquant dans --env-file (chaque instance a son propre robot Telegram)."
    exit 1
  fi
done
unset _var

# Étape 2 — Coffre Obsidian : défaut dérivé du profil.
OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/${COFFRE}}"
if [[ "$INTERACTIVE" == true ]]; then
  echo "=== Installation Secretarius (profil ${PROFIL}) ==="
  read -rp "Coffre Obsidian [${OBSIDIAN_PATH}]: " v || true
  OBSIDIAN_PATH="${v:-$OBSIDIAN_PATH}"
  read -rp "Nom de l'assistant [${ASSISTANT_NAME}]: " v || true
  ASSISTANT_NAME="${v:-$ASSISTANT_NAME}"
  unset v
fi
OBSIDIAN_PATH="${OBSIDIAN_PATH/#\~/$HOME}"
mkdir -p "$OBSIDIAN_PATH"
info "Coffre Obsidian ✓ (${OBSIDIAN_PATH})"
WIKI_PATH="${OBSIDIAN_PATH}/Wiki_LM"

# Étape 2b — Copie locale de Wikipédia (ZIM fournis par l'utilisateur).
WIKI_ZIM_DIR="$(resoudre_zim)" || exit 1
if [[ -z "$WIKI_ZIM_DIR" ]]; then
  WIKI_ZIM_DIR="${SECRETARIUS_ROOT}/Wiki_LM/zim"
  mkdir -p "$WIKI_ZIM_DIR"   # dossier vide : le montage du conteneur reste valide
  ZIM_ETAT="absent : enrichissement par Wikipédia désactivé"
else
  ZIM_ETAT="$(cd "$WIKI_ZIM_DIR" && ls -1 *.zim | tr '\n' ' ')(${WIKI_ZIM_DIR})"
fi

OPENCLAW_PATH="${OPENCLAW_PATH/#\~/$HOME}"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
export SECRETARIUS_ROOT OBSIDIAN_PATH WIKI_PATH ASSISTANT_NAME OPENCLAW_PATH FORCE INTERACTIVE \
       MODULES MAIN_BRAIN WIKI_LLM COFFRE WIKI_ZIM_DIR TELEGRAM_BOT_TOKEN TELEGRAM_OWNER_ID SYSTEMD_USER_DIR

# Étape 3 — Socle OpenClaw (openclaw.json élagué, gateway, enveloppes, espaces de travail)
info "Génération de la configuration OpenClaw..."
export OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
export GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
bash "${SECRETARIUS_ROOT}/openclaw-config/install.sh"

# Étape 4 — Wiki_LM/.env : écrit par le module wiki (ecrire_env_wiki), venv présent.

# Étape 5 — Dépendances Python (venv). Bloquant : les services en dépendent.
info "Installation des dépendances Python..."
WIKI_LM_PATH="${SECRETARIUS_ROOT}/Wiki_LM"
VENV_PATH="${WIKI_LM_PATH}/.venv"
python3 -m venv "$VENV_PATH" || { error "venv impossible — lancer : sudo ./amorcage-systeme.sh"; exit 1; }
"${VENV_PATH}/bin/pip" install -r "${WIKI_LM_PATH}/requirements.txt" --quiet \
  || { error "dépendances Python non installées : ${VENV_PATH}/bin/pip install -r ${WIKI_LM_PATH}/requirements.txt"; exit 1; }
info "Dépendances Python ✓ (venv: ${VENV_PATH})"

# Étape 5a — Plugin derisk-deleg : copie automatique (dist committé à jour).
PLUGIN_SRC="${SECRETARIUS_ROOT}/derisk-deleg"
PLUGIN_DST="${OPENCLAW_PATH}/extensions/derisk-deleg"
[[ -d "${PLUGIN_SRC}/dist" ]] || { error "derisk-deleg/dist absent — construire le plugin (npm run build)"; exit 1; }
mkdir -p "$PLUGIN_DST"
cp -r "${PLUGIN_SRC}/dist" "${PLUGIN_SRC}/node_modules" \
      "${PLUGIN_SRC}/openclaw.plugin.json" "${PLUGIN_SRC}/package.json" "$PLUGIN_DST/"
info "plugin derisk-deleg copié ✓"

# Étape 5b — Images du socle : image de base (openclaw >= 2026.7 ne la livre plus)
# et image de l'agent main. Les images des modules sont construites par les modules.
source "${SECRETARIUS_ROOT}/openclaw-config/modules.sh"
if ! docker image inspect openclaw-sandbox:bookworm-slim &>/dev/null; then
  info "Image de base openclaw-sandbox:bookworm-slim absente — construction..."
  docker build -q -f "${OPENCLAW_CONFIG_PATH}/Dockerfile.sandbox-base" \
       -t openclaw-sandbox:bookworm-slim "${OPENCLAW_CONFIG_PATH}" >/dev/null \
    || { error "construction de openclaw-sandbox:bookworm-slim échouée"; exit 1; }
  info "image openclaw-sandbox:bookworm-slim ✓"
fi
_construire_image "${OPENCLAW_CONFIG_PATH}/Dockerfile.tiron" secretarius-tiron:latest

# Étape 6 — Modules du profil
SERVICES_A_VERIFIER=("openclaw-gateway")
for _m in $MODULES; do
  declare -F "module_${_m}" >/dev/null || { error "Module inconnu : ${_m}"; exit 1; }
  info "--- module ${_m} ---"
  "module_${_m}"
done
unset _m

# Étape 7 — Gateway
systemctl --user daemon-reload
systemctl --user enable openclaw-gateway.service
systemctl --user restart openclaw-gateway.service
info "openclaw-gateway démarré ✓"

# Résumé
echo ""
info "=== Installation terminée (profil ${PROFIL}) ==="
echo "  Modules   : ${MODULES}"
echo "  Coffre    : ${OBSIDIAN_PATH}"
echo "  Wikipédia : ${ZIM_ETAT}"
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  echo ""
  warn "Points d'attention :"
  for w in "${WARNINGS[@]}"; do
    echo -e "  - ${w}"
  done
fi
echo ""
echo "Étapes de compte (une fois) :"
echo "  - Appairer Telegram : envoyer /start au robot, puis"
echo "       openclaw pairing approve telegram <CODE>"
