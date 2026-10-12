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
