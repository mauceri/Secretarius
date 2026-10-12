# openclaw-config/modules.sh — une fonction par module d'installation
# (spec 2026-10-12). Sourcé par install.sh APRÈS le socle (venv, openclaw.json,
# plugin) ; chaque module installe ses unités, images et espaces de travail,
# et ajoute ses services à SERVICES_A_VERIFIER ("unite|url", url facultative).
# Un module absent du profil n'est jamais appelé : il ne laisse aucune trace.

SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
SCRIPT_DIR="${SCRIPT_DIR:-${OPENCLAW_CONFIG_PATH}}"

# Construit une image Docker ; un échec arrête l'installation (spec : plus
# d'avertissement non bloquant, une installation sans image se croirait réussie).
_construire_image() {  # $1 = fichier Dockerfile, $2 = étiquette
  docker build -q -f "$1" -t "$2" "${SECRETARIUS_ROOT}" >/dev/null || {
    echo "[ERREUR] construction de l'image $2 échouée :" >&2
    echo "    docker build -f $1 -t $2 ${SECRETARIUS_ROOT}" >&2
    exit 1
  }
  info "image $2 ✓"
}

module_wiki() {
  ecrire_env_wiki || { echo "[ERREUR] Wiki_LM/.env : modèle du wiki « ${WIKI_LLM} » non appliqué" >&2; exit 1; }
  info "Wiki_LM/.env écrit (modèle ${WIKI_LLM}, pages web résumées par DeepSeek)"
  # Service Wiki_LM Query Server (Flask :5051, pour le template de requête Obsidian).
  # Installé quel que soit le profil ; activé seulement si le venv et server.py existent.
  WIKI_SERVER_DST="${SYSTEMD_USER_DIR}/wiki-lm-server.service"
  if [[ -f "$WIKI_SERVER_DST" && "$FORCE" != "true" ]]; then
    info "wiki-lm-server.service existe déjà — ignoré"
  else
    cp "${SCRIPT_DIR}/wiki-lm-server.service" "$WIKI_SERVER_DST"
    info "wiki-lm-server.service installé dans ${SYSTEMD_USER_DIR}"
  fi
  if [[ -x "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3" && -f "${HOME}/Secretarius/Wiki_LM/tools/server.py" ]]; then
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable wiki-lm-server.service 2>/dev/null && \
      info "wiki-lm-server.service activé au boot" || \
      warn "Activation de wiki-lm-server.service échouée"
  fi
  systemctl --user start wiki-lm-server.service 2>/dev/null || true
  SERVICES_A_VERIFIER+=("wiki-lm-server|http://127.0.0.1:5051/health")

  # Timer Wiki_LM embeddings : recalcul incrémental périodique (+ reload du serveur).
  for _wl in wiki-lm-embed.service wiki-lm-embed.timer; do
    _wl_dst="${SYSTEMD_USER_DIR}/${_wl}"
    if [[ -f "$_wl_dst" && "$FORCE" != "true" ]]; then
      info "${_wl} existe déjà — ignoré"
    else
      cp "${SCRIPT_DIR}/${_wl}" "$_wl_dst"
      info "${_wl} installé dans ${SYSTEMD_USER_DIR}"
    fi
  done
  if [[ -x "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3" && -f "${HOME}/Secretarius/Wiki_LM/tools/embed.py" ]]; then
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable --now wiki-lm-embed.timer 2>/dev/null && \
      info "wiki-lm-embed.timer activé (toutes les 6 h)" || \
      warn "Activation de wiki-lm-embed.timer échouée"
  fi

  _construire_image "${OPENCLAW_CONFIG_PATH}/Dockerfile.wiki" secretarius-wiki:latest
}

module_scout() {
  mkdir -p "${OPENCLAW_PATH}/workspace-scout/tasks/pending" \
           "${OPENCLAW_PATH}/workspace-scout/tasks/done" \
           "${OPENCLAW_PATH}/workspace-scout/results"
  info "Répertoires scout (tasks/results) créés"

  # injection-guard : filtre le contenu externe (motifs regex + DeBERTa) avant
  # toute lecture par un LLM. Installé et démarré AVANT scout-watcher : le
  # watcher l'appelle (via scout_process.py) dès la première tâche traitée.
  cp "${SCRIPT_DIR}/injection_guard.py" "${HOME}/.local/bin/injection_guard.py"
  chmod +x "${HOME}/.local/bin/injection_guard.py"
  info "injection_guard.py installé dans ${HOME}/.local/bin"
  GUARD_SVC_DST="${SYSTEMD_USER_DIR}/openclaw-injection-guard.service"
  if [[ -f "$GUARD_SVC_DST" && "$FORCE" != "true" ]]; then
    info "openclaw-injection-guard.service existe déjà — ignoré"
  else
    cp "${SCRIPT_DIR}/openclaw-injection-guard.service" "$GUARD_SVC_DST"
    info "openclaw-injection-guard.service installé dans ${SYSTEMD_USER_DIR}"
  fi
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now openclaw-injection-guard.service 2>/dev/null && \
    info "openclaw-injection-guard.service activé et démarré" || \
    warn "Activation de openclaw-injection-guard.service échouée"

  # scout-watcher : surveille workspace-scout/tasks/pending, pré-fetch les URLs
  # (curl côté watcher, pas d'exec dans scout) et signale scout via tasks/done.
  mkdir -p "${HOME}/.local/bin"
  cp "${SCRIPT_DIR}/scout-watcher" "${HOME}/.local/bin/scout-watcher"
  cp "${SCRIPT_DIR}/scout_process.py" "${HOME}/.local/bin/scout_process.py"
  chmod +x "${HOME}/.local/bin/scout-watcher" "${HOME}/.local/bin/scout_process.py"
  info "scout-watcher et scout_process.py installés dans ${HOME}/.local/bin"
  SCOUT_SVC_DST="${SYSTEMD_USER_DIR}/openclaw-scout.service"
  if [[ -f "$SCOUT_SVC_DST" && "$FORCE" != "true" ]]; then
    info "openclaw-scout.service existe déjà — ignoré"
  else
    cp "${SCRIPT_DIR}/openclaw-scout.service" "$SCOUT_SVC_DST"
    info "openclaw-scout.service installé dans ${SYSTEMD_USER_DIR}"
  fi
  systemctl --user daemon-reload 2>/dev/null || true
  # --now : sans lui, le watcher restait éteint jusqu'au prochain boot sur une
  # installation neuve (relevé par la revue DSH du 16/09/2026, start.sh ne le
  # relance pas non plus).
  systemctl --user enable --now openclaw-scout.service 2>/dev/null && \
    info "openclaw-scout.service activé et démarré" || \
    warn "Activation de openclaw-scout.service échouée"
  SERVICES_A_VERIFIER+=("openclaw-injection-guard" "openclaw-scout")
}

module_obfuscateur() {
  # Un seul proxy du Qwen3-14B obfusqué, sur docker0 (172.17.0.1:8001) :
  # joignable de l'hôte (agent main) comme des conteneurs (ingestion du wiki).
  # La clé de permutation et le tokenizer restent locaux ; Modal ne reçoit que
  # des identifiants permutés.
  verifier_proxy_obfuscateur || exit 1
  [[ -n "${ALOEPRI_API_KEY:-}" ]] || {
    echo "[ERREUR] ALOEPRI_API_KEY manquante (--env-file) : requise par le module obfuscateur" >&2
    exit 1
  }
  local obf_dir="${HOME}/obfuscator" venv="${WIKI_LM_PATH}/.venv"
  if [[ ! -d "$obf_dir" ]]; then
    info "Clonage de ${obf_dir} (proxy obfusqué, dépôt séparé de Secretarius)"
    git clone https://github.com/mauceri/obfuscator.git "$obf_dir"
  fi
  "${venv}/bin/pip" install -q fastapi uvicorn requests
  if [[ ! -f "${obf_dir}/artifacts/obfuscation_keys.json" ]]; then
    info "Régénération de la clé de permutation (déterministe, seed documentée)"
    mkdir -p "${obf_dir}/artifacts"
    OBF_DIR="$obf_dir" "${venv}/bin/python" - <<'PYEOF'
import json, os, random
V = 151936
rng = random.Random(0)
permuted = list(range(V)); rng.shuffle(permuted)
perm = dict(zip(range(V), permuted))
unperm = {v: k for k, v in perm.items()}
json.dump({"vocab_permutation": {str(k): v for k, v in perm.items()},
           "vocab_unpermute": {str(k): v for k, v in unperm.items()},
           "seed": 0}, open(os.path.join(os.environ["OBF_DIR"], "artifacts/obfuscation_keys.json"), "w"))
PYEOF
  fi
  # Comme le reste de l'installateur : rien n'est écrasé sans --force.
  mkdir -p "${HOME}/.config/obfuscator"
  if [[ ! -f "${HOME}/.config/obfuscator/proxy.env" || "$FORCE" == "true" ]]; then
    printf 'ALOEPRI_API_KEY=%s\n' "$ALOEPRI_API_KEY" > "${HOME}/.config/obfuscator/proxy.env"
    chmod 600 "${HOME}/.config/obfuscator/proxy.env"
  fi
  if [[ ! -f "${SYSTEMD_USER_DIR}/obfuscator-proxy.service" || "$FORCE" == "true" ]]; then
    cp "${SCRIPT_DIR}/obfuscator-proxy.service" "${SYSTEMD_USER_DIR}/obfuscator-proxy.service"
  fi
  systemctl --user daemon-reload
  systemctl --user enable --now obfuscator-proxy.service
  local pret=false
  for _ in $(seq 1 6); do
    curl -s -o /dev/null "http://172.17.0.1:8001/health" && { pret=true; break; }
    sleep 5
  done
  [[ "$pret" == true ]] || warn "Proxy obfusqué pas encore prêt après 30 s (réveil de Modal possible) : curl http://172.17.0.1:8001/health"
  export OBFUSQUE_BASE_URL="http://172.17.0.1:8001/v1"
  SERVICES_A_VERIFIER+=("obfuscator-proxy|http://172.17.0.1:8001/health")
}

module_laya() {
  # Routeur Tiron (tiron-router) — versionné, câblé au cerveau via tiron-router.env.
  cp "${SCRIPT_DIR}/tiron-router.service" "${SYSTEMD_USER_DIR}/tiron-router.service"
  # Env du routeur : checkpoint Laya + FAQ_PATH. Non écrasé s'il existe déjà.
  if [[ ! -f "${OPENCLAW_PATH}/tiron-router.env" || "$FORCE" == "true" ]]; then
    printf 'LAYA_CHECKPOINT=%s\nFAQ_PATH=%s\n' "$LAYA_CHECKPOINT" "${WIKI_PATH}/faits/faits.md" \
      > "${OPENCLAW_PATH}/tiron-router.env"
  else
    info "tiron-router.env déjà présent — conservé"
  fi
  python3 -m pip install --user --break-system-packages -q "laya[onnx]" 2>/dev/null && \
    info "laya[onnx] installé" || \
    warn "Installation de laya échouée — installez manuellement : python3 -m pip install --user --break-system-packages 'laya[onnx]'"

  # FAQ de faits (non-clobber : jamais écrasée, même --force)
  local faits_dest="${WIKI_PATH}/faits/faits.md"
  if [[ ! -f "$faits_dest" ]]; then
    mkdir -p "$(dirname "$faits_dest")"
    cp "${SECRETARIUS_ROOT}/amorçage/faits.md" "$faits_dest"
    info "FAQ de faits amorcée (${faits_dest})"
  else
    info "FAQ de faits déjà présente, conservée (${faits_dest})"
  fi

  # Activé ici, venv présent (le socle l'a créé avant les modules).
  systemctl --user daemon-reload
  systemctl --user enable --now tiron-router.service
  SERVICES_A_VERIFIER+=("tiron-router")
}

module_courrier() {
  # Relève de tiron@secretarius.co : unités copiées, activation laissée à l'utilisateur
  # (systemctl --user enable --now tiron-mail-releve.timer), cf. docs/components/courrier.md.
  for _tr in tiron-mail-releve.service tiron-mail-releve.timer; do
    _tr_dst="${SYSTEMD_USER_DIR}/${_tr}"
    if [[ -f "$_tr_dst" && "$FORCE" != "true" ]]; then
      info "${_tr} existe déjà — ignoré"
    else
      cp "${SCRIPT_DIR}/${_tr}" "$_tr_dst"
      info "${_tr} installé dans ${SYSTEMD_USER_DIR} (non activé)"
    fi
  done
}

module_cron() {
  # tiron_cron s'exécute depuis le dépôt : rien à installer ici. Le webhook de
  # livraison (cron.webhookToken d'openclaw.json) n'est pas encore dans le
  # gabarit ; à reprendre avec le profil lourd (morceau 4).
  info "cron : rien à installer hors du dépôt"
}

module_gog() {
  mkdir -p "${OPENCLAW_PATH}/workspace/.gog-config"
  info "Répertoire .gog-config créé"
  _construire_image "${OPENCLAW_CONFIG_PATH}/Dockerfile.gog" secretarius-gog:latest
}
