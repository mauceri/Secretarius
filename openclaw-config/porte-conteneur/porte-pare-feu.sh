#!/usr/bin/env bash
# Pare-feu du conteneur de la porte Grok (spec 2026-11-12-porte-sanroque).
# Depuis le pont br-porte, seul Ollama (hôte:11434) est joignable ; tout le reste
# (autres services de l'hôte, Internet) est rejeté. Lancé en root par
# porte-pare-feu.service ; `verifier` se passe de root (témoin dans /run).
set -euo pipefail

IPTABLES="${IPTABLES:-iptables}"
DOCKER="${DOCKER:-docker}"
TEMOIN="${TEMOIN:-/run/porte-pare-feu.ok}"
PONT="${PONT:-br-porte}"
PORT_OLLAMA="${PORT_OLLAMA:-11434}"

# Ordre final voulu dans chaque chaîne, de haut en bas.
regles() {
  echo "DOCKER-USER -i $PONT -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN"
  echo "DOCKER-USER -i $PONT -j DROP"
  echo "INPUT -i $PONT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT"
  echo "INPUT -i $PONT -p tcp --dport $PORT_OLLAMA -j ACCEPT"
  echo "INPUT -i $PONT -j DROP"
}

poser() {
  "$IPTABLES" -L DOCKER-USER -n >/dev/null 2>&1 || "$IPTABLES" -N DOCKER-USER
  # Insertion en tête, dans l'ordre inverse : l'ordre final est celui de regles().
  local chaine regle
  while read -r chaine regle; do
    # shellcheck disable=SC2086
    "$IPTABLES" -C "$chaine" $regle 2>/dev/null || "$IPTABLES" -I "$chaine" $regle
  done < <(regles | tac)
  touch "$TEMOIN"
  chmod 644 "$TEMOIN"
}

retirer() {
  local chaine regle
  # Fermer avant d'ouvrir : témoin d'abord (la relance par Restart=always échoue
  # alors à `verifier`), puis arrêt du conteneur, et seulement ensuite les règles.
  rm -f "$TEMOIN"
  "$DOCKER" stop porte-grok >/dev/null 2>&1 || true
  while read -r chaine regle; do
    # shellcheck disable=SC2086
    if "$IPTABLES" -C "$chaine" $regle 2>/dev/null; then "$IPTABLES" -D "$chaine" $regle; fi
  done < <(regles)
}

case "${1:-}" in
  poser) poser ;;
  retirer) retirer ;;
  verifier) [[ -f "$TEMOIN" ]] || { echo "porte-pare-feu : règles absentes ($TEMOIN)" >&2; exit 1; } ;;
  *) echo "usage : $0 poser|retirer|verifier" >&2; exit 2 ;;
esac
