# Porte Grok en conteneur (spec 2026-11-12-porte-sanroque). Image minimale :
# python3.12 = cible exacte des liens du .venv de Wiki_LM, monté au même chemin.
# Aucune dépendance Python ici : un changement de code se prend au redémarrage.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY demarrer.sh /usr/local/bin/demarrer.sh
ENTRYPOINT ["/usr/local/bin/demarrer.sh"]
