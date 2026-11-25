# Porte Grok en conteneur (spec 2026-11-12-porte-sanroque). Image minimale :
# python3 (3.12 sur 24.04) = cible exacte des liens du .venv (/usr/bin/python3), monté au même chemin.
# Aucune dépendance Python ici : un changement de code se prend au redémarrage.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY demarrer.sh /usr/local/bin/demarrer.sh
ENTRYPOINT ["/usr/local/bin/demarrer.sh"]
