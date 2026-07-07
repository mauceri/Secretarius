# Capacités wiki de Secretarius

Le wiki (Wiki_LM) est la base de connaissances personnelle de l'utilisateur, stockée en fichiers Markdown (coffre Obsidian).

## Commandes
- /c <url|note> : capturer une page web ou une note dans le wiki.
- /ingest : lancer le traitement des captures en attente (opération asynchrone).
- /q <question> : interroger la base de connaissances et obtenir une synthèse.
- /source <url> : lire immédiatement une page web externe via l'agent Scout (protection anti-injection), sans la sauvegarder.
- /wikistatus : afficher l'état de l'ingestion du wiki.

## Fonctionnement
Les captures passent d'abord dans une file, puis l'ingestion extrait les expressions, calcule des plongements (embeddings) et met à jour la base interrogeable par /q.
