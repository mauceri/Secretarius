# Secretarius

Secretarius est un assistant local et frugal orienté mémoire documentaire:
- découpage sémantique d'un texte en chunks;
- extraction d'expressions caractéristiques par chunk;
- contrat JSON stable prêt pour une base sémantique.

## État actuel
- CLI locale d'extraction: `python -m secretarius.cli`
- Serveur HTTP local:
  - `GET /health`
  - `POST /extract`
- Serveur MCP (stubs): `python -m secretarius.mcp_serve`
- Tests: `python -m pytest`

## Démarrage rapide
```bash
cd /home/mauceric/Secretarius
source .venv/bin/activate
python -m pytest
```

## MCP

Le serveur MCP expose trois outils:
- `extract_expressions`
- `expressions_to_embeddings`
- `semantic_graph_search`

Etat courant:
- `extract_expressions`: implemente (llama.cpp + chunking)
- `expressions_to_embeddings`: implemente (multilingue, dimension 384)
- `semantic_graph_search`: implemente (Milvus, mode unifie insertion/recherche)

Documentation d'integration OpenClaw: `docs_mcp_openclaw.md`.
Documentation runtime multi-canaux (Open WebUI + extensibilite Telegram/WhatsApp/Email): `docs_channels.md`.
Connecteur Telegram pret a l'emploi: `serverTelegram.py`.
Connecteur Email (IMAP/SMTP) pret a l'emploi: `serverEmail.py`.
Units `systemd --user` prêtes: `deploy/systemd-user/`.
Inclut aussi l'API OpenAI pour Open WebUI: `deploy/systemd-user/secretarius-openwebui-api.service`.

## Milvus (infra)
La configuration Milvus est versionnée dans `infra/milvus`, avec données persistantes hors dépôt.
