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
- Tests: `python -m pytest`

## Démarrage rapide
```bash
cd /home/mauceric/Secretarius
source .venv/bin/activate
python -m pytest
```

## Milvus (infra)
La configuration Milvus est versionnée dans `infra/milvus`, avec données persistantes hors dépôt.
