# Secretarius - Reprise rapide

## État actuel (16 février 2026)
- Contrat JSON phase 1 implémenté:
  - `ordre_chunk` (index 0..n-1)
  - `chunk`
  - `expressions_caracteristiques`
- Pipeline locale en place (`chunking -> extraction`) avec validation runtime stricte.
- CLI extraction disponible: `python -m secretarius.cli`.
- Serveur HTTP local disponible:
  - `GET /health`
  - `POST /extract`
- Tests: `11 passed, 1 skipped`.

## Commandes pour reprendre
```bash
cd /home/mauceric/Secretarius
source ../secretarius_venv/bin/activate
python -m pytest
```

## Exécuter la CLI
```bash
python -m secretarius.cli \
  --text "Le camail est vert. Le voile est blanc." \
  --min-sentences 1 \
  --max-sentences 1 \
  --pretty
```

## Exécuter le serveur HTTP
```bash
python -m secretarius.serve --host 0.0.0.0 --port 8090
```

### Vérification rapide
```bash
curl -s http://localhost:8090/health
curl -s http://localhost:8090/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"Le camail est vert. Le voile est blanc.","min_sentences":1,"max_sentences":1}'
```

## Fichiers clés
- `secretarius/pipeline.py`: chunking, extraction, contrat JSON, validation.
- `secretarius/cli.py`: point d’entrée CLI extraction.
- `secretarius/server.py`: logique endpoint `/extract`.
- `secretarius/serve.py`: lancement serveur HTTP.
- `infra/milvus/compose.yml`: stack Milvus versionnée (etcd/minio/milvus).
- `infra/milvus/.env.example`: variables d'environnement Milvus.
- `infra/milvus/README.md`: démarrage et migration des données.
- `tests/test_pipeline_contract.py`: contrat et validation.
- `tests/test_cli_contract.py`: contrat via CLI.
- `tests/test_server_contract.py`: contrat via endpoint HTTP.

## Prochaines étapes recommandées
1. Ajouter un endpoint `POST /extract/batch` pour traiter plusieurs textes en une requête.
2. Ajouter retries/timeouts configurables et logs structurés côté extracteur HTTP.
3. Ajouter tests d’intégration smoke contre un vrai `llama-server` local (`RUN_LLAMA_SMOKE=1`).
4. Introduire schéma JSON versionné (ex: `contract_version`) pour geler l’API.
