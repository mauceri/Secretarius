# Wiki_LM — Outils

Implémentation du pattern LLM Wiki (Andrej Karpathy).

## Architecture

- **Ce dépôt** (`~/Secretarius/Wiki_LM/`) : code des outils (ingest, query, lint, search)
- **Données** (`~/Documents/Arbath/Wiki_LM/`) : wiki Markdown + sources brutes, synchronisé via Obsidian Sync

## Configuration

```bash
export WIKI_PATH="$HOME/Documents/Arbath/Wiki_LM"
```

## Structure des données (Arbath/Wiki_LM)

```
raw/        ← sources brutes immutables
wiki/       ← pages Markdown LLM-maintained
index.md    ← catalogue du wiki (une ligne par page)
log.md      ← historique append-only des opérations
schema.md   ← conventions du wiki pour ce domaine
```

## Outils (tools/)

- `ingest.py`  — ingestion d'une source dans le wiki
- `query.py`   — interrogation du wiki
- `lint.py`    — health-check du wiki
- `search.py`  — recherche BM25 sur les pages
