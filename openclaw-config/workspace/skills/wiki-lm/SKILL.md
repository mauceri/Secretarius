---
name: wiki-lm
description: Interroger et gérer le wiki personnel Wiki_LM (Secretarius). Ingestion de sources, requêtes en langage naturel, lint du wiki. Déclencher sur /wiki-lm ou toute demande liée au wiki personnel.
---

# Skill : wiki-lm

## Rôle

Interroger le wiki personnel Wiki_LM (Secretarius) en langage naturel.

## Configuration

```
Wiki      : ${OBSIDIAN_PATH}/Wiki_LM/
Code      : ~/Secretarius/Wiki_LM/tools/
Venv      : ~/Secretarius/Wiki_LM/.venv/
Backend   : DeepSeek (OPENAI_BASE_URL=https://api.deepseek.com)
```

## Commandes disponibles

### Interroger le wiki

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && \
  python tools/query.py "<question>" --top 5
```

### Interroger et sauvegarder la synthèse comme page wiki

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && \
  python tools/query.py "<question>" --top 5 --save
```

### Recherche rapide BM25 (sans LLM)

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && \
  python tools/search.py "<mots-clés>" --top 5
```

### État du wiki (lint)

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && \
  python tools/lint.py
```

### Reconstruire le wiki depuis raw/

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && python tools/ingest.py --raw
```

Ingestion incrémentale : traite uniquement les fichiers de `raw/` non encore ingérés.

Pour reconstruire entièrement :

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate && python tools/ingest.py --raw --force
```

Types supportés : `.url` → URL, `.md` → note, `.pdf` → PDF

## Comportement

- Toujours commencer par une recherche BM25 pour identifier les pages pertinentes
- Si la question est précise → `query.py` avec synthèse LLM
- Si l'utilisateur trouve la réponse particulièrement intéressante → proposer `--save`
- Citer les pages sources avec [[slug]] dans la réponse
- Si aucun résultat BM25 → dire clairement que le wiki ne couvre pas encore ce sujet

## Déclencheurs typiques

- "Wiki : comment Shannon et Salton sont-ils reliés ?"
- "Qu'est-ce que le Memex ?"
- "Cherche dans le wiki : pensée associative"
- "Quel est l'état du wiki ?" → lint
- "Reconstruit le wiki" / "Ingère raw/" → ingest --raw
