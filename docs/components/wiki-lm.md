---
tags: [documentation, LLM_Wiki, secretarius]
date: 2026-05-14
---

# Composant : wiki-lm

## Rôle

Pipeline complet d'une knowledge base personnelle basée sur le patron *LLM Wiki*
d'Andrej Karpathy. Un LLM ingère des sources (URLs, PDFs, textes, signets) et maintient
de façon incrémentale un wiki Markdown interconnecté : résumés, pages de concepts,
pages d'entités, clustering thématique, base de connaissance compactée.

## Prérequis

- Python 3.11+
- `pip install -r Wiki_LM/requirements.txt`
- Clé API DeepSeek (ou autre backend LLM) dans `Wiki_LM/.env`

## Installation

```bash
cd ~/Secretarius
python3 -m venv Wiki_LM/.venv
Wiki_LM/.venv/bin/pip install -r Wiki_LM/requirements.txt
cp Wiki_LM/.env.template Wiki_LM/.env
nano Wiki_LM/.env   # renseigner WIKI_PATH et DEEPSEEK_API_KEY
```

## Désinstallation

```bash
rm -rf ~/Secretarius/Wiki_LM/.venv
# Les données (wiki/, raw/, embeddings/, knowledge_base/) restent sous WIKI_PATH
```

## Configuration

Fichier `Wiki_LM/.env` (copié depuis `.env.template`) :

```
WIKI_LLM_BACKEND=openai
DEEPSEEK_API_KEY=<clé API>
OPENAI_BASE_URL=https://api.deepseek.com/v1
WIKI_PATH=/chemin/vers/coffre/Wiki_LM
```

Variables principales :

| Variable | Description |
|----------|-------------|
| `WIKI_PATH` | Répertoire contenant wiki/, raw/, embeddings/ |
| `WIKI_LLM_BACKEND` | `openai` (DeepSeek), `ollama`, `claude` |
| `DEEPSEEK_API_KEY` | Clé API DeepSeek |
| `OPENAI_BASE_URL` | URL du backend LLM compatible OpenAI |

## Usage des outils

Tous les outils utilisent le venv et chargent `.env` automatiquement :

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate
```

### Ingestion

**`ingest.py`** — Ingestion d'une source dans le wiki

```bash
python tools/ingest.py https://example.com/article
python tools/ingest.py chemin/vers/fichier.pdf
python tools/ingest.py --raw          # ingestion incrémentale depuis raw/
python tools/ingest.py --raw --force  # réingestion complète
```

Types supportés : `.url` (URL), `.md` (note), `.pdf` (PDF).

**`capture.py`** — Capture rapide (URLs, textes, fichiers, #tags)

```bash
python tools/capture.py "https://arxiv.org/abs/1706.03762"
python tools/capture.py "#attention Note importante sur les transformers"
python tools/capture.py --file /tmp/document.pdf "#recherche"
```

**`bookmarks_to_raw.py`** — Export des signets Brave vers raw/

```bash
python tools/bookmarks_to_raw.py   # lit ~/snap/brave/*/Bookmarks
```

### Recherche et consultation

**`search.py`** — Recherche BM25 rapide (sans LLM)

```bash
python tools/search.py "mémoire associative" --top 5
```

**`query.py`** — Interrogation en langage naturel (BM25 + LLM)

```bash
python tools/query.py "Comment fonctionne le Memex ?" --top 5
python tools/query.py "Karpathy et les wikis" --top 5 --save
```

**`lint.py`** — Health-check du wiki

```bash
python tools/lint.py   # détecte liens brisés, pages orphelines
```

**`server.py`** — Serveur Flask (port 5051) pour Obsidian

Interroge le wiki par HTTP. C'est le serveur qu'appelle le **template de requête
Obsidian** (voir `docs/components/obsidian.md`).

Géré par le service systemd **`wiki-lm-server.service`** (installé par `install.sh`
quel que soit le profil, activé au boot) :

```bash
systemctl --user status  wiki-lm-server      # état
systemctl --user restart wiki-lm-server      # redémarrer
journalctl --user -u wiki-lm-server -f       # logs
# Lancement manuel (debug) : python tools/server.py [--port 5051] [--mode hybrid]
```

Endpoints : `POST /query` `{question, top_k, mode}` → `{text, references, saved_slug}` ;
`GET /health` → `{status, pages}` ; `POST /reload` (reconstruit l'index) ;
`POST /embed`, `POST /cluster` (tâches de fond).

**Rechargement à chaud (auto-reload).** Le serveur charge son index au démarrage.
Pour que les documents **ingérés après** le lancement deviennent cherchables sans
redémarrage, un thread surveille le dossier `wiki/` et **reconstruit l'index dès
qu'un document change** (intervalle ~30 s, réglable via `WIKI_WATCH_INTERVAL`).
Un rechargement immédiat reste possible : `curl -X POST http://localhost:5051/reload`.
Note : l'index sémantique dépend des embeddings sur disque (`knowledge_base/embeddings/`) ;
relancer `embed.py` pour les rafraîchir. En mode `hybrid` (défaut), BM25 couvre les
documents récents même sans embeddings à jour.

### Embeddings et similarité

**`embed.py`** — Calcule les embeddings BGE-M3 pour toutes les pages

```bash
python tools/embed.py
```

**`dedup.py`** — Détection de doublons sémantiques

```bash
python tools/dedup.py --threshold 0.95
```

### Clustering

**`cluster.py`** — Clustering des pages sources

```bash
python tools/cluster.py --n-clusters 20
```

**`name_clusters.py`** — Nommage des clusters via LLM

```bash
python tools/name_clusters.py
```

### Base de connaissance

**`kb_update.py`** — Met à jour la base de connaissance depuis le wiki

```bash
python tools/kb_update.py
```

**`kb_query.py`** — Retourne les axes thématiques les plus proches

```bash
python tools/kb_query.py "apprentissage par renforcement" --top 5
```

**`kb_tags.py`** — Construit le dictionnaire de tags canoniques

```bash
python tools/kb_tags.py --algo transfers
```

## Notes d'architecture

Le wiki suit le patron LLM Wiki de Karpathy : les sources brutes (`raw/`) sont
immuables, le wiki est la représentation compilée maintenue par le LLM.
Voir [[PATTERN]] pour la description complète du patron.

La recherche utilise BM25 (pas de vectoriel au query time) ; les embeddings BGE-M3
servent uniquement au clustering et à la base de connaissance.
