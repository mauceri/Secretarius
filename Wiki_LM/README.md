# Wiki_LM

Implémentation du patron *LLM Wiki* (Andrej Karpathy) : un wiki Markdown personnel maintenu de façon incrémentale par un LLM. Chaque source ingérée est résumée, ses concepts et entités identifiés et enrichis, et le tout est interconnecté via des liens internes.

Voir `PATTERN.md` pour la description complète du patron et de ses cas d'usage.

---

## Architecture

```
Wiki_LM/
├── tools/              ← pipeline d'ingestion et outils CLI
├── tests/              ← suite pytest (170 tests)
├── wiki/               ← pages Markdown (données, hors dépôt)
├── raw/                ← sources brutes (données, hors dépôt)
├── embeddings/         ← vecteurs BGE-M3 (données, hors dépôt)
├── knowledge_base/     ← base de connaissance compactée (données, hors dépôt)
├── zim/                ← fichiers Kiwix ZIM optionnels (hors dépôt)
├── PATTERN.md          ← description du patron
└── requirements.txt
```

Les données (`wiki/`, `raw/`, `embeddings/`, `knowledge_base/`, `zim/`) sont hors dépôt.  
Elles vivent sous le coffre Obsidian : `$WIKI_PATH/wiki/`, `$WIKI_PATH/../raw/`, etc.  
`WIKI_PATH` est lu depuis `Wiki_LM/.env` (chargé automatiquement à l'import de `wiki_paths`).

Chaque wiki contient deux fichiers d'index des tags générés par `ingest.py` :

- `tags.md` — index complet avec renvois `[[src-*]]`, trié alphabétiquement par tag
- `liste_mots_clés.md` — liste des tags seuls, triés par fréquence décroissante

---

## Outils (`tools/`)

### Ingestion

| Outil | Description |
|-------|-------------|
| `ingest.py` | Ingestion d'une source (URL, PDF, texte) dans le wiki |
| `capture.py` | Capture rapide depuis OpenClaw / CLI (URLs, texte, `#tags`, fichiers joints) |
| `bookmarks_to_raw.py` | Export des signets Brave vers `raw/` |
| `build_wiki_cache.py` | Préchauffage du cache Wikipedia sur les pages existantes |
| `summarize.py` | Résumé d'un document via LLM |
| `build_summary_corpus.py` | Construction d'un corpus de résumés |

### Recherche et consultation

| Outil | Description |
|-------|-------------|
| `query.py` | Interrogation du wiki en langage naturel |
| `search.py` | Recherche BM25 sur les pages |
| `server.py` | Serveur Flask pour interroger le wiki depuis Obsidian |
| `lint.py` | Health-check : liens brisés, pages orphelines |
| `wiki_lookup.py` | Lookup Wikipedia (ZIM → cache SQLite → API REST) |

### Embeddings et similarité

| Outil | Description |
|-------|-------------|
| `embed.py` | Calcule et persiste les embeddings BGE-M3 pour toutes les pages |
| `similarity.py` | Calcul de matrices de similarité entre pages (`src-`) |
| `dedup.py` | Détection et nettoyage de doublons sémantiques |

### Clustering

| Outil | Description |
|-------|-------------|
| `cluster.py` | Clustering des pages `src-` (algo des transferts ou k-means) |
| `transfers.py` | Algorithme des transferts (O(k × N × C)) |

### Base de connaissance

| Outil | Description |
|-------|-------------|
| `kb_update.py` | Met à jour la base de connaissance depuis un wiki archivé |
| `kb_query.py` | Retourne les axes thématiques les plus proches d'un vecteur |
| `kb_tags.py` | Construit le dictionnaire de tags canoniques (`--algo greedy\|transfers`) |
| `name_clusters.py` | Renomme les clusters via LLM (titre thématique + description) |

### Utilitaires

| Outil | Description |
|-------|-------------|
| `wiki_paths.py` | Navigation dans la structure du wiki |
| `llm.py` | Abstraction LLM (DeepSeek / Anthropic / Ollama) |

> Les scripts `migrate_wiki_structure.py`, `patch_lien_source.py`, `patch_src_slugs.py` et `patch_wiki_abstracts.py` sont des migrations ponctuelles, sans vocation à être relancés.

---

## Démarrage rapide

`install.sh` crée `.venv` et installe les dépendances. Activer l'environnement :

```bash
cd ~/Secretarius/Wiki_LM
source .venv/bin/activate

# Ingérer une URL
python tools/ingest.py https://example.com/article

# Ingérer tout raw/ (mode batch)
python tools/ingest.py --raw

# Forcer la reconstruction complète du wiki
python tools/ingest.py --raw --force

# Rechercher dans le wiki
python tools/search.py "large language model"

# Interroger en langage naturel
python tools/query.py "Quels sont les avantages de llama.cpp ?"

# Exporter des signets Brave vers raw/
python tools/bookmarks_to_raw.py --folders IA "GPT local"
python tools/bookmarks_to_raw.py --list-folders   # lister les dossiers disponibles
```

---

## Ingestion

### Mode fichier unique

```bash
python tools/ingest.py https://example.com/article
python tools/ingest.py /chemin/vers/document.pdf
```

### Mode batch (`--raw`)

Traite tous les fichiers nouveaux dans `raw/` (manifeste `.ingest_manifest` pour les ignorer au prochain lancement) :

```bash
python tools/ingest.py --raw
python tools/ingest.py --raw --force   # repart de zéro (recrée l'index)
```

Les fichiers `.url` dans `raw/` contiennent une URL par fichier.  
Les fichiers PDF/texte sont copiés dans `raw/` à l'ingestion.

### Déduplication

- Fichiers binaires (PDF…) : clé SHA-256 — pas de double copie même si le nom diffère.
- Fichiers `.url` : URL normalisée (paramètres de tracking supprimés, fragment ignoré).

---

## Wikipedia lookup (`wiki_lookup.py`)

Les pages de concepts et d'entités sont enrichies avec un extrait Wikipedia pour ancrer le LLM :

1. **ZIM local** (Kiwix) — lecture hors-ligne si un fichier `.zim` est présent dans `zim/`
2. **Cache SQLite** (`wiki_cache.db`) — évite les appels réseau répétés
3. **API REST Wikipedia** — fallback en ligne (FR puis EN)

```bash
# Préchauffer le cache sur les pages existantes
python tools/build_wiki_cache.py
```

Placer un fichier ZIM dans `zim/` (ex. `wikipedia_fr_all_mini_2026-02.zim`) pour activer le mode hors-ligne.

---

## Export de signets (`bookmarks_to_raw.py`)

```bash
# Aperçu sans écriture
python tools/bookmarks_to_raw.py --dry-run --folders IA

# Export effectif
python tools/bookmarks_to_raw.py --folders IA Ordinateur "GPT local"

# Ingestion en arrière-plan
nohup python -u tools/ingest.py --raw > /tmp/ingest.log 2>&1 &
tail -f /tmp/ingest.log
```

Le script déduplique automatiquement les URLs déjà présentes dans `raw/`.

---

## Configuration LLM

Le LLM utilisé est sélectionné par variable d'environnement (dans `.env` ou l'environnement shell) :

| Variable | Valeur | Backend |
|----------|--------|---------|
| `WIKI_LLM_BACKEND` | `openai` | DeepSeek API (`DEEPSEEK_API_KEY`, `OPENAI_MODEL=deepseek-v4-flash`) |
| `WIKI_LLM_BACKEND` | `claude` | Anthropic API (`ANTHROPIC_API_KEY`) |
| `WIKI_LLM_BACKEND` | `ollama` | Ollama local (`OLLAMA_MODEL`) |

---

## Tests

```bash
cd ~/Secretarius/Wiki_LM
source .venv/bin/activate
python -m pytest tests/ -v
```

170 tests couvrant : ingestion, déduplication, index, normalisation, slugification, Wikipedia lookup, embeddings, clustering (algorithme des transferts), base de connaissance (kb_update, kb_query, kb_tags).  
Tous les tests sont isolés (pas de réseau, pas de LLM réel — `MockLLM` + `tmp_path`).

---

## Roadmap

### Détection de contradictions (`audit_contradictions.py`)

Non implémenté. Le patron Karpathy mentionne l'audit de contradictions entre pages comme opération périodique, mais c'est la fonctionnalité la plus critique du patron — en particulier dans un contexte multi-sources où des points de vue divergents peuvent être silencieusement fusionnés par le LLM lors de l'ingestion.

Approche envisagée : pour chaque ingestion (ou en mode audit périodique), rechercher via BGE-M3 les pages existantes sémantiquement proches de la source entrante, soumettre les paires au LLM avec la question "ces affirmations se contredisent-elles ?", et écrire les contradictions détectées dans une page dédiée `wiki/contradictions.md` plutôt que de les résoudre automatiquement.

### Recherche locale dans Obsidian

Non implémenté. Objectif : une recherche hybride BM25+sémantique sur les pages wiki, sans serveur, directement depuis Obsidian.

Deux approches envisagées :
- **Plugin communautaire Omnisearch** : BM25 full-text local, rien à développer, mais pas d'accès aux embeddings BGE-M3 existants.
- **Page HTML statique dans le coffre** : un `search.html` chargé dans le navigateur, alimenté par un index JSON exporté par un nouvel outil `export_search_index.py`. Les embeddings BGE-M3 pré-calculés (tableaux de floats) peuvent être inclus pour la similarité cosinus côté client via FlexSearch ou Lunr.js. Résultats sous forme de liens `obsidian://`. Entièrement hors ligne.
