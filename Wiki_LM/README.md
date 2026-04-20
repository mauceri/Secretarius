# Wiki_LM

Implémentation du patron *LLM Wiki* (Andrej Karpathy) : un wiki Markdown personnel maintenu de façon incrémentale par un LLM. Chaque source ingérée est résumée, ses concepts et entités identifiés et enrichis, et le tout est interconnecté via des liens internes.

Voir `PATTERN.md` pour la description complète du patron et de ses cas d'usage.

---

## Architecture

```
Wiki_LM/
├── tools/          ← pipeline d'ingestion et outils CLI
├── tests/          ← suite pytest (74 tests)
├── wiki/           ← pages Markdown (données, hors dépôt)
├── raw/            ← sources brutes (données, hors dépôt)
├── zim/            ← fichiers Kiwix ZIM optionnels (hors dépôt)
├── PATTERN.md      ← description du patron
└── requirements.txt
```

Les données (`wiki/`, `raw/`, `zim/`) sont hors dépôt (`.gitignore`).  
Le répertoire wiki par défaut est `~/Secretarius/Wiki_LM/wiki/` ;  
le répertoire raw par défaut est `~/Secretarius/Wiki_LM/raw/`.

---

## Outils (`tools/`)

| Outil | Description |
|-------|-------------|
| `ingest.py` | Ingestion d'une source (URL, PDF, texte) dans le wiki |
| `capture.py` | Capture rapide depuis OpenClaw / CLI |
| `query.py` | Interrogation du wiki en langage naturel |
| `search.py` | Recherche BM25 sur les pages |
| `lint.py` | Health-check : liens brisés, pages orphelines |
| `wiki_lookup.py` | Lookup Wikipedia (ZIM → cache SQLite → API REST) |
| `bookmarks_to_raw.py` | Export des signets Brave vers `raw/` |
| `build_wiki_cache.py` | Préchauffage du cache Wikipedia sur les pages existantes |
| `summarize.py` | Résumé d'un document via LLM |
| `build_summary_corpus.py` | Construction d'un corpus de résumés |
| `llm.py` | Abstraction LLM (DeepSeek / Anthropic / Ollama) |

---

## Démarrage rapide

```bash
cd ~/Secretarius/Wiki_LM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

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
| `LLM_BACKEND` | `deepseek` | DeepSeek API (`DEEPSEEK_API_KEY`) |
| `LLM_BACKEND` | `anthropic` | Anthropic API (`ANTHROPIC_API_KEY`) |
| `LLM_BACKEND` | `ollama` | Ollama local (`OLLAMA_MODEL`) |

---

## Tests

```bash
cd ~/Secretarius/Wiki_LM
source .venv/bin/activate
python -m pytest tests/ -v
```

74 tests couvrant : ingestion, déduplication, index, normalisation, slugification, Wikipedia lookup.  
Tous les tests sont isolés (pas de réseau, pas de LLM réel — `MockLLM` + `tmp_path`).
