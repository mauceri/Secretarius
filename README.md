# Secretarius

Secrétaire documentaire personnel, local et frugal.

## Composants

### Wiki_LM

Pipeline de knowledge base personnelle basé sur le patron *LLM Wiki* (Andrej Karpathy).  
Un LLM ingère des sources (URLs, PDFs, textes) et maintient de façon incrémentale un wiki Markdown interconnecté — résumés, pages de concepts, pages d'entités, liens croisés.

→ Voir [`Wiki_LM/README.md`](Wiki_LM/README.md) et [`Wiki_LM/PATTERN.md`](Wiki_LM/PATTERN.md)

**Fonctionnalités principales :**
- Ingestion batch depuis `raw/` avec déduplication (SHA-256 / URL normalisée)
- Enrichissement Wikipedia anti-hallucination (ZIM Kiwix → cache SQLite → API REST)
- Export des signets Brave vers `raw/`
- Recherche BM25 + requêtes en langage naturel
- Suite de tests pytest isolée (74 tests, zéro réseau)

### Infrastructure LLM locale

| Service | Port | Modèle | Rôle |
|---------|------|--------|------|
| llama.cpp | 8989 | Phi-4-mini LoRA (Wikipedia FR) | Extraction d'expressions |
| Ollama | 11434 | Qwen | LLM généraliste |

Les artefacts LoRA sont dans `~/lora_local/` (hors dépôt).

## Démarrage rapide

```bash
cd ~/Secretarius/Wiki_LM
python -m venv .venv && source .venv/bin/activate
pip install -r Wiki_LM/requirements.txt

# Ingérer une URL
python Wiki_LM/tools/ingest.py https://example.com/article

# Lancer les tests
python -m pytest Wiki_LM/tests/
```

## Structure du dépôt

```
Secretarius/
├── Wiki_LM/
│   ├── tools/          ← pipeline (ingest, capture, search, query…)
│   ├── tests/          ← suite pytest
│   ├── PATTERN.md      ← description du patron LLM Wiki
│   └── requirements.txt
├── CLAUDE.md           ← instructions pour Claude Code
└── README.md
```
