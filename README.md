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
├── install.sh                 # Script d'installation principal
├── install.conf               # Configuration par défaut (sourceable)
├── CLAUDE.md                  # Instructions pour Claude Code / agents
├── README.md                  # Ce fichier
│
├── Wiki_LM/                   # Coeur du projet LLM Wiki
│   ├── tools/                 # Pipeline CLI (ingest, query, search…)
│   ├── tests/                 # Suite pytest (74 tests)
│   ├── .env.template          # Template de configuration LLM
│   ├── PATTERN.md             # Description du pattern LLM Wiki
│   └── requirements.txt
│
├── openclaw-config/           # Templates pour OpenClaw
│   ├── openclaw.json.template # Config OpenClaw (placeholders)
│   ├── gateway.systemd.env.template # Secrets (généré à l'install)
│   ├── openclaw-gateway.service     # Unité systemd user
│   └── install.sh             # Sous-script de génération
│
├── docs/                      # Documentation
│   ├── architecture/          # Décisions d'architecture, patterns
│   ├── history/               # Historique du projet
│   └── superpowers/           # Specs et plans d'implémentation
│       ├── specs/
│       └── plans/
│
├── data/                      # Données runtime (hors git)
│   ├── raw/                   # Sources brutes ingérées
│   └── wiki/                  # Wiki généré
│
└── worktrees/                 # Git worktrees (hors git)
```
