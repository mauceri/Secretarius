# Secretarius

Knowledge base personnelle locale basée sur le patron *LLM Wiki* (Andrej Karpathy) : un LLM ingère des sources (URLs, PDFs, textes) et maintient de façon incrémentale un wiki Markdown interconnecté — résumés, pages de concepts, pages d'entités, clustering thématique, base de connaissance compactée.

→ Voir [`PATTERN.md`](PATTERN.md) pour la description complète du patron.

## Prérequis

- Python 3.11+
- Node.js 22+ et npm (`nvm install 22` recommandé)
- [OpenClaw](https://openclaw.dev) : `npm install -g openclaw`
- `envsubst` (`apt install gettext` / `brew install gettext`)
- Git

## Installation

> **Important** : ne jamais lancer `openclaw` manuellement avant la fin de l'installation — OpenClaw initialiserait le workspace avec ses fichiers par défaut (en anglais) et écraserait la configuration française.

### 1. Cloner

```bash
git clone https://github.com/mauceri/Secretarius
cd Secretarius
```

### 2. Installation

```bash
./install.sh
```

Le script pose les questions interactivement :
- Chemin du coffre Obsidian (ex. `~/Documents/Obsidian`)
- Nom de l'assistant (défaut : `Tiron`)
- Backend LLM (`deepseek` | `ollama` | `claude`)

Puis éditer `~/.openclaw/gateway.systemd.env` pour renseigner `TELEGRAM_BOT_TOKEN` et `DEEPSEEK_API_KEY` (`OPENCLAW_GATEWAY_TOKEN` est généré automatiquement). Les secrets sont lus par OpenClaw au démarrage du service — pas besoin de relancer l'installation.

Options disponibles :

```
--obsidian-path PATH    Chemin du coffre Obsidian
--assistant-name NAME   Nom de l'assistant
--llm BACKEND           deepseek | ollama | claude (défaut: deepseek)
--env-file FILE         Fichier de secrets (clés API, tokens)
--force                 Écrase les fichiers existants
```

### 3. Démarrer le service

```bash
systemctl --user daemon-reload
systemctl --user enable --now openclaw-gateway.service
```

### 4. Appairer Telegram

Envoyer `/start` au bot Telegram, puis :

```bash
openclaw pairing approve telegram <CODE>
```

## Composants

### Wiki_LM

Pipeline complet de knowledge base personnelle.

→ Voir [`Wiki_LM/README.md`](Wiki_LM/README.md)

**Fonctionnalités :**
- Ingestion batch depuis `raw/` avec déduplication (SHA-256 / URL normalisée)
- Enrichissement Wikipedia anti-hallucination (ZIM Kiwix → cache SQLite → API REST)
- Export des signets Brave vers `raw/`
- Recherche BM25 + requêtes en langage naturel (hybride BM25 + BGE-M3)
- Clustering thématique (algorithme des transferts)
- Base de connaissance compactée (axes thématiques, centroïdes BGE-M3)
- Suite de tests pytest isolée (170 tests, zéro réseau)

### OpenClaw

Agent conversationnel configuré pour opérer sur le wiki via les outils Wiki_LM.  
La configuration est générée depuis `openclaw-config/openclaw.json.template` à l'installation.

## Structure du dépôt

```
Secretarius/
├── install.sh                 # Script d'installation principal
├── install.conf               # Valeurs par défaut (sourceable)
├── PATTERN.md                 # Le patron LLM Wiki
├── CLAUDE.md                  # Instructions pour Claude Code / agents
├── README.md
│
├── Wiki_LM/                   # Outils pipeline LLM Wiki
│   ├── tools/                 # 24 outils CLI (ingest, query, cluster, kb_*)
│   ├── tests/                 # Suite pytest (170 tests)
│   ├── .env.template          # Template de configuration LLM
│   └── requirements.txt
│
├── openclaw-config/           # Templates de configuration OpenClaw
│   ├── openclaw.json.template # Config complète (variables ${HOME}, ${HOSTNAME}…)
│   ├── gateway.systemd.env.template # Secrets (tokens — non versionné)
│   ├── openclaw-gateway.service     # Unité systemd user
│   └── install.sh             # Génère ~/.openclaw/ via envsubst
│
└── docs/
    ├── architecture/          # Décisions d'architecture
    ├── history/               # Historique du projet
    └── superpowers/           # Specs et plans d'implémentation
        ├── specs/
        └── plans/
```

Les données wiki (`raw/`, `wiki/`, `embeddings/`, `knowledge_base/`) vivent sous le coffre Obsidian (`OBSIDIAN_PATH/Wiki_LM/`) et ne sont pas versionnées.
