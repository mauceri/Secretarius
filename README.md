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

### 3. Appairer Telegram

Si `TELEGRAM_BOT_TOKEN` est déjà renseigné dans `gateway.systemd.env`, les services démarrent automatiquement à l'étape 2. Sinon, éditer le fichier puis démarrer :

```bash
systemctl --user start openclaw-gateway.service openclaw-scout.service
```

Envoyer `/start` au bot Telegram, puis :

```bash
openclaw pairing approve telegram <CODE>
```

(La connexion Telegram est interrompue lors du pairing — relancer `openclaw gateway restart` pour la rétablir.)

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

Deux agents sont configurés :

- **main** (`Tiron`) — agent principal, accessible via Telegram.
- **scout** — agent isolé, sans accès réseau direct. Utilisé par Tiron pour lire des sources externes (URL, pages web) à travers `sessions_spawn`. Le contenu est pré-fetché par `scout-watcher` (curl côté hôte) et transmis via des fichiers JSON dans `~/.openclaw/agents/scout/workspace/`. Scout n'exécute jamais de commandes réseau. `scout-watcher` est lancé par `openclaw-scout.service` (service systemd user, démarre après `openclaw-gateway.service`).

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
│   ├── openclaw-gateway.service     # Unité systemd user (agent principal)
│   ├── openclaw-scout.service       # Unité systemd user (watcher scout)
│   ├── scout-watcher          # Script bash : poll tasks/pending/, pré-fetch URL, signal scout
│   ├── install.sh             # Génère ~/.openclaw/ via envsubst
│   ├── workspace/             # Workspace de l'agent principal (SOUL.md, AGENTS.md, skills/)
│   └── agents/scout/workspace/ # Workspace isolé de l'agent scout
│
└── docs/
    ├── architecture/          # Décisions d'architecture
    ├── history/               # Historique du projet
    └── superpowers/           # Specs et plans d'implémentation
        ├── specs/
        └── plans/
```

Les données wiki (`raw/`, `wiki/`, `embeddings/`, `knowledge_base/`) vivent dans le coffre Obsidian (`OBSIDIAN_PATH/Wiki_LM/`) et ne sont pas versionnées.

## Roadmap

### Synchronisation du workspace Tiron vers le dépôt

Tension structurelle actuelle : le dépôt versionne les *templates* de workspace (`openclaw-config/workspace/`), mais le workspace réel sur santiago (`~/.openclaw/workspace/`) peut diverger au fil des sessions — Tiron est censé modifier ses propres fichiers (`SOUL.md`, `AGENTS.md`) comme mécanisme de mémoire. Un `./install.sh --force` écrase ces évolutions.

Solution envisagée : donner à Tiron accès à `git` (safeBins), configurer des credentials en écriture sur santiago, et lui instruire de committer ses modifications de workspace sur une branche dédiée (`tiron/workspace`). La branche est mergée manuellement par l'utilisateur. Estimation : demi-session.

### Défense structurelle contre l'injection de prompt indirecte

Le skill `prompt-injection-guard` est déclaratif (instructions comportementales au LLM) et ne peut pas être déclenché de façon fiable sur les résultats de `sessions_spawn`. Une injection placée dans le contenu web fetché par Scout se retrouve dans le JSON retourné à Tiron — les balises `<UNTRUSTED>` et les instructions SOUL.md atténuent le risque mais ne l'éliminent pas.

Piste à approfondir : existe-t-il dans l'écosystème OpenClaw ou LLM un mécanisme structurel (non comportemental) permettant de filtrer le contenu non fiable avant qu'il n'atteigne le contexte de l'agent principal ? Les approches déterministes (regex dans scout-watcher) couvrent les cas évidents mais fragilisent la chaîne et ne résistent pas aux injections sophistiquées.
