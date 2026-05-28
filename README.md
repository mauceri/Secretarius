# Secretarius

**Secretarius est une configuration sécurisée d'[OpenClaw](https://openclaw.dev) pour individus et petites structures**, enrichie d'une base de connaissances personnelle.

OpenClaw est un agent IA puissant, mais dangereux dans sa configuration standard : il donne à l'IA un accès large à la machine hôte et reste vulnérable aux injections de prompt. Secretarius corrige cela par une architecture de sécurité en plusieurs couches, tout en ajoutant une mémoire documentaire durable.

Trois principes fondateurs :
- **Indépendance** : hébergé sur votre propre machine ou serveur, sans dépendance aux fournisseurs cloud d'IA.
- **Frugalité** : consommation comparable à un ordinateur de jeu.
- **Confidentialité** : vos données restent sous votre contrôle.

→ `docs/Secretarius.md` — présentation complète du projet

---

## Architecture

Secretarius repose sur [OpenClaw](https://openclaw.dev), enrichi d'un gestionnaire de base de connaissances inspiré du patron *LLM Wiki* (Andrej Karpathy).

Trois composants principaux :

**Tiron** — l'agent principal, accessible via Telegram. Il mémorise, recherche, synthétise et agit selon les instructions de l'utilisateur. Il s'appuie sur un système de *skills* (compétences réutilisables décrites en langage naturel) et sur le wiki personnel.

**Scout** — agent isolé, sans accès réseau direct. Toute lecture de source externe (URL, page web) passe par Scout, qui filtre le contenu avant de le transmettre à Tiron. Cela réduit fortement le risque d'injection de prompt indirecte.

**Wiki_LM** — pipeline de base de connaissances personnelle. Il ingère des sources (URLs, PDFs, notes), génère des pages wiki Markdown structurées (résumés, concepts, entités), et expose 7 outils MCP à Tiron : `wiki_capture`, `wiki_ingest`, `wiki_ingest_status`, `wiki_list_pending`, `wiki_query`, `wiki_tags`, `wiki_kb_update`.

---

## Prérequis

- Python 3.11+
- Node.js 22+ via NVM (`nvm install 22` recommandé)
- OpenClaw : `npm install -g openclaw`
- `envsubst` : `apt install gettext` / `brew install gettext`
- Git
- Un bot Telegram (token obtenu via [@BotFather](https://t.me/botfather))
- Une clé API DeepSeek (ou Ollama en local)

---

## Installation

> **Important** : ne pas lancer `openclaw` manuellement avant la fin de l'installation — OpenClaw initialiserait le workspace avec ses fichiers par défaut (en anglais) et écraserait la configuration française.

### 1. Cloner

```bash
git clone https://github.com/mauceri/Secretarius
cd Secretarius
```

### 2. Installer

```bash
./install.sh
```

Le script pose trois questions interactivement :
- Chemin du coffre Obsidian (ex. `~/Documents/Obsidian`)
- Nom de l'assistant (défaut : `Tiron`)
- Backend LLM (`deepseek` | `ollama` | `claude`)

### 3. Renseigner les secrets

```bash
nano ~/.openclaw/gateway.systemd.env
```

Compléter :
```
TELEGRAM_BOT_TOKEN=<token BotFather>
DEEPSEEK_API_KEY=<clé API DeepSeek>
```
(`OPENCLAW_GATEWAY_TOKEN` est généré automatiquement.)

### 4. Démarrer

```bash
./start.sh
```

Le script démarre les services, charge le plugin MCP et attend la confirmation que les 7 outils wiki sont enregistrés.

### 5. Appairer Telegram

Envoyer `/start` au bot Telegram, puis :

```bash
openclaw pairing approve telegram <CODE>
systemctl --user restart openclaw-gateway.service
```

---

## Mise à jour

```bash
git pull && ./install.sh --force && ./start.sh
```

---

## Structure du dépôt

```
Secretarius/
├── install.sh                      # Installation idempotente
├── start.sh                        # Démarrage des services au quotidien
├── uninstall_openclaw.sh           # Désinstallation
├── install.conf                    # Valeurs par défaut
├── PATTERN.md                      # Patron LLM Wiki (Karpathy)
│
├── Wiki_LM/
│   ├── tools/                      # Pipeline : ingest, query, cluster, kb_*
│   ├── tests/                      # Suite pytest (170+ tests, zéro réseau)
│   ├── .env.template               # Configuration LLM
│   └── requirements.txt
│
├── openclaw-config/
│   ├── openclaw.json.template      # Configuration OpenClaw complète
│   ├── workspace/                  # Workspace Tiron (SOUL.md, AGENTS.md, skills/)
│   ├── agents/scout/workspace/     # Workspace Scout (isolé)
│   ├── injection_guard.py          # Filtre anti-injection de prompt
│   ├── scout_process.py            # Logique de fetch sécurisé
│   └── install.sh                  # Génère ~/.openclaw/ via envsubst
│
└── docs/
    ├── Secretarius.md              # Présentation complète du projet
    └── history/                    # Historique des sessions de développement
```

Les données wiki (`raw/`, `wiki/`, `knowledge_base/`) vivent dans le coffre Obsidian (`OBSIDIAN_PATH/Wiki_LM/`) et ne sont pas versionnées.

---

## Sécurité

OpenClaw présente des risques de sécurité importants dans sa configuration standard. Secretarius les atténue par :

- **Sandbox strict** : Tiron n'a accès qu'à son workspace, pas à la machine hôte.
- **Scout** : tout contenu externe passe par un agent isolé avant d'atteindre Tiron.
- **Injection guard** : filtre heuristique sur le contenu fetché.
- **Allowlist réseau** : les communications externes sont limitées aux destinations pré-approuvées.

Ces mesures réduisent fortement la surface d'attaque sans l'éliminer complètement. Voir `docs/Secretarius.md` pour une analyse détaillée.
