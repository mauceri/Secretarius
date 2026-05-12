# Secretarius — Design d'intégration (v2)

Date : 2026-05-12
Statut : Approuvé
Remplace : `2026-04-27-secretarius-integration-design.md`

## Objectif

Construire un dépôt GitHub partageable qui intègre les outils Wiki_LM et la configuration
OpenClaw, avec un script d'installation permettant à un nouvel utilisateur de configurer
son propre instance en définissant : chemin du coffre Obsidian, nom de l'assistant, LLM
à utiliser.

## Périmètre

- **Inclus** : Wiki_LM/tools, Wiki_LM/tests, config OpenClaw (templates), PATTERN.md, docs
- **Exclu** : Prototype/ (conservé uniquement dans la branche `Expressions`), modèle LoRA
  local, données wiki (raw/, wiki actif, knowledge_base/)

## Structure du dépôt

```
Secretarius/
├── install.sh                  # Script principal (idempotent, bash + envsubst)
├── install.conf                # Valeurs par défaut (sourceable)
├── PATTERN.md                  # Le patron LLM Wiki (document de référence)
├── README.md
├── CLAUDE.md
│
├── Wiki_LM/                    # Outils pipeline (inchangé)
│   ├── tools/
│   ├── tests/
│   ├── .env.template
│   └── requirements.txt
│
├── openclaw-config/
│   ├── openclaw.json.template  # Config complète, variables ${HOME} etc.
│   ├── gateway.systemd.env.template
│   ├── openclaw-gateway.service
│   └── install.sh              # Sous-script : génère ~/.openclaw/
│
└── docs/
    ├── history/                # point-11-05-2026.md, HistoriqueSecretarius.md, etc.
    ├── architecture/           # Briefings, décisions d'architecture
    └── superpowers/specs/      # Specs issues des sessions brainstorming
```

Fichiers exclus du dépôt via `.gitignore` : `raw/`, wiki actif, `knowledge_base/`,
`.env`, `gateway.systemd.env`.

## Script d'installation

### Usage

```
./install.sh [options]

Options :
  --obsidian-path PATH    Chemin du coffre Obsidian (défaut: ~/Documents/Obsidian)
  --assistant-name NAME   Nom de l'assistant OpenClaw (défaut: Tiron)
  --llm BACKEND           LLM : deepseek | ollama | local (défaut: deepseek)
  --openclaw-path PATH    Où installer la config OpenClaw (défaut: ~/.openclaw)
  --env-file FILE         Fichier contenant les secrets (tokens, API keys)
  --interactive           Mode interactif
  --force                 Écrase les fichiers déjà présents
  --help
```

### Étapes

1. **Prérequis** : Python 3.11+, git, `envsubst` (paquet `gettext`, standard Linux)
2. **OpenClaw** : vérifie présence dans le PATH, avertit sinon (ne bloque pas)
3. **Config OpenClaw** : `openclaw-config/install.sh` substitue les variables dans
   `openclaw.json.template` → `~/.openclaw/openclaw.json` ; génère
   `gateway.systemd.env` (chmod 600) ; copie le service systemd user
4. **Wiki_LM/.env** : génère depuis `.env.template` si absent (`WIKI_PATH` →
   `${OBSIDIAN_PATH}/Wiki_LM`)
5. **Dépendances Python** : `pip install -r Wiki_LM/requirements.txt`
6. **Résumé** : affiche les étapes manuelles restantes

### Propriétés

- Idempotent : fichiers existants non écrasés sans `--force`
- Non-interactif par défaut, compatible CI
- Valeurs par défaut dans `install.conf`

## openclaw.json.template — stratégie de substitution

Source : `~/.openclaw/openclaw.json` live.

| Valeur actuelle | Placeholder |
|---|---|
| `/home/mauceric` (chemins absolus) | `${HOME}` |
| `sanroque` (hostname dans les URLs) | `${HOSTNAME}` |
| Nom de l'assistant (`Tiron`) | `${ASSISTANT_NAME}` |
| LLM par défaut dans les agents | `${LLM_BACKEND}` |
| Sections modèle LoRA (`llamacpp`) | Supprimées |

Tout le reste (flows, agents, canvas, extensions, credentials vides) est fourni tel
quel comme point de départ.

`gateway.systemd.env.template` : variables vides uniquement.
```
TELEGRAM_BOT_TOKEN=
GATEWAY_TOKEN=
GATEWAY_PASSWORD=
```

## Migration depuis l'état actuel

Les fichiers suivants sont des ébauches à remplacer :

| Fichier | Action |
|---|---|
| `install.sh` | Réécrit intégralement |
| `install.conf` | Conservé, ajusté si nécessaire |
| `openclaw-config/install.sh` | Réécrit |
| `openclaw-config/openclaw.json.template` | Régénéré depuis `~/.openclaw/openclaw.json` |
| `openclaw-config/gateway.systemd.env.template` | Conservé |
| `openclaw-config/openclaw-gateway.service` | Conservé |
| `Wiki_LM/.env.template` | Créé s'il n'existe pas |
| `point-11-05-2026.md` (racine) | Déplacé dans `docs/history/` |
| Docs `Secretarius_dev/` pertinentes | Déplacées dans `docs/` |

## Sécurité

- `gateway.systemd.env` créé avec permissions 600
- `.gitignore` exclut `.env`, `gateway.systemd.env`, `raw/`, wiki actif, `knowledge_base/`
- Aucun secret ni chemin machine dans les fichiers versionnés
