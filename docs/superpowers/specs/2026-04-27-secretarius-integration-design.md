# Secretarius — Design d'intégration

Date : 2026-04-27
Statut : Approuvé

## Objectif

Intégrer Wiki_LM (code), OpenClaw (config) et la documentation projet dans un dépôt unique reproductible avec script d'installation, définissant le chemin du coffre Obsidian, le nom de l'assistant (Tiron), et le LLM à utiliser (DeepSeek).

## Structure du dépôt Secretarius

```
Secretarius/
├── install.sh                 # Script d'installation principal
├── install.conf               # Configuration par défaut (sourceable)
├── CLAUDE.md                  # Instructions pour Claude Code
├── README.md                  # Documentation du projet
│
├── Wiki_LM/                   # Coeur du projet (inchangé)
│   ├── tools/                 # Pipeline CLI (ingest, query, search, lint…)
│   ├── tests/                 # 74 tests pytest
│   ├── .env.template          # Template de config LLM
│   └── requirements.txt
│
├── openclaw-config/           # Template pour ~/.openclaw/
│   ├── openclaw.json.template # Config OpenClaw sans secrets
│   ├── gateway.systemd.env.template # Secrets (bot token, gateway token/password)
│   ├── openclaw-gateway.service     # Unité systemd user
│   └── install.sh             # Sous-script : génère ~/.openclaw/ + active service
│
├── docs/
│   ├── architecture/          # Briefings, patterns, décisions d'architecture
│   │   └── llm-wiki-pattern.md
│   ├── history/               # HistoriqueSecretarius.md
│   └── superpowers/specs/     # Specs issues du processus brainstorming
│
└── scripts/                   # Utilitaires (optionnel)
```

## Script d'installation

### Usage

```
./install.sh [options]

Options:
  --obsidian-path PATH    Chemin du coffre Obsidian (défaut: ~/Documents/Obsidian)
  --assistant-name NAME   Nom de l'assistant OpenClaw (défaut: Tiron)
  --llm BACKEND           LLM par défaut: deepseek | ollama | local (défaut: deepseek)
  --openclaw-path PATH    Où installer la config OpenClaw (défaut: ~/.openclaw)
  --env-file FILE         Fichier contenant les secrets (API keys, tokens)
  --interactive           Mode interactif: pose les questions une par une
  --help                  Affiche l'aide
```

### Étapes

1. **Vérification des prérequis** : Python 3.11+, OpenClaw >= version minimale, git
2. **Validation du coffre Obsidian** : le chemin existe
3. **Génération de `~/.openclaw/`** :
   - `openclaw.json` depuis le template (expansion des variables : OBSIDIAN_PATH, ASSISTANT_NAME, LLM_BACKEND)
   - `gateway.systemd.env` depuis le template (secrets depuis l'environnement ou fichier)
   - Copie/installation de `~/.config/systemd/user/openclaw-gateway.service`
   - Activation : `systemctl --user daemon-reload && systemctl --user enable --now openclaw-gateway.service`
4. **Configuration de `Wiki_LM/.env`** : pointage LLM, chemin du wiki
5. **Création des répertoires de données** : `raw/`, `wiki/` (hors dépôt)
6. **Installation des dépendances Python** : `pip install -r Wiki_LM/requirements.txt`

### Propriétés

- Idempotent : relançable sans casse
- Non-interactif par défaut, compatible CI
- Les valeurs par défaut viennent de `install.conf`
- Mode `--interactive` pour usage humain

## openclaw-config/

La config OpenClaw n'est pas versionnée directement (contiendrait des secrets). On versionne un template.

### openclaw.json.template

- Tous les champs de la config OpenClaw sont présents
- Les valeurs qui varient par machine sont des placeholders : `${OBSIDIAN_PATH}`, `${ASSISTANT_NAME}`, `${LLM_BACKEND}`
- Les clés API sont des chaînes vides ou commentées
- La section `agents.main` est configurée avec le LLM choisi

### gateway.systemd.env.template

Contient les variables d'environnement nécessaires à la gateway :
- `TELEGRAM_BOT_TOKEN=`
- `GATEWAY_TOKEN=`
- `GATEWAY_PASSWORD=`

Les valeurs sont vides dans le template, renseignées par l'utilisateur.

### openclaw-gateway.service

Unité systemd user qui démarre la gateway OpenClaw au login.

## Compatibilité OpenClaw

- Version minimale requise documentée dans README.md
- `install.sh` vérifie `openclaw --version` >= X.Y avant de générer la config
- Si incompatibilité : message d'erreur explicite avec la version requise
- Approche standard de type dotfiles repo (vérification de version, pas d'adaptation automatique)

## Sécurité

- `gateway.systemd.env` est créé avec les permissions 600
- `.gitignore` exclut `gateway.systemd.env`, `.env`, les répertoires `raw/`, `wiki/`
- Aucun secret dans les templates versionnés
- Les placeholders dans le template sont en forme `${VAR}` — pas de vraies valeurs

## Déplacements depuis l'existant

- `~/Secretarius_dev/HistoriqueSecretarius.md` → `Secretarius/docs/history/`
- `~/Secretarius_dev/briefing_claude_code_llm_wiki.md` → `Secretarius/docs/architecture/`
- `~/Secretarius_dev/Nouveau_départ.md` → `Secretarius/docs/history/`
- `~/Secretarius_dev/objectif_second_cerveau.md` → `Secretarius/docs/architecture/`
- `~/Secretarius_dev/LLM_Wiki.md` → `Secretarius/docs/architecture/`
