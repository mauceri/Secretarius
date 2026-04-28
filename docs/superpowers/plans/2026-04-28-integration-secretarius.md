# Intégration Secretarius — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Structurer le dépôt Secretarius avec script d'installation, templates OpenClaw, documentation organisée et .gitignore complet.

**Architecture:** Script `install.sh` idempotent qui lit `install.conf`, génère `~/.openclaw/` depuis des templates, configure `Wiki_LM/.env`, et installe les dépendances. Les templates OpenClaw contiennent des placeholders `${VAR}` pour les valeurs machine-dépendantes et des champs vides pour les secrets.

**Tech Stack:** bash, python, systemd (user units), templates de configuration

---

## Fichiers à créer ou modifier

| Fichier | Action | Responsable |
|---------|--------|-------------|
| `install.sh` | Créer | Script principal d'installation |
| `install.conf` | Déjà existant | Configuration par défaut (sourceable) |
| `openclaw-config/openclaw.json.template` | Créer | Template config OpenClaw |
| `openclaw-config/gateway.systemd.env.template` | Créer | Template secrets systemd |
| `openclaw-config/openclaw-gateway.service` | Créer | Unité systemd user |
| `openclaw-config/install.sh` | Créer | Sous-script génération ~/.openclaw/ |
| `.gitignore` | Modifier | Exclure secrets, worktrees, données runtime |
| `docs/architecture/llm-wiki-pattern.md` | Créer | Pattern LLM Wiki (depuis docs existantes) |
| `docs/history/` | Créer + peupler | Historique et documents de briefing |
| `Wiki_LM/.env.template` | Créer (renommer .env.example) | Template de config Wiki_LM |
| `README.md` | Modifier | Mettre à jour la structure du dépôt |

---

### Task 1: .gitignore complet

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Mettre à jour `.gitignore`**

```gitignore
# Environnements Python
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# Données runtime Wiki_LM
Wiki_LM/wiki/
Wiki_LM/raw/
Wiki_LM/zim/
Wiki_LM/wiki_cache.db
Wiki_LM/bookmarks.json
Wiki_LM/corpus-pkm.txt

# Secrets / config locale
.env
*.key
gateway.systemd.env

# Worktrees
worktrees/

# OpenClaw généré
openclaw-config/gateway.systemd.env

# Divers
*.log
.DS_Store
```

- [ ] **Step 2: Vérifier que le fichier est correct**

```bash
cat ~/Secretarius/worktrees/intégration/.gitignore
```

- [ ] **Step 3: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add .gitignore
git commit -m "chore: compléter .gitignore pour secrets et données runtime"
```

---

### Task 2: Renommer .env.example en .env.template

**Files:**
- Rename: `Wiki_LM/.env.example` → `Wiki_LM/.env.template`

- [ ] **Step 1: Renommer le fichier**

```bash
cd ~/Secretarius/worktrees/intégration
mv Wiki_LM/.env.example Wiki_LM/.env.template
```

- [ ] **Step 2: Commit**

```bash
git add Wiki_LM/.env.example Wiki_LM/.env.template
git commit -m "chore: renommer .env.example en .env.template pour cohérence"
```

---

### Task 3: Templates OpenClaw — openclaw.json.template

**Files:**
- Create: `openclaw-config/openclaw.json.template`

- [ ] **Step 1: Créer le répertoire et le template**

```bash
mkdir -p ~/Secretarius/worktrees/intégration/openclaw-config
```

```json
{
  "assistant_name": "${ASSISTANT_NAME}",
  "obsidian_path": "${OBSIDIAN_PATH}",
  "llm": {
    "provider": "${LLM_BACKEND}",
    "model": "",
    "api_key": ""
  },
  "agents": {
    "main": {
      "name": "${ASSISTANT_NAME}",
      "llm": "${LLM_BACKEND}"
    }
  },
  "skills": [],
  "commands": {}
}
```

- [ ] **Step 2: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add openclaw-config/openclaw.json.template
git commit -m "feat: ajouter template openclaw.json avec placeholders"
```

---

### Task 4: Templates OpenClaw — gateway.systemd.env.template

**Files:**
- Create: `openclaw-config/gateway.systemd.env.template`

- [ ] **Step 1: Créer le template de secrets**

```bash
cat > ~/Secretarius/worktrees/intégration/openclaw-config/gateway.systemd.env.template << 'TEMPLATE'
# gateway.systemd.env — Secrets pour la gateway OpenClaw
# Ce fichier est généré par install.sh — NE PAS éditer manuellement
# Permissions: 600 (lecture/écriture root uniquement)

TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
GATEWAY_TOKEN=${GATEWAY_TOKEN}
GATEWAY_PASSWORD=${GATEWAY_PASSWORD}
TEMPLATE
```

- [ ] **Step 2: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add openclaw-config/gateway.systemd.env.template
git commit -m "feat: ajouter template gateway.systemd.env avec placeholders secrets"
```

---

### Task 5: Templates OpenClaw — openclaw-gateway.service

**Files:**
- Create: `openclaw-config/openclaw-gateway.service`

- [ ] **Step 1: Créer l'unité systemd**

```ini
[Unit]
Description=OpenClaw Gateway Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.openclaw/gateway.systemd.env
ExecStart=/usr/bin/openclaw gateway
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add openclaw-config/openclaw-gateway.service
git commit -m "feat: ajouter unité systemd user pour openclaw-gateway"
```

---

### Task 6: Sous-script openclaw-config/install.sh

**Files:**
- Create: `openclaw-config/install.sh`

- [ ] **Step 1: Créer le sous-script de génération**

```bash
#!/usr/bin/env bash
# openclaw-config/install.sh — Génère ~/.openclaw/ depuis les templates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source la config par défaut
source "${SCRIPT_DIR}/../install.conf"

# Variables d'environnement ou valeurs par défaut
OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"

# Secrets (doivent être définis)
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"

if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
  echo "ERREUR: TELEGRAM_BOT_TOKEN non défini" >&2
  echo "Définissez-le via --env-file ou variable d'environnement" >&2
  exit 1
fi

echo "→ Génération de ${OPENCLAW_PATH}..."

mkdir -p "$OPENCLAW_PATH"

# Générer openclaw.json
envsubst '${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND}' \
  < "${SCRIPT_DIR}/openclaw.json.template" \
  > "${OPENCLAW_PATH}/openclaw.json"

# Générer gateway.systemd.env (permissions 600)
envsubst '${TELEGRAM_BOT_TOKEN} ${GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
  < "${SCRIPT_DIR}/gateway.systemd.env.template" \
  > "${OPENCLAW_PATH}/gateway.systemd.env"
chmod 600 "${OPENCLAW_PATH}/gateway.systemd.env"

# Installer le service systemd user
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
cp "${SCRIPT_DIR}/openclaw-gateway.service" "$SYSTEMD_USER_DIR/"

echo "→ Configuration OpenClaw générée dans ${OPENCLAW_PATH}"
echo "→ Pour activer le service :"
echo "   systemctl --user daemon-reload"
echo "   systemctl --user enable --now openclaw-gateway.service"
```

- [ ] **Step 2: Rendre exécutable**

```bash
chmod +x ~/Secretarius/worktrees/intégration/openclaw-config/install.sh
```

- [ ] **Step 3: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add openclaw-config/install.sh
git commit -m "feat: ajouter sous-script de génération config OpenClaw"
```

---

### Task 7: Script d'installation principal install.sh

**Files:**
- Create: `install.sh`

- [ ] **Step 1: Créer le script principal**

```bash
#!/usr/bin/env bash
# install.sh — Installation idempotente de Secretarius
# Usage: ./install.sh [--obsidian-path PATH] [--assistant-name NAME] [--llm BACKEND] \
#                      [--openclaw-path PATH] [--env-file FILE] [--interactive] [--help]
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Couleurs ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

# ── Defaults depuis install.conf ──
source "${SECRETARIUS_ROOT}/install.conf"

# ── Parse arguments ──
INTERACTIVE=false
ENV_FILE=""

usage() {
  cat << 'EOF'
Usage: ./install.sh [options]

Options:
  --obsidian-path PATH    Chemin du coffre Obsidian (défaut: ~/Documents/Obsidian)
  --assistant-name NAME   Nom de l'assistant OpenClaw (défaut: Tiron)
  --llm BACKEND           LLM par défaut: deepseek | ollama | local (défaut: deepseek)
  --openclaw-path PATH    Où installer la config OpenClaw (défaut: ~/.openclaw)
  --env-file FILE         Fichier contenant les secrets (API keys, tokens)
  --interactive           Mode interactif: pose les questions une par une
  --help                  Affiche l'aide
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --obsidian-path)   OBSIDIAN_PATH="$2"; shift 2 ;;
    --assistant-name)  ASSISTANT_NAME="$2"; shift 2 ;;
    --llm)             LLM_BACKEND="$2"; shift 2 ;;
    --openclaw-path)   OPENCLAW_PATH="$2"; shift 2 ;;
    --env-file)        ENV_FILE="$2"; shift 2 ;;
    --interactive)     INTERACTIVE=true; shift ;;
    --help)            usage ;;
    *)                 error "Option inconnue: $1"; usage ;;
  esac
done

# ── Mode interactif ──
if [[ "$INTERACTIVE" == true ]]; then
  echo "=== Installation Secretarius (mode interactif) ==="
  echo ""
  read -rp "Chemin du coffre Obsidian [${OBSIDIAN_PATH}]: " val
  OBSIDIAN_PATH="${val:-$OBSIDIAN_PATH}"
  read -rp "Nom de l'assistant [${ASSISTANT_NAME}]: " val
  ASSISTANT_NAME="${val:-$ASSISTANT_NAME}"
  read -rp "LLM backend (deepseek|ollama|local) [${LLM_BACKEND}]: " val
  LLM_BACKEND="${val:-$LLM_BACKEND}"
  read -rp "Chemin config OpenClaw [${OPENCLAW_PATH}]: " val
  OPENCLAW_PATH="${val:-$OPENCLAW_PATH}"
  read -rp "Fichier de secrets (optionnel): " val
  ENV_FILE="${val:-$ENV_FILE}"
fi

# ── Charger les secrets depuis --env-file ──
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    error "Fichier de secrets introuvable: $ENV_FILE"
    exit 1
  fi
  source "$ENV_FILE"
fi

# ── Étape 1: Vérification des prérequis ──
info "Vérification des prérequis..."

# Python 3.11+
if ! command -v python3 &>/dev/null; then
  error "Python3 non installé"
  exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
  error "Python 3.11+ requis (trouvé: $PYTHON_VERSION)"
  exit 1
fi
info "Python $PYTHON_VERSION ✓"

# Git
if ! command -v git &>/dev/null; then
  error "Git non installé"
  exit 1
fi
info "Git ✓"

# OpenClaw (version minimale)
if command -v openclaw &>/dev/null; then
  OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "0.0.0")
  info "OpenClaw $OPENCLAW_VERSION détecté"
  # Comparaison simple de versions (YYYY.M.DD)
  if [[ "$OPENCLAW_VERSION" < "$OPENCLAW_MIN_VERSION" ]]; then
    warn "OpenClaw >= $OPENCLAW_MIN_VERSION recommandé (trouvé: $OPENCLAW_VERSION)"
  fi
else
  warn "OpenClaw non installé — la config sera générée mais le service ne démarrera pas"
fi

# ── Étape 2: Validation du coffre Obsidian ──
info "Validation du coffre Obsidian..."
if [[ ! -d "$OBSIDIAN_PATH" ]]; then
  warn "Coffre Obsidian introuvable: $OBSIDIAN_PATH"
  read -rp "Créer le répertoire ? [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy] ]]; then
    mkdir -p "$OBSIDIAN_PATH"
    info "Répertoire créé: $OBSIDIAN_PATH"
  else
    error "Coffre Obsidian requis. Annulation."
    exit 1
  fi
else
  info "Coffre Obsidian trouvé: $OBSIDIAN_PATH ✓"
fi

# ── Étape 3: Génération de ~/.openclaw/ ──
info "Génération de la configuration OpenClaw..."
export OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND OPENCLAW_PATH
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
export GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"

bash "${OPENCLAW_CONFIG_PATH}/install.sh"

# ── Étape 4: Configuration Wiki_LM/.env ──
info "Configuration de Wiki_LM/.env..."
WIKI_ENV_FILE="${WIKI_LM_PATH}/../.env"
if [[ ! -f "$WIKI_ENV_FILE" ]]; then
  cp "${WIKI_LM_PATH}/.env.template" "$WIKI_ENV_FILE"
  # Remplacer WIKI_PATH avec le chemin du wiki dans le vault Obsidian
  WIKI_DATA_PATH="${OBSIDIAN_PATH}/Wiki_LM"
  sed -i "s|^WIKI_PATH=.*|WIKI_PATH=${WIKI_DATA_PATH}|" "$WIKI_ENV_FILE"
  info "Wiki_LM/.env créé depuis le template"
else
  info "Wiki_LM/.env existe déjà — inchangé"
fi

# ── Étape 5: Création des répertoires de données ──
info "Création des répertoires de données..."
DATA_ROOT="${SECRETARIUS_ROOT}/data"
mkdir -p "${DATA_ROOT}/raw" "${DATA_ROOT}/wiki"
info "Répertoires créés: ${DATA_ROOT}/{raw,wiki}"

# ── Étape 6: Installation des dépendances Python ──
info "Installation des dépendances Python..."
if command -v pip3 &>/dev/null; then
  pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet
  info "Dépendances Python installées ✓"
else
  warn "pip3 non trouvé — installez manuellement: pip install -r Wiki_LM/requirements.txt"
fi

# ── Résumé ──
echo ""
info "=== Installation terminée ==="
echo ""
echo "Prochaines étapes :"
echo "  1. Activer le service OpenClaw :"
echo "     systemctl --user daemon-reload"
echo "     systemctl --user enable --now openclaw-gateway.service"
echo ""
echo "  2. Tester Wiki_LM :"
echo "     cd ${WIKI_LM_PATH}"
echo "     python -m pytest tests/"
echo ""
echo "  3. Ingérer une source :"
echo "     python tools/ingest.py https://example.com"
```

- [ ] **Step 2: Rendre exécutable**

```bash
chmod +x ~/Secretarius/worktrees/intégration/install.sh
```

- [ ] **Step 3: Vérifier la syntaxe bash**

```bash
bash -n ~/Secretarius/worktrees/intégration/install.sh
```

- [ ] **Step 4: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add install.sh
git commit -m "feat: ajouter script d'installation idempotent"
```

---

### Task 8: Documentation — architecture et history

**Files:**
- Create: `docs/architecture/llm-wiki-pattern.md`
- Create: `docs/history/HistoriqueSecretarius.md`

- [ ] **Step 1: Créer les répertoires**

```bash
mkdir -p ~/Secretarius/worktrees/intégration/docs/{architecture,history}
```

- [ ] **Step 2: Créer llm-wiki-pattern.md**

Ce fichier documente le pattern LLM Wiki d'Andrej Karpathy adapté au projet. Contenu à rédiger basé sur `Wiki_LM/PATTERN.md` existant — le copier/adapté :

```bash
cp ~/Secretarius/worktrees/intégration/Wiki_LM/PATTERN.md ~/Secretarius/worktrees/intégration/docs/architecture/llm-wiki-pattern.md
```

- [ ] **Step 3: Créer les fichiers d'historique (vides avec en-tête si pas de source)**

```bash
cat > ~/Secretarius/worktrees/intégration/docs/history/HistoriqueSecretarius.md << 'EOF'
# Historique Secretarius

> Document de suivi de l'historique du projet Secretarius.

## Origine

Projet initié en 2026 — assistant documentaire personnel local et frugal.

## Jalons

- 2026-04-27 : Design d'intégration approuvé (spec dans `docs/superpowers/specs/`)
- 2026-04-27 : Création du worktree `intégration`
EOF
```

- [ ] **Step 4: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add docs/architecture/ docs/history/
git commit -m "docs: ajouter documentation architecture et historique"
```

---

### Task 9: README.md — mise à jour de la structure

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Remplacer la section "Structure du dépôt" dans README.md**

Remplacer le bloc de structure existant (lignes 44-55) par :

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
cd ~/Secretarius/worktrees/intégration
git add README.md
git commit -m "docs: mettre à jour la structure du dépôt dans README"
```

---

### Task 10: Copier install.conf dans le worktree et commit final

**Files:**
- Copy: `install.conf` → worktree (déjà présent à la racine du repo principal)

- [ ] **Step 1: Vérifier que install.conf est dans le worktree**

```bash
ls -la ~/Secretarius/worktrees/intégration/install.conf 2>&1 || echo "MANQUANT"
```

Si manquant, le copier :

```bash
cp ~/Secretarius/install.conf ~/Secretarius/worktrees/intégration/
```

- [ ] **Step 2: Supprimer continuation.md (document de travail temporaire)**

```bash
cd ~/Secretarius/worktrees/intégration
git rm continuation.md 2>/dev/null || rm continuation.md
```

- [ ] **Step 3: Commit final de consolidation**

```bash
cd ~/Secretarius/worktrees/intégration
git add -A
git status
git commit -m "feat: intégration complète — structure, install, templates, docs"
```

---

### Task 11: Vérification finale

- [ ] **Step 1: Vérifier la structure complète**

```bash
cd ~/Secretarius/worktrees/intégration
find . -not -path './.git/*' -not -path './.git' -not -path './Wiki_LM/.git/*' | sort
```

- [ ] **Step 2: Vérifier que install.sh passe la syntaxe**

```bash
bash -n ~/Secretarius/worktrees/intégration/install.sh && echo "Syntaxe OK"
bash -n ~/Secretarius/worktrees/intégration/openclaw-config/install.sh && echo "Syntaxe OK"
```

- [ ] **Step 3: Vérifier les tests Wiki_LM (si environnement disponible)**

```bash
cd ~/Secretarius/worktrees/intégration/Wiki_LM
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 4: Vérifier le log git**

```bash
cd ~/Secretarius/worktrees/intégration
git log --oneline
```

---

## Self-Review

### 1. Couverture du spec

| Requirement spec | Task |
|------------------|------|
| install.sh idempotent, non-interactif + --interactive | Task 7 |
| install.conf sourceable | Task 10 (déjà existant) |
| Vérification prérequis (Python 3.11+, OpenClaw, git) | Task 7, Step 1 |
| Validation coffre Obsidian | Task 7, Step 2 |
| Génération ~/.openclaw/ (json + env + service) | Tasks 3, 4, 5, 6 |
| Configuration Wiki_LM/.env | Task 7, Step 4 |
| Création répertoires raw/, wiki/ | Task 7, Step 5 |
| Installation dépendances Python | Task 7, Step 6 |
| openclaw.json.template avec placeholders | Task 3 |
| gateway.systemd.env.template secrets vides | Task 4 |
| openclaw-gateway.service systemd user | Task 5 |
| Permissions 600 sur gateway.systemd.env | Task 6 |
| .gitignore complet | Task 1 |
| docs/architecture/, docs/history/ | Task 8 |
| README mis à jour | Task 9 |
| Vérification version OpenClaw | Task 7 |
| Secrets jamais commités | Task 1 (.gitignore) + Task 4 (template vide) |

### 2. Scan de placeholders

Aucun "TBD", "TODO", "implement later" dans le plan. Chaque step contient le code exact.

### 3. Cohérence des types/noms

- Variables cohérentes entre `install.conf`, `install.sh`, et les templates : `OBSIDIAN_PATH`, `ASSISTANT_NAME`, `LLM_BACKEND`, `OPENCLAW_PATH`, `TELEGRAM_BOT_TOKEN`, `GATEWAY_TOKEN`, `GATEWAY_PASSWORD`
- `envsubst` utilise la syntaxe `${VAR}` cohérente avec les templates
- Chemins cohérents : `${SECRETARIUS_ROOT}`, `${WIKI_LM_PATH}`, `${OPENCLAW_CONFIG_PATH}`

---

Plan complet. Prêt pour exécution.
