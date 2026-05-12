# Intégration Secretarius — Plan d'implémentation (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réécrire proprement les scripts d'installation et les templates OpenClaw pour produire un dépôt GitHub partageable permettant à un tiers de déployer son propre couple openclaw + Wiki_LM.

**Architecture:** `install.sh` idempotent + `envsubst` pour générer `~/.openclaw/openclaw.json` depuis un template sans secrets ni chemins machine ; `Wiki_LM/.env` configuré selon le backend LLM choisi ; structure `docs/` organisée.

**Tech Stack:** bash, envsubst (gettext), python3, systemd user units

---

## Fichiers à créer ou modifier

| Fichier | Action |
|---------|--------|
| `docs/history/point-11-05-2026.md` | Déplacer depuis la racine |
| `openclaw-config/openclaw.json.template` | Régénérer depuis `~/.openclaw/openclaw.json` |
| `openclaw-config/install.sh` | Réécrire intégralement |
| `Wiki_LM/.env.template` | Créer |
| `install.sh` | Réécrire intégralement |
| `install.conf` | Vérifier et ajuster |

---

## Task 1 : Déplacer point-11-05-2026.md dans docs/history/

**Files:**
- Modify: `point-11-05-2026.md` → `docs/history/point-11-05-2026.md`

- [ ] **Déplacer le fichier**

```bash
git -C ~/Secretarius mv point-11-05-2026.md docs/history/point-11-05-2026.md
```

- [ ] **Vérifier**

```bash
ls ~/Secretarius/docs/history/
```
Attendu : `HistoriqueSecretarius.md  point-11-05-2026.md`

- [ ] **Committer**

```bash
git -C ~/Secretarius commit -m "docs: déplacer point-11-05-2026.md dans docs/history/"
```

---

## Task 2 : Générer openclaw.json.template depuis la config live

**Files:**
- Create/Modify: `openclaw-config/openclaw.json.template`

La config live `~/.openclaw/openclaw.json` contient des chemins absolus (`/home/mauceric`) et le hostname (`sanroque`) à remplacer par des variables shell, et une section `llamacpp` (modèle LoRA local) à supprimer.

- [ ] **Supprimer la section llamacpp et substituer les chemins**

```bash
python3 - << 'EOF'
import json, re

with open('/home/mauceric/.openclaw/openclaw.json') as f:
    d = json.load(f)

# Supprimer le provider llamacpp
providers = d.get('models', {}).get('providers', {})
providers.pop('llamacpp', None)

# Supprimer l'alias llamacpp dans agents.defaults.models
agent_models = d.get('agents', {}).get('defaults', {}).get('models', {})
keys_to_remove = [k for k in agent_models if k.startswith('llamacpp/')]
for k in keys_to_remove:
    del agent_models[k]

# Sérialiser
content = json.dumps(d, indent=2, ensure_ascii=False)

# Substituer les chemins machine
content = content.replace('/home/mauceric', '${HOME}')
content = content.replace('sanroque', '${HOSTNAME}')

with open('/home/mauceric/Secretarius/openclaw-config/openclaw.json.template', 'w') as f:
    f.write(content)
    f.write('\n')

print("Template généré.")
EOF
```

- [ ] **Vérifier qu'aucune occurrence de /home/mauceric ou sanroque ne subsiste**

```bash
grep -c '/home/mauceric\|sanroque' ~/Secretarius/openclaw-config/openclaw.json.template
```
Attendu : `0`

- [ ] **Vérifier que llamacpp est absent**

```bash
grep -c 'llamacpp' ~/Secretarius/openclaw-config/openclaw.json.template
```
Attendu : `0`

- [ ] **Vérifier que le template est du JSON valide une fois les variables substituées**

```bash
HOME=/tmp/testhome HOSTNAME=testhost \
  envsubst < ~/Secretarius/openclaw-config/openclaw.json.template \
  | python3 -c "import json,sys; json.load(sys.stdin); print('JSON valide')"
```
Attendu : `JSON valide`

- [ ] **Committer**

```bash
git -C ~/Secretarius add openclaw-config/openclaw.json.template
git -C ~/Secretarius commit -m "feat(openclaw-config): régénérer openclaw.json.template depuis config live"
```

---

## Task 3 : Réécrire openclaw-config/install.sh

**Files:**
- Modify: `openclaw-config/install.sh`

- [ ] **Écrire le nouveau script**

Remplacer intégralement `~/Secretarius/openclaw-config/install.sh` par :

```bash
#!/usr/bin/env bash
# openclaw-config/install.sh — Génère ~/.openclaw/ depuis les templates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../install.conf"

OBSIDIAN_PATH="${OBSIDIAN_PATH:-$HOME/Documents/Obsidian}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Tiron}"
LLM_BACKEND="${LLM_BACKEND:-deepseek}"
OPENCLAW_PATH="${OPENCLAW_PATH:-$HOME/.openclaw}"
FORCE="${FORCE:-false}"

TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*"; }
error() { echo "[ERREUR] $*" >&2; }

mkdir -p "$OPENCLAW_PATH"

# openclaw.json
TARGET="${OPENCLAW_PATH}/openclaw.json"
if [[ -f "$TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw.json existe déjà — ignoré (utilisez --force pour écraser)"
else
  export HOME HOSTNAME OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND
  envsubst '${HOME} ${HOSTNAME} ${OBSIDIAN_PATH} ${ASSISTANT_NAME} ${LLM_BACKEND}' \
    < "${SCRIPT_DIR}/openclaw.json.template" \
    > "$TARGET"
  info "openclaw.json généré dans ${OPENCLAW_PATH}"
fi

# gateway.systemd.env
ENV_TARGET="${OPENCLAW_PATH}/gateway.systemd.env"
if [[ -f "$ENV_TARGET" && "$FORCE" != "true" ]]; then
  info "gateway.systemd.env existe déjà — ignoré"
else
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — gateway.systemd.env généré avec des valeurs vides"
  fi
  export TELEGRAM_BOT_TOKEN GATEWAY_TOKEN GATEWAY_PASSWORD
  envsubst '${TELEGRAM_BOT_TOKEN} ${GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
  chmod 600 "$ENV_TARGET"
  info "gateway.systemd.env généré (600)"
fi

# Service systemd user
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
SERVICE_TARGET="${SYSTEMD_USER_DIR}/openclaw-gateway.service"
if [[ -f "$SERVICE_TARGET" && "$FORCE" != "true" ]]; then
  info "openclaw-gateway.service existe déjà — ignoré"
else
  cp "${SCRIPT_DIR}/openclaw-gateway.service" "$SERVICE_TARGET"
  info "Service systemd installé dans ${SYSTEMD_USER_DIR}"
fi
```

- [ ] **Rendre le script exécutable**

```bash
chmod +x ~/Secretarius/openclaw-config/install.sh
```

- [ ] **Tester en mode simulation (sans écraser la config live)**

```bash
OPENCLAW_PATH=/tmp/test-openclaw \
OBSIDIAN_PATH=/tmp/obsidian \
FORCE=true \
bash ~/Secretarius/openclaw-config/install.sh
```
Attendu : les trois fichiers générés sans erreur dans `/tmp/test-openclaw/`, `gateway.systemd.env` avec permissions 600.

```bash
ls -la /tmp/test-openclaw/
stat -c "%a %n" /tmp/test-openclaw/gateway.systemd.env
```
Attendu : `600 /tmp/test-openclaw/gateway.systemd.env`

- [ ] **Committer**

```bash
git -C ~/Secretarius add openclaw-config/install.sh
git -C ~/Secretarius commit -m "feat(openclaw-config): réécrire install.sh — idempotent, --force, sans blocage secrets"
```

---

## Task 4 : Créer Wiki_LM/.env.template

**Files:**
- Create: `Wiki_LM/.env.template`

Les outils Wiki_LM lisent `Wiki_LM/.env` via `_load_dotenv()`. Les variables pertinentes sont dans `tools/llm.py` : `WIKI_LLM_BACKEND`, `DEEPSEEK_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, et dans les outils : `WIKI_PATH`, `WIKI_RAW_PATH`.

- [ ] **Créer le fichier**

```
# Wiki_LM/.env.template — Copier en .env et remplir les valeurs
#
# Backend LLM : openai (pour DeepSeek/compatible) | ollama | claude
WIKI_LLM_BACKEND=openai

# Pour DeepSeek (WIKI_LLM_BACKEND=openai)
DEEPSEEK_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1

# Pour un serveur local OpenAI-compatible (Ollama, LM Studio, etc.)
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama

# Pour Anthropic Claude (WIKI_LLM_BACKEND=claude)
# ANTHROPIC_API_KEY=

# Chemin du wiki actif (répertoire de fichiers Markdown)
WIKI_PATH=~/Documents/Obsidian/Wiki_LM

# Chemin des sources brutes (optionnel, défaut: WIKI_PATH/../raw)
# WIKI_RAW_PATH=~/Documents/Obsidian/Wiki_LM/raw

# Port du serveur Flask (optionnel, défaut: 5051)
# WIKI_SERVER_PORT=5051
```

- [ ] **Vérifier que le fichier est exclu par .gitignore** (seul le template est versionné)

```bash
echo "Wiki_LM/.env" | git -C ~/Secretarius check-ignore --stdin
```
Attendu : `Wiki_LM/.env`

- [ ] **Committer**

```bash
git -C ~/Secretarius add Wiki_LM/.env.template
git -C ~/Secretarius commit -m "feat(wiki-lm): ajouter .env.template"
```

---

## Task 5 : Réécrire install.sh et mettre à jour install.conf

**Files:**
- Modify: `install.sh`
- Modify: `install.conf`

- [ ] **Mettre à jour install.conf** pour supprimer les références à Prototype/

Lire `~/Secretarius/install.conf`. Vérifier qu'il ne contient aucune référence à `Prototype` ni à `/home/mauceric`. S'il en contient, les corriger. La version actuelle est déjà correcte (ne contient que des chemins relatifs) — aucune modification requise sauf vérification.

```bash
grep -n 'Prototype\|/home/mauceric' ~/Secretarius/install.conf || echo "Aucune référence — OK"
```
Attendu : `Aucune référence — OK`

- [ ] **Réécrire install.sh**

Remplacer intégralement `~/Secretarius/install.sh` par :

```bash
#!/usr/bin/env bash
# install.sh — Installation idempotente de Secretarius
# Usage: ./install.sh [--obsidian-path PATH] [--assistant-name NAME] [--llm BACKEND]
#                     [--openclaw-path PATH] [--env-file FILE] [--interactive] [--force] [--help]
set -euo pipefail

SECRETARIUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SECRETARIUS_ROOT}/install.conf"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERREUR]${NC} $*" >&2; }

INTERACTIVE=false
FORCE=false
ENV_FILE=""

usage() {
  cat << 'EOF'
Usage: ./install.sh [options]

  --obsidian-path PATH    Chemin du coffre Obsidian (défaut: ~/Documents/Obsidian)
  --assistant-name NAME   Nom de l'assistant (défaut: Tiron)
  --llm BACKEND           deepseek | ollama | claude (défaut: deepseek)
  --openclaw-path PATH    Chemin config OpenClaw (défaut: ~/.openclaw)
  --env-file FILE         Fichier de secrets (API keys, tokens)
  --interactive           Pose les questions une par une
  --force                 Écrase les fichiers existants
  --help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --obsidian-path)  OBSIDIAN_PATH="$2"; shift 2 ;;
    --assistant-name) ASSISTANT_NAME="$2"; shift 2 ;;
    --llm)            LLM_BACKEND="$2"; shift 2 ;;
    --openclaw-path)  OPENCLAW_PATH="$2"; shift 2 ;;
    --env-file)       ENV_FILE="$2"; shift 2 ;;
    --interactive)    INTERACTIVE=true; shift ;;
    --force)          FORCE=true; shift ;;
    --help)           usage ;;
    *) error "Option inconnue: $1"; usage ;;
  esac
done

# Mode interactif
if [[ "$INTERACTIVE" == true ]]; then
  echo "=== Installation Secretarius ==="
  read -rp "Coffre Obsidian [${OBSIDIAN_PATH}]: " v; OBSIDIAN_PATH="${v:-$OBSIDIAN_PATH}"
  read -rp "Nom de l'assistant [${ASSISTANT_NAME}]: " v; ASSISTANT_NAME="${v:-$ASSISTANT_NAME}"
  read -rp "LLM (deepseek|ollama|claude) [${LLM_BACKEND}]: " v; LLM_BACKEND="${v:-$LLM_BACKEND}"
  read -rp "Config OpenClaw [${OPENCLAW_PATH}]: " v; OPENCLAW_PATH="${v:-$OPENCLAW_PATH}"
  read -rp "Fichier de secrets (optionnel): " v; ENV_FILE="${v:-$ENV_FILE}"
fi

# Charger les secrets
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || { error "Fichier introuvable: $ENV_FILE"; exit 1; }
  source "$ENV_FILE"
fi

export OBSIDIAN_PATH ASSISTANT_NAME LLM_BACKEND OPENCLAW_PATH FORCE

# Étape 1 — Prérequis
info "Vérification des prérequis..."

python3 -c "import sys; assert sys.version_info >= (3,11), f'Python 3.11+ requis (trouvé {sys.version})'" \
  && info "Python $(python3 --version | cut -d' ' -f2) ✓" \
  || { error "Python 3.11+ requis"; exit 1; }

command -v git &>/dev/null && info "git ✓" || { error "git requis"; exit 1; }
command -v envsubst &>/dev/null && info "envsubst ✓" \
  || { error "envsubst requis (apt install gettext / brew install gettext)"; exit 1; }
command -v openclaw &>/dev/null \
  && info "openclaw $(openclaw --version 2>/dev/null || echo '?') ✓" \
  || warn "openclaw non trouvé — config générée mais service inactif"

# Étape 2 — Coffre Obsidian
info "Validation du coffre Obsidian: ${OBSIDIAN_PATH}"
OBSIDIAN_PATH="${OBSIDIAN_PATH/#\~/$HOME}"
if [[ ! -d "$OBSIDIAN_PATH" ]]; then
  read -rp "Répertoire absent. Créer ? [y/N] " c
  [[ "$c" =~ ^[Yy] ]] && mkdir -p "$OBSIDIAN_PATH" && info "Créé: $OBSIDIAN_PATH" \
    || { error "Coffre Obsidian requis. Annulation."; exit 1; }
fi
info "Coffre Obsidian ✓"

# Étape 3 — Config OpenClaw
info "Génération de la configuration OpenClaw..."
OPENCLAW_PATH="${OPENCLAW_PATH/#\~/$HOME}"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
export GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
bash "${SECRETARIUS_ROOT}/openclaw-config/install.sh"

# Étape 4 — Wiki_LM/.env
info "Configuration de Wiki_LM/.env..."
WIKI_ENV="${SECRETARIUS_ROOT}/Wiki_LM/.env"
WIKI_ENV_TEMPLATE="${SECRETARIUS_ROOT}/Wiki_LM/.env.template"
if [[ -f "$WIKI_ENV" && "$FORCE" != "true" ]]; then
  info "Wiki_LM/.env existe déjà — ignoré"
else
  cp "$WIKI_ENV_TEMPLATE" "$WIKI_ENV"
  WIKI_PATH="${OBSIDIAN_PATH}/Wiki_LM"
  sed -i "s|^WIKI_PATH=.*|WIKI_PATH=${WIKI_PATH}|" "$WIKI_ENV"
  case "$LLM_BACKEND" in
    deepseek) sed -i "s|^WIKI_LLM_BACKEND=.*|WIKI_LLM_BACKEND=openai|" "$WIKI_ENV" ;;
    ollama)   sed -i "s|^WIKI_LLM_BACKEND=.*|WIKI_LLM_BACKEND=ollama|" "$WIKI_ENV"
              sed -i "s|^OPENAI_BASE_URL=.*|# OPENAI_BASE_URL=https://api.deepseek.com/v1|" "$WIKI_ENV" ;;
    claude)   sed -i "s|^WIKI_LLM_BACKEND=.*|WIKI_LLM_BACKEND=claude|" "$WIKI_ENV" ;;
  esac
  info "Wiki_LM/.env créé"
fi

# Étape 5 — Dépendances Python
info "Installation des dépendances Python..."
WIKI_LM_PATH="${SECRETARIUS_ROOT}/Wiki_LM"
if command -v pip3 &>/dev/null; then
  pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet
  info "Dépendances Python ✓"
else
  warn "pip3 non trouvé — installez manuellement: pip install -r Wiki_LM/requirements.txt"
fi

# Résumé
echo ""
info "=== Installation terminée ==="
echo ""
echo "Prochaines étapes :"
echo "  1. Renseigner les secrets dans ${OPENCLAW_PATH}/gateway.systemd.env"
echo "  2. Activer le service OpenClaw :"
echo "       systemctl --user daemon-reload"
echo "       systemctl --user enable --now openclaw-gateway.service"
echo "  3. Tester Wiki_LM :"
echo "       cd ${WIKI_LM_PATH} && python -m pytest tests/"
echo "  4. Ingérer une première source :"
echo "       python ${WIKI_LM_PATH}/tools/ingest.py https://example.com"
```

- [ ] **Rendre le script exécutable**

```bash
chmod +x ~/Secretarius/install.sh
```

- [ ] **Tester en mode simulation (répertoire temporaire, sans secrets)**

```bash
OPENCLAW_PATH=/tmp/test-openclaw-full \
bash ~/Secretarius/install.sh \
  --obsidian-path /tmp/test-obsidian \
  --assistant-name TestUser \
  --llm deepseek \
  --openclaw-path /tmp/test-openclaw-full \
  --force
```
Attendu : les 5 étapes s'exécutent sans erreur, avertissement sur TELEGRAM_BOT_TOKEN vide accepté.

- [ ] **Vérifier les fichiers produits**

```bash
ls /tmp/test-openclaw-full/
grep WIKI_PATH ~/Secretarius/Wiki_LM/.env
grep WIKI_LLM_BACKEND ~/Secretarius/Wiki_LM/.env
```
Attendu : `WIKI_PATH=/tmp/test-obsidian/Wiki_LM` et `WIKI_LLM_BACKEND=openai`

- [ ] **Nettoyer les fichiers de test**

```bash
rm -rf /tmp/test-openclaw-full /tmp/test-obsidian /tmp/test-openclaw
```

- [ ] **Committer**

```bash
git -C ~/Secretarius add install.sh install.conf
git -C ~/Secretarius commit -m "feat(install): réécrire install.sh — idempotent, --force, --llm, backend Wiki_LM"
```

---

## Task 6 : Vérifier .gitignore et pousser

**Files:**
- Verify: `.gitignore`

- [ ] **Vérifier que Wiki_LM/.env est bien ignoré**

```bash
echo "Wiki_LM/.env" | git -C ~/Secretarius check-ignore --stdin
```
Attendu : `Wiki_LM/.env`

Si absent, ajouter la ligne `Wiki_LM/.env` dans `.gitignore` (la section `# Secrets / config locale`) :

```bash
grep -q "^Wiki_LM/\.env$" ~/Secretarius/.gitignore \
  || sed -i '/# Secrets \/ config locale/a Wiki_LM/.env' ~/Secretarius/.gitignore
```

- [ ] **Vérifier que knowledge_base/ est ignoré**

```bash
echo "knowledge_base/" | git -C ~/Secretarius check-ignore --stdin
```
Si absent, ajouter :

```bash
grep -q "^Wiki_LM/knowledge_base" ~/Secretarius/.gitignore \
  || echo "Wiki_LM/knowledge_base/" >> ~/Secretarius/.gitignore
```

- [ ] **Vérifier l'état git — aucun fichier sensible dans le staging**

```bash
git -C ~/Secretarius status
```
Vérifier qu'aucun `.env`, `gateway.systemd.env`, clé API, ou chemin `/home/mauceric` n'est dans les fichiers à committer.

- [ ] **Committer si .gitignore modifié, puis pousser**

```bash
git -C ~/Secretarius add .gitignore 2>/dev/null
git -C ~/Secretarius diff --cached --quiet || \
  git -C ~/Secretarius commit -m "chore(.gitignore): ajouter Wiki_LM/.env et knowledge_base/"
git -C ~/Secretarius push
```
Attendu : `main -> main` sans erreur.
