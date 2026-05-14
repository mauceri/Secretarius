# Robustesse Installation — Plan d'implémentation (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre `install.sh` robuste (prérequis bloquants/avertissements avec remédiation), mettre à jour `gateway.systemd.env.template` avec toutes les variables secrets, et corriger la migration `GATEWAY_TOKEN` → `OPENCLAW_GATEWAY_TOKEN` dans `openclaw-config/install.sh`.

**Architecture:** Modifications bash pures. `install.sh` accumule les avertissements dans un tableau `WARNINGS[]` et les affiche en bloc final. `openclaw-config/install.sh` gère la compatibilité descendante GATEWAY_TOKEN. Le template gateway liste toutes les variables connues, obligatoires décommentées, optionnelles commentées.

**Tech Stack:** bash, envsubst (gettext)

**Repo sur sanroque :** `~/Secretarius/`

---

## Fichiers modifiés

| Fichier | Action |
|---------|--------|
| `openclaw-config/gateway.systemd.env.template` | Remplacer intégralement |
| `openclaw-config/install.sh` | Modifier — OPENCLAW_GATEWAY_TOKEN + compat GATEWAY_TOKEN |
| `install.sh` | Modifier — prérequis bloquants avec remédiation + WARNINGS[] |

---

## Task 1 : Mettre à jour `gateway.systemd.env.template`

**Files:**
- Modify: `openclaw-config/gateway.systemd.env.template`

- [ ] **Remplacer intégralement le fichier**

Contenu cible :

```bash
# gateway.systemd.env — Secrets Secretarius (chmod 600, jamais commite)
# Genere par openclaw-config/install.sh depuis ce template.
# Editer directement apres installation : nano ~/.openclaw/gateway.systemd.env

# --- Telegram / OpenClaw (obligatoires) ---
TELEGRAM_BOT_TOKEN=
OPENCLAW_GATEWAY_TOKEN=
GATEWAY_PASSWORD=

# --- LLM backends (decomenter selon votre choix dans install.conf) ---
# DEEPSEEK_API_KEY=
# OPENAI_API_KEY=
# GEMINI_API_KEY=
# OPENROUTER_API_KEY=

# --- Skills optionnels ---
# GOG_KEYRING_PASSWORD=        # skill gog
# IMAP_HOST=                   # email-prompt-injection-defense (IMAP)
# IMAP_USER=
# IMAP_PASSWORD=
# GMAIL_CLIENT_ID=             # email-prompt-injection-defense (Gmail OAuth2)
# GMAIL_CLIENT_SECRET=
# GMAIL_REFRESH_TOKEN=
```

- [ ] **Vérifier que le template ne contient plus GATEWAY_TOKEN (ancienne variable)**

```bash
ssh mauceric@sanroque "grep -n 'GATEWAY_TOKEN' ~/Secretarius/openclaw-config/gateway.systemd.env.template"
```

Attendu : uniquement `OPENCLAW_GATEWAY_TOKEN`, pas de `GATEWAY_TOKEN` seul.

- [ ] **Committer**

```bash
ssh mauceric@sanroque "
git -C ~/Secretarius add openclaw-config/gateway.systemd.env.template
git -C ~/Secretarius commit -m 'feat(gateway): enrichir template secrets — LLM backends, skills optionnels, OPENCLAW_GATEWAY_TOKEN'
"
```

---

## Task 2 : Corriger `openclaw-config/install.sh` — migration GATEWAY_TOKEN

**Files:**
- Modify: `openclaw-config/install.sh`

Le script actuel utilise `GATEWAY_TOKEN` pour générer le template. Il faut :
1. Exporter `OPENCLAW_GATEWAY_TOKEN` (nouvelle variable du template)
2. Gérer la compatibilité : si `GATEWAY_TOKEN` est défini mais pas `OPENCLAW_GATEWAY_TOKEN`, utiliser sa valeur

- [ ] **Remplacer le bloc secrets dans `openclaw-config/install.sh`**

Remplacer :
```bash
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
```

Par :
```bash
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
GATEWAY_PASSWORD="${GATEWAY_PASSWORD:-}"
# Migration GATEWAY_TOKEN → OPENCLAW_GATEWAY_TOKEN
if [[ -z "${OPENCLAW_GATEWAY_TOKEN:-}" && -n "${GATEWAY_TOKEN:-}" ]]; then
  warn "GATEWAY_TOKEN est déprécié — utiliser OPENCLAW_GATEWAY_TOKEN dans gateway.systemd.env"
  OPENCLAW_GATEWAY_TOKEN="$GATEWAY_TOKEN"
else
  OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
fi
```

- [ ] **Remplacer le bloc envsubst gateway dans `openclaw-config/install.sh`**

Remplacer :
```bash
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — gateway.systemd.env généré avec des valeurs vides"
  fi
  export TELEGRAM_BOT_TOKEN GATEWAY_TOKEN GATEWAY_PASSWORD
  envsubst '${TELEGRAM_BOT_TOKEN} ${GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
```

Par :
```bash
  if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
    warn "TELEGRAM_BOT_TOKEN non défini — à renseigner dans ${ENV_TARGET} avant de démarrer OpenClaw"
  fi
  if [[ -z "$OPENCLAW_GATEWAY_TOKEN" ]]; then
    warn "OPENCLAW_GATEWAY_TOKEN non défini — à renseigner dans ${ENV_TARGET} avant de démarrer OpenClaw"
  fi
  export TELEGRAM_BOT_TOKEN OPENCLAW_GATEWAY_TOKEN GATEWAY_PASSWORD
  envsubst '${TELEGRAM_BOT_TOKEN} ${OPENCLAW_GATEWAY_TOKEN} ${GATEWAY_PASSWORD}' \
    < "${SCRIPT_DIR}/gateway.systemd.env.template" \
    > "$ENV_TARGET"
```

- [ ] **Tester en simulation**

```bash
ssh mauceric@sanroque "
OPENCLAW_PATH=/tmp/test-oc2 FORCE=true \
bash ~/Secretarius/openclaw-config/install.sh
cat /tmp/test-oc2/gateway.systemd.env
grep 'OPENCLAW_GATEWAY_TOKEN' /tmp/test-oc2/gateway.systemd.env
grep -v 'GATEWAY_TOKEN=' /tmp/test-oc2/gateway.systemd.env | grep 'GATEWAY_TOKEN' || echo 'ancienne variable absente OK'
rm -rf /tmp/test-oc2
"
```

Attendu : `OPENCLAW_GATEWAY_TOKEN=` présent, `GATEWAY_TOKEN=` absent (hors commentaires).

- [ ] **Tester la compatibilité descendante (GATEWAY_TOKEN fourni)**

```bash
ssh mauceric@sanroque "
OPENCLAW_PATH=/tmp/test-oc3 FORCE=true GATEWAY_TOKEN=test123 \
bash ~/Secretarius/openclaw-config/install.sh 2>&1 | grep -E 'WARN|déprécié'
cat /tmp/test-oc3/gateway.systemd.env | grep OPENCLAW_GATEWAY_TOKEN
rm -rf /tmp/test-oc3
"
```

Attendu : avertissement de dépréciation + `OPENCLAW_GATEWAY_TOKEN=test123`.

- [ ] **Committer**

```bash
ssh mauceric@sanroque "
git -C ~/Secretarius add openclaw-config/install.sh
git -C ~/Secretarius commit -m 'fix(openclaw-config): migrer GATEWAY_TOKEN → OPENCLAW_GATEWAY_TOKEN avec compat descendante'
"
```

---

## Task 3 : Robustesse `install.sh` — prérequis bloquants avec remédiation

**Files:**
- Modify: `install.sh`

Actuellement la section prérequis utilise `exit 1` sans message de remédiation.
La réécrire pour afficher les commandes d'installation.

- [ ] **Remplacer le bloc "Étape 1 — Prérequis" dans `install.sh`**

Remplacer :
```bash
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
```

Par :
```bash
# Étape 1 — Prérequis
info "Vérification des prérequis..."
WARNINGS=()

# --- Bloquants ---
if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
  info "Python $(python3 --version | cut -d' ' -f2) ✓"
else
  error "Python 3.11+ requis"
  error "  Ubuntu/Debian : sudo apt install python3.11 python3.11-venv"
  error "  macOS         : brew install python@3.11"
  exit 1
fi

if command -v git &>/dev/null; then
  info "git ✓"
else
  error "git requis"
  error "  Ubuntu/Debian : sudo apt install git"
  error "  macOS         : brew install git"
  exit 1
fi

if command -v envsubst &>/dev/null; then
  info "envsubst ✓"
else
  error "envsubst requis (paquet gettext)"
  error "  Ubuntu/Debian : sudo apt install gettext"
  error "  macOS         : brew install gettext && brew link gettext --force"
  exit 1
fi

# --- Non-bloquants ---
if command -v openclaw &>/dev/null; then
  info "openclaw $(openclaw --version 2>/dev/null | head -1 || echo '?') ✓"
else
  WARNINGS+=("openclaw non trouvé — le service restera inactif\n    Installer : npm install -g openclaw")
fi

if ! systemctl --user status &>/dev/null 2>&1; then
  WARNINGS+=("systemd user non disponible (WSL ou macOS ?) — démarrer openclaw manuellement\n    openclaw start")
fi
```

- [ ] **Tester les prérequis bloquants**

```bash
ssh mauceric@sanroque "
# Simuler python manquant en créant un faux python3 qui renvoie version 3.10
mkdir -p /tmp/fake-bin
echo '#!/bin/bash
if [[ \"\$1\" == \"-c\" ]]; then
  python3.11 \"\$@\" 2>/dev/null || (echo \"Python 3.10\"; exit 1)
fi' > /tmp/fake-bin/python3
chmod +x /tmp/fake-bin/python3
PATH=/tmp/fake-bin:\$PATH bash ~/Secretarius/install.sh --help 2>&1 | head -3 || true
rm -rf /tmp/fake-bin
"
```

Attendu : message `[ERREUR] Python 3.11+ requis` avec commandes Ubuntu/macOS.

- [ ] **Tester que les prérequis bloquants existants passent**

```bash
ssh mauceric@sanroque "
bash ~/Secretarius/install.sh \
  --obsidian-path /tmp/test-obs-prereq \
  --openclaw-path /tmp/test-oc-prereq \
  --force 2>&1 | grep -E 'INFO.*✓|WARN|ERREUR' | head -10
rm -rf /tmp/test-obs-prereq /tmp/test-oc-prereq
"
```

Attendu : `[INFO] Python ... ✓`, `[INFO] git ✓`, `[INFO] envsubst ✓` sans `[ERREUR]`.

- [ ] **Committer**

```bash
ssh mauceric@sanroque "
git -C ~/Secretarius add install.sh
git -C ~/Secretarius commit -m 'feat(install): prérequis bloquants avec commandes de remédiation Ubuntu/macOS'
"
```

---

## Task 4 : Robustesse `install.sh` — WARNINGS[] et résumé final

**Files:**
- Modify: `install.sh`

Ajouter : vérification pip3/venv, vérification google-auth si Gmail configuré,
affichage du bloc "Points d'attention" en fin de script.

- [ ] **Remplacer le bloc "Étape 5 — Dépendances Python" dans `install.sh`**

Remplacer :
```bash
# Étape 5 — Dépendances Python
info "Installation des dépendances Python..."
WIKI_LM_PATH="${SECRETARIUS_ROOT}/Wiki_LM"
if command -v pip3 &>/dev/null; then
  if pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet 2>/dev/null; then
    info "Dépendances Python ✓"
  elif pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet --break-system-packages 2>/dev/null; then
    info "Dépendances Python ✓ (--break-system-packages)"
  else
    warn "pip3 a échoué — installez dans un venv : python -m venv .venv && .venv/bin/pip install -r Wiki_LM/requirements.txt"
  fi
else
  warn "pip3 non trouvé — installez manuellement: pip install -r Wiki_LM/requirements.txt"
fi
```

Par :
```bash
# Étape 5 — Dépendances Python
info "Installation des dépendances Python..."
WIKI_LM_PATH="${SECRETARIUS_ROOT}/Wiki_LM"
if command -v pip3 &>/dev/null; then
  if pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet 2>/dev/null; then
    info "Dépendances Python ✓"
  elif pip3 install -r "${WIKI_LM_PATH}/requirements.txt" --quiet --break-system-packages 2>/dev/null; then
    info "Dépendances Python ✓ (--break-system-packages)"
  else
    WARNINGS+=("dépendances Python non installées\n    Installer dans un venv :\n    python3 -m venv ${WIKI_LM_PATH}/.venv && ${WIKI_LM_PATH}/.venv/bin/pip install -r ${WIKI_LM_PATH}/requirements.txt")
  fi
else
  WARNINGS+=("pip3 non trouvé — dépendances Python non installées\n    Ubuntu/Debian : sudo apt install python3-pip\n    Puis : pip3 install -r ${WIKI_LM_PATH}/requirements.txt")
fi

# Vérification google-auth si Gmail configuré
if [[ -n "${GMAIL_CLIENT_ID:-}" ]]; then
  if ! python3 -c "import google.auth" 2>/dev/null; then
    WARNINGS+=("google-auth non installé (requis pour Gmail OAuth2)\n    pip3 install google-auth google-api-python-client")
  fi
fi
```

- [ ] **Ajouter le bloc résumé à la fin de `install.sh`**, juste avant la dernière ligne `echo` du résumé existant

Remplacer :
```bash
# Résumé
echo ""
info "=== Installation terminée ==="
```

Par :
```bash
# Résumé
echo ""
info "=== Installation terminée ==="
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  echo ""
  warn "Points d'attention :"
  for w in "${WARNINGS[@]}"; do
    echo -e "  - ${w}"
  done
fi
```

- [ ] **Tester le bloc WARNINGS avec openclaw absent**

```bash
ssh mauceric@sanroque "
# Masquer openclaw temporairement
PATH_SANS_NVM=\$(echo \$PATH | tr ':' '\n' | grep -v openclaw | grep -v nvm | tr '\n' ':')
OPENCLAW_PATH=/tmp/test-warn OBSIDIAN_PATH=/tmp/test-obs-warn \
PATH=\$PATH_SANS_NVM bash ~/Secretarius/install.sh --force 2>&1 | grep -A5 'Points d.attention' || echo 'bloc absent'
rm -rf /tmp/test-warn /tmp/test-obs-warn
"
```

Attendu : bloc "Points d'attention" avec ligne openclaw et commande npm.

- [ ] **Tester installation propre (tous prérequis présents)**

```bash
ssh mauceric@sanroque "
OPENCLAW_PATH=/tmp/test-clean OBSIDIAN_PATH=/tmp/test-obs-clean \
bash ~/Secretarius/install.sh --force 2>&1 | grep -c 'Points d.attention' || echo '0'
rm -rf /tmp/test-clean /tmp/test-obs-clean
"
```

Attendu : `0` (bloc absent quand tout est OK).

- [ ] **Committer**

```bash
ssh mauceric@sanroque "
git -C ~/Secretarius add install.sh
git -C ~/Secretarius commit -m 'feat(install): accumuler avertissements WARNINGS[] et afficher résumé en fin d installation'
"
```

---

## Task 5 : Pousser sur GitHub et vérification finale

- [ ] **Pousser toutes les modifications**

```bash
ssh mauceric@sanroque "git -C ~/Secretarius push"
```

- [ ] **Vérifier le log**

```bash
ssh mauceric@sanroque "git -C ~/Secretarius log --oneline -5"
```

Attendu : 4 commits récents (gateway template, openclaw-config/install.sh,
install.sh bloquants, install.sh WARNINGS).

- [ ] **Test d'intégration complet**

```bash
ssh mauceric@sanroque "
OPENCLAW_PATH=/tmp/final-test OBSIDIAN_PATH=/tmp/final-obs \
bash ~/Secretarius/install.sh \
  --assistant-name TestUser \
  --llm deepseek \
  --force 2>&1
rm -rf /tmp/final-test /tmp/final-obs
"
```

Attendu : les 5 étapes s'exécutent, warnings eventuels pour openclaw/systemd,
**pas de ERREUR**, bloc "Points d'attention" si nécessaire.
