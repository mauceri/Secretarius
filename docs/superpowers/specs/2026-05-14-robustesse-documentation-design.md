# Secretarius — Robustesse & Documentation (v1)

Date : 2026-05-14
Statut : Approuvé
Contexte : `~/Documents/Arbath/Secretarius_dev/Que_faire.md`

## Objectif

Deux axes d'amélioration du dépôt Secretarius :

1. **Robustesse de `install.sh`** : distinguer prérequis bloquants et non-bloquants,
   fournir des commandes de remédiation précises, afficher un résumé des avertissements
   en fin d'installation.

2. **Documentation complète** : un fichier `docs/components/<composant>.md` par composant,
   ingéré dans le wiki Obsidian comme démonstration du patron Karpathy. Implémentation
   complète des trois skills manquants (`prompt-injection-guard`,
   `email-prompt-injection-defense`, `superpowers`).

---

## Axe 1 — Robustesse de `install.sh`

### Catégories de prérequis

| Prérequis | Catégorie | Comportement |
|-----------|-----------|--------------|
| Python 3.11+ | Bloquant | `exit 1` + commande de remédiation |
| git | Bloquant | `exit 1` + commande de remédiation |
| envsubst (`gettext`) | Bloquant | `exit 1` + commande de remédiation |
| openclaw | Avertissement | warn + `npm install -g openclaw` |
| pip3 / venv | Avertissement | warn + alternative venv + commande |
| systemd user (nouveau) | Avertissement | warn si absent (WSL, macOS) |

### Commandes de remédiation

Chaque prérequis bloquant manquant affiche la commande pour le corriger :

```
[ERREUR] Python 3.11+ requis (trouvé 3.10.x)
         Ubuntu/Debian : sudo apt install python3.11
         macOS         : brew install python@3.11
```

```
[ERREUR] envsubst non trouvé
         Ubuntu/Debian : sudo apt install gettext
         macOS         : brew install gettext && brew link gettext
```

### Accumulation des avertissements

Les avertissements non-bloquants sont accumulés dans un tableau `WARNINGS[]`.
En fin de script, si `WARNINGS` est non vide, un bloc "Points d'attention" est affiché :

```
=== Installation terminee ===

Points d'attention :
  - openclaw non trouve -> le service restera inactif
    Installer : npm install -g openclaw
  - systemd user non disponible -> demarrer openclaw manuellement
    openclaw start
```

Si aucun avertissement : le bloc n'apparaît pas.

### Ce qui ne change pas

- Structure générale (5 étapes, options CLI, idempotence, `--force`)
- Coffre Obsidian, génération templates via `envsubst`, dépendances Python

---

## Axe 2 — Documentation

### Structure `docs/components/`

Un fichier par composant, template uniforme :

```markdown
# Composant : <nom>
## Rôle
## Prérequis
## Installation
## Désinstallation
## Configuration
## Usage
## Notes d'architecture
```

**Fichiers à créer :**

| Fichier | Composant |
|---------|-----------|
| `docs/components/wiki-lm.md` | Pipeline Wiki_LM + section par outil (`ingest.py`, `query.py`, `search.py`, `lint.py`, `cluster.py`, `kb_*.py`, `capture.py`, `embed.py`, etc.) |
| `docs/components/scout.md` | Agent d'isolation web, watcher, format tâches/résultats |
| `docs/components/obsidian.md` | Installation headless sync, archivage coffre |
| `docs/components/prompt-injection-guard.md` | Interface, heuristiques, intégration avec scout/email/c |
| `docs/components/email-prompt-injection-defense.md` | IMAP générique + Gmail OAuth2, secrets requis |
| `docs/components/c-telegram.md` | Connexion Telegram à Secretarius, BotFather, appairage |
| `docs/components/superpowers.md` | Plugin Claude Code, workflow brainstorming→writing-plans→executing-plans |
| `docs/components/secretarius-document-normalizer.md` | Schéma `secretarius.document.v0.1`, pipeline d'orchestration |
| `docs/components/switch-model.md` | Modèles disponibles, bascule, redémarrage gateway |
| `docs/components/gog.md` | Google Workspace CLI, OAuth, commandes Gmail/Calendar/Drive |

### Intégration wiki (patron Karpathy)

Après chaque `docs/components/<composant>.md` écrit et commité, ingestion immédiate :

```bash
cd ~/Secretarius/Wiki_LM && source .venv/bin/activate
python tools/ingest.py ../docs/components/<composant>.md
```

La doc devient une page wiki : résumé, liens entre composants, page d'entité.
Cette pratique est documentée dans `PATTERN.md` (paragraphe "Documentation comme source wiki").

---

## Axe 3 — Trois skills manquants

### 3.1 `prompt-injection-guard`

**Rôle :** Couche de sanitisation partagée entre tous les canaux d'entrée (scout, email,
Telegram). Analyse un texte non fiable et retourne un verdict structuré.

**Fichiers :**

```
openclaw-config/workspace/skills/prompt-injection-guard/
├── SKILL.md       — interface pour l'agent OpenClaw
└── guard.py       — détection heuristique (stdlib Python, zéro dépendance externe)
```

**Interface `guard.py` :**

```bash
echo "<texte>" | python guard.py
# ou
python guard.py --text "<texte>"
```

Sortie JSON :
```json
{
  "verdict": "safe|suspect|blocked",
  "score": 0.0,
  "reasons": ["..."]
}
```

**Heuristiques (sans ML) :**
- Patterns d'injection connus : "ignore previous", "you are now", "disregard",
  "forget your instructions", "jailbreak", "do anything now"
- Instructions imbriquées : balises `<`, `[INST]`, markdown structurel dans du
  contenu censément non formaté
- Densité anormale de verbes impératifs en début de phrase
- Unicode suspect : homoglyphes, caractères RTL override (U+202E, etc.)

**Seuils :**
- `score < 0.3` → `safe`
- `0.3 ≤ score < 0.7` → `suspect` (passer à l'agent avec avertissement)
- `score ≥ 0.7` → `blocked` (ne pas transmettre à l'agent principal)

**Intégration :** scout, email-prompt-injection-defense et c appellent `guard.py`
avant de transmettre tout contenu externe à l'agent principal.

```
scout ──────────┐
email-defense ──┼──► prompt-injection-guard ──► agent principal
c (Telegram) ───┘
```

### 3.2 `email-prompt-injection-defense`

**Rôle :** Lire les emails entrants via IMAP générique ou Gmail OAuth2, filtrer
les tentatives d'injection avant de passer le contenu à l'agent principal.

**Fichiers :**

```
openclaw-config/workspace/skills/email-prompt-injection-defense/
├── SKILL.md
└── email_fetch.py   — IMAP + Gmail API, filtre via guard.py
```

**Backends supportés :**

- **IMAP générique** : tout serveur IMAP (Outlook, Protonmail, serveur auto-hébergé)
- **Gmail OAuth2** : via `google-auth` + `google-api-python-client`

Sélection automatique selon les variables d'environnement disponibles :
- `GMAIL_CLIENT_ID` présent → backend Gmail
- Sinon → backend IMAP (`IMAP_HOST`, `IMAP_USER`, `IMAP_PASSWORD`)

**Variables d'environnement** (dans `gateway.systemd.env`) :

```bash
# IMAP générique
IMAP_HOST=
IMAP_USER=
IMAP_PASSWORD=

# Gmail OAuth2 (optionnel)
GMAIL_CLIENT_ID=
GMAIL_CLIENT_SECRET=
GMAIL_REFRESH_TOKEN=
```

**`install.sh`** : vérifie `google-auth` si `GMAIL_CLIENT_ID` est défini (avertissement
non-bloquant si absent).

### 3.3 `superpowers`

**Deux composantes :**

**a) Plugin Claude Code** — déjà présent sur la machine de développement
(`~/.codex/.tmp/plugins/plugins/superpowers/`).

Source : `https://github.com/anthropics/claude-code-config` (répertoire `plugins/superpowers/`).
À confirmer lors de l'implémentation si l'URL a changé.

`install.sh` ajoute une étape non-bloquante :
```bash
# Vérifier/installer le plugin superpowers pour Claude Code
SUPERPOWERS_PLUGIN_DIR="${HOME}/.claude/plugins/superpowers"
if command -v claude &>/dev/null; then
  if [[ ! -d "$SUPERPOWERS_PLUGIN_DIR" ]]; then
    git clone --depth=1 --filter=blob:none --sparse \
      https://github.com/anthropics/claude-code-config \
      /tmp/claude-code-config 2>/dev/null \
      && git -C /tmp/claude-code-config sparse-checkout set plugins/superpowers \
      && cp -r /tmp/claude-code-config/plugins/superpowers "$SUPERPOWERS_PLUGIN_DIR" \
      && rm -rf /tmp/claude-code-config \
      && info "Plugin superpowers installé" \
      || warn "impossible d'installer le plugin superpowers — à faire manuellement"
  else
    info "Plugin superpowers déjà installé ✓"
  fi
else
  warn "claude non trouvé — plugin superpowers non installé"
fi
```

**b) Skill OpenClaw** :

```
openclaw-config/workspace/skills/superpowers/
└── SKILL.md   — workflow brainstorming→writing-plans→executing-plans pour OpenClaw
```

Le SKILL.md explique à l'agent OpenClaw comment utiliser la méthodologie superpowers
pour les tâches de développement : exploration → spec → plan → implémentation.

---

## `gateway.systemd.env.template` — version cible

```bash
# gateway.systemd.env — Secrets Secretarius (chmod 600, jamais commite)

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
# IMAP_HOST=                   # email-prompt-injection-defense
# IMAP_USER=
# IMAP_PASSWORD=
# GMAIL_CLIENT_ID=             # email-prompt-injection-defense (Gmail OAuth2)
# GMAIL_CLIENT_SECRET=
# GMAIL_REFRESH_TOKEN=
```

`GATEWAY_TOKEN` (ancienne variable) est remplacé par `OPENCLAW_GATEWAY_TOKEN` pour
cohérence avec le nommage OpenClaw.

**Migration :** `install.sh` accepte les deux noms (`GATEWAY_TOKEN` ou
`OPENCLAW_GATEWAY_TOKEN`) pour la compatibilité descendante ; si seul `GATEWAY_TOKEN`
est défini, sa valeur est copiée dans `OPENCLAW_GATEWAY_TOKEN` avec un avertissement
invitant à mettre à jour `gateway.systemd.env`.

`install.sh` avertit si `TELEGRAM_BOT_TOKEN` ou `OPENCLAW_GATEWAY_TOKEN` sont vides
après l'étape secrets.

---

## Ordre d'implémentation (Approche B — un composant à la fois)

1. `install.sh` robustesse + `gateway.systemd.env.template` — fondation
2. `docs/components/` pour les skills existants + ingestion wiki progressive
3. `prompt-injection-guard` — implémentation + doc + ingestion wiki
4. `email-prompt-injection-defense` — implémentation + doc + ingestion wiki
5. `superpowers` — implémentation + doc + ingestion wiki
6. Intégration des 3 nouveaux skills dans `install.sh`
7. Enrichissement `PATTERN.md` (paragraphe "Documentation comme source wiki")

---

## Fichiers modifies ou crees

| Fichier | Action |
|---------|--------|
| `install.sh` | Modifier — robustesse prérequis, accumulation warnings |
| `openclaw-config/gateway.systemd.env.template` | Modifier — variables enrichies |
| `docs/components/*.md` (×10) | Créer |
| `openclaw-config/workspace/skills/prompt-injection-guard/SKILL.md` | Créer |
| `openclaw-config/workspace/skills/prompt-injection-guard/guard.py` | Créer |
| `openclaw-config/workspace/skills/email-prompt-injection-defense/SKILL.md` | Créer |
| `openclaw-config/workspace/skills/email-prompt-injection-defense/email_fetch.py` | Créer |
| `openclaw-config/workspace/skills/superpowers/SKILL.md` | Créer |
| `PATTERN.md` | Modifier — paragraphe "Documentation comme source wiki" |

## Hors périmètre

- Logique interne des agents scout / c (non modifiée)
- Wiki_LM/tools (non modifiés)
- Données wiki (`raw/`, `wiki/`, `embeddings/`)
