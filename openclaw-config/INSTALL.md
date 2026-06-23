# Installation de Secretarius (OpenClaw)

## Prérequis

| Composant | Version minimale | Notes |
|-----------|-----------------|-------|
| Node.js | 20+ | Via NVM recommandé |
| npm | 10+ | Inclus avec Node |
| Docker | 24+ | Pour les images sandbox |
| Python 3 | 3.10+ | Pour Wiki_LM |
| `gog-bin` | — | Binaire CLI Google, non versionnés — voir §4 |

```bash
node --version   # v20+
docker info      # doit répondre
python3 --version
```

### openclaw

openclaw **doit être installé avant** de lancer `install.sh`, via l'installeur CLI (sans onboarding) :

```bash
curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install-cli.sh | bash
openclaw --version
```

`install.sh` détecte le binaire automatiquement (PATH, versions NVM, ou prefix npm global).

---

## Architecture

```
Telegram
    │
    ▼
openclaw-gateway  (port 18789, service systemd user)
    │
    ├── agent main   (Euria — Mistral-Small-4 ou Qwen3.5-397B)
    │       │  workspace : ~/.openclaw/workspace/
    │       │  sandbox   : image Docker secretarius-tiron:latest
    │       │              monte ~/.openclaw/workspace/.gog-config → /workspace/.gog-config
    │       │
    │       ├── agent wiki   (Euria — Mistral-Small-4)
    │       │       workspace : ~/.openclaw/workspace-wiki/
    │       │       sandbox   : image Docker secretarius-wiki:latest
    │       │                   monte ~/Documents/Arbath/Wiki_LM → /Wiki_LM  (rw)
    │       │                   monte ~/Secretarius/Wiki_LM/tools → /wiki-tools  (ro)
    │       │                   monte ~/.openclaw/secrets/euria-key → /run/euria-key  (ro)
    │       │                   monte ~/Secretarius/Wiki_LM/zim → /zim  (ro) ← ZIM Wikipedia FR
    │       │
    │       ├── agent scout  (DeepSeek — deepseek-chat)
    │       │       workspace : ~/.openclaw/workspace-scout/
    │       │       sans sandbox Docker (outils read/write/process uniquement)
    │       │
    │       └── agent gog    (Euria — Mistral-Small-4)
    │               workspace : ~/.openclaw/workspace-gog/
    │               sandbox   : image Docker secretarius-gog:latest
    │                           monte ~/.openclaw/workspace/.gog-config → /gog-config  (rw)
    │
    └── plugin derisk-deleg
            fournit : gog_send, gog_inbox, gog_get, gog_search, gog_reply,
                      gog_drive_search, gog_connect_start,
                      wiki_capture, wiki_ingest, wiki_status, wiki_query,
                      wiki_kb_update, wiki_tags, source_read
            intercepte : /confirm, /annuler  (flux d'envoi email)
```

### Images Docker

| Image | Dockerfile | Rôle |
|-------|-----------|------|
| `secretarius-tiron:latest` | `openclaw-config/Dockerfile.tiron` | Sandbox agent main (gog CLI) |
| `secretarius-wiki:latest` | `openclaw-config/Dockerfile.wiki` | Sandbox agent wiki (outils Python Wiki_LM) |
| `secretarius-gog:latest` | `openclaw-config/Dockerfile.gog` | Sandbox agent gog (gog CLI) |

Build depuis `~/Secretarius/` :

```bash
docker build -f openclaw-config/Dockerfile.tiron -t secretarius-tiron:latest .
docker build -f openclaw-config/Dockerfile.wiki  -t secretarius-wiki:latest  .
docker build -f openclaw-config/Dockerfile.gog   -t secretarius-gog:latest   .
```

### Skills (commandes déterministes)

15 skills dans `~/.openclaw/workspace/skills/` — chacun a `command-dispatch: tool` (aucune décision LLM) :

| Commande | Outil plugin |
|----------|-------------|
| `/inbox` | `gog_inbox` |
| `/chercher` | `gog_search` |
| `/lire` | `gog_get` |
| `/repondre` | `gog_reply` |
| `/drive` | `gog_drive_search` |
| `/connecter` | `gog_connect_start` |
| `/c` | `wiki_capture` |
| `/ingest` | `wiki_ingest` |
| `/q` | `wiki_query` |
| `/wikistatus` | `wiki_status` |
| `/kbupdate` | `wiki_kb_update` |
| `/tags` | `wiki_tags` |
| `/source` | `source_read` |
| `/scout` | délégation agent scout |
| `/wiki-deleg` | délégation agent wiki |

---

## Variables d'environnement

Créer `~/.config/secrets.env` (sourcé automatiquement par `.bashrc` via `set -a`) :

```bash
# Bot Telegram de cette instance
TELEGRAM_BOT_TOKEN=<token>      # sanroque = bot dev, santiago = bot prod

# Euria (Infomaniak AI) — backend principal
EURIA_API_KEY=<clé 80 chars>    # vérifier longueur : echo -n "$EURIA_API_KEY" | wc -c
EURIA_PRODUCT_ID=109005

# DeepSeek — agent scout uniquement
DEEPSEEK_API_KEY=<clé>

# Google (gog CLI)
GOG_ACCOUNT=<email@gmail.com>
```

> **Piège** : sur sanroque, `secrets.env` est sourcé par `.bashrc` et ses valeurs ont
> priorité sur ce que `install.sh` tenterait d'écrire dans `gateway.systemd.env`.
> Mettre le bon token dans `secrets.env` directement.

---

## Procédure d'installation

### 1. Première installation

```bash
cd ~/Secretarius/openclaw-config
bash install.sh
```

`install.sh` :
- Génère `~/.openclaw/openclaw.json` depuis le template
- Génère `~/.openclaw/gateway.systemd.env` (token vide à compléter si secrets.env absent)
- Écrit `auth-profiles.json` pour chaque agent (main/wiki/gog/scout)
- Déploie `AGENTS.md` et `*/SKILL.md` dans les workspaces
- Active les services systemd (gateway + wiki-lm-server + wiki-lm-embed.timer)

### 2. Plugin derisk-deleg (manuel si NVM)

`openclaw plugins install .` échoue avec NVM (symlink cassé). Copie manuelle :

```bash
SRC=~/Secretarius/derisk-deleg
DST=~/.openclaw/extensions/derisk-deleg
mkdir -p "$DST"
cp -r "$SRC/dist" "$SRC/node_modules" "$SRC/openclaw.plugin.json" "$SRC/package.json" "$DST/"
```

Puis dans la Control UI, ouverte depuis un poste ayant accès réseau (direct, Tailscale ou tunnel SSH si headless). Le jeton de connexion (mode **jeton**, mot de passe vide) est dans `gateway.systemd.env` :

```bash
grep '^OPENCLAW_GATEWAY_TOKEN=' ~/.openclaw/gateway.systemd.env
```

- Aller dans **Plugins** → activer `derisk-deleg`
- Cocher **Hooks → allowConversationAccess**
- Redémarrer le gateway : `systemctl --user restart openclaw-gateway`

> **Piège --force** : `install.sh --force` régénère `openclaw.json` avec `plugins.entries:{}`,
> ce qui efface l'entrée du plugin. Après chaque `--force`, ré-activer le plugin dans l'UI.

> **Piège token gateway** : le gateway (systemd) s'authentifie via `OPENCLAW_GATEWAY_TOKEN`
> (env), mais le CLI et la Control UI lisent `gateway.auth.token` / `gateway.remote.token`
> dans `openclaw.json`. **Toute commande `openclaw` lancée sans `OPENCLAW_GATEWAY_TOKEN`
> dans l'environnement régénère un token aléatoire dans `openclaw.json`** et casse l'auth
> (« unauthorized: gateway token mismatch »). `install.sh` installe un *wrapper*
> `~/.local/bin/openclaw` qui charge ce token automatiquement — utilisez toujours cette
> commande (et non le binaire NVM direct). Le jeton à coller dans l'UI :
> `grep '^OPENCLAW_GATEWAY_TOKEN=' ~/.openclaw/gateway.systemd.env` (mode jeton, mot de passe vide).

### 3. Credentials Google (gog)

Les credentials vivent dans `~/.openclaw/workspace/.gog-config/gogcli/` :
- `credentials.json` — tokens OAuth
- `keyring/` — clé chiffrée
- `keyring-password` — mot de passe du keyring

Sur une **nouvelle machine**, les copier depuis une machine déjà authentifiée :

```bash
# Depuis la machine source
scp -r ~/.openclaw/workspace/.gog-config/gogcli/ user@cible:~/.openclaw/workspace/.gog-config/
```

Ou lancer l'authentification OAuth depuis zéro :

```bash
# Dans le conteneur gog (via une session agent gog)
/inbox  # déclenche gog_inbox → gog_connect_start si non authentifié
/connecter
```

### 4. Fichier ZIM Wikipedia FR

Le conteneur wiki monte `~/Secretarius/Wiki_LM/zim/` en lecture seule pour la lookup Wikipedia offline (`WIKI_ZIM_DIR=/zim`, `WIKI_LOOKUP_OFFLINE=1`). Le fichier ZIM doit être présent :

```bash
ls ~/Secretarius/Wiki_LM/zim/
# Doit contenir un fichier .zim (Wikipedia FR)
```

Si absent, le lookup Wikipedia est désactivé mais le reste du wiki fonctionne.

---

## Réinstallation propre

Préférer `uninstall` + `install` plutôt que `--force` (évite les deux pièges cités) :

```bash
cd ~/Secretarius/openclaw-config
bash uninstall.sh --yes
bash install.sh
# Puis refaire §2 (plugin) et vérifier §3 (credentials)
```

---

## Modèles disponibles

| Alias | ID complet | Agent | Notes |
|-------|-----------|-------|-------|
| `Euria` | `euria/mistralai/Mistral-Small-4-119B-2603` | main, wiki, gog | Défaut si Qwen397 indispo |
| `Qwen397` | `euria/Qwen/Qwen3.5-397B-A17B-FP8` | main | Recommandé : meilleur routage wiki |
| `Qwen122` | `euria/Qwen/Qwen3.5-122B-A10B-FP8` | main | Variante légère |
| `Gemma4` | `euria/google/gemma-4-31B-it` | main | — |
| `Nemotron3` | `euria/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | main | — |
| — | `deepseek/deepseek-chat` | scout | DEEPSEEK_API_KEY requis |

Changer le modèle de l'agent main :

```bash
switch-model Qwen397   # ou Euria, Qwen122, etc.
systemctl --user restart openclaw-gateway
```

> **Santiago** : `Qwen3.5-397B` est indisponible sur ce compte Euria (product_id 109005).
> L'agent main tourne sur Mistral-Small-4. Surveiller la qualité du routage `/c`→wiki.

---

## Vérification rapide

```bash
# Service actif ?
systemctl --user status openclaw-gateway

# Logs gateway (30 dernières lignes)
journalctl --user -u openclaw-gateway -n 30

# Plugin chargé ?
grep -i "derisk-deleg" ~/.openclaw/openclaw.json

# Auth-profiles des agents
for a in main wiki gog scout; do
  echo "=== $a ==="; cat ~/.openclaw/agents/$a/agent/auth-profiles.json 2>/dev/null || echo "(absent)"
done

# Clé Euria : doit faire 80 chars
echo -n "$EURIA_API_KEY" | wc -c
```

---

## Deux instances (sanroque vs santiago)

| | sanroque | santiago |
|---|----------|----------|
| Bot Telegram | `@secretarius_tiron_bot` (dev) | `@secretarius1789_bot` (prod) |
| Path | `~/.openclaw` | `~/.openclaw` |
| Port | 18789 | 18789 |
| Qwen397 | disponible | **indisponible** (fallback Mistral) |

Ne jamais faire tourner deux gateways pointant sur le même bot token → `Conflict: getUpdates`.
