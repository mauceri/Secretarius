# Secretarius `v0.2.0`

**Secretarius est une configuration sécurisée d'[OpenClaw](https://openclaw.dev) pour individus et petites structures**, enrichie d'une base de connaissances personnelle.

OpenClaw est un agent IA puissant, mais risqué dans sa configuration standard : il donne à l'IA un accès large à la machine hôte et reste vulnérable aux injections de prompt. Secretarius corrige cela par une architecture de sécurité en plusieurs couches, tout en ajoutant une mémoire documentaire durable.

Trois principes fondateurs :
- **Indépendance** : hébergé sur votre propre machine ou serveur.
- **Frugalité** : backend cloud léger (Euria/Infomaniak), pas de GPU requis.
- **Confidentialité** : vos données restent sous votre contrôle.

→ `docs/Secretarius.md` — présentation complète du projet  
→ `openclaw-config/INSTALL.md` — procédure d'installation détaillée

---

## Architecture

Quatre agents spécialisés, un plugin, des sandboxes Docker isolés :

```
Telegram ──► openclaw-gateway
                  │
                  ├── Tiron (main)   Euria/Qwen3.5-397B ou Mistral-Small-4
                  │       sandbox Docker secretarius-tiron
                  │       15 skills déterministes (/inbox /c /q /ingest …)
                  │
                  ├── wiki           Euria/Mistral-Small-4
                  │       sandbox Docker secretarius-wiki
                  │       outils Python Wiki_LM + ZIM Wikipedia FR offline
                  │
                  ├── scout          DeepSeek
                  │       fetch sécurisé de contenus externes (anti-injection)
                  │
                  └── gog            Euria/Mistral-Small-4
                          sandbox Docker secretarius-gog
                          CLI Google (email, agenda, drive)

Plugin derisk-deleg : fournit gog_* et wiki_* aux agents ; intercepte /confirm et /annuler
```

**Tiron** reçoit les demandes via Telegram. Il dispose de 15 commandes déterministes (aucune décision LLM : `/inbox`, `/q`, `/c`, `/ingest`, `/repondre`…) et délègue au sous-agent `wiki` pour les questions documentaires.

**Scout** reçoit les URLs à lire. Il fetche, nettoie, filtre le contenu avant de le transmettre — réduisant le risque d'injection de prompt indirecte.

**Wiki** gère la base de connaissances personnelle (capture, ingest, query). Il tourne dans un sandbox isolé avec accès en lecture seule aux fichiers ZIM Wikipedia FR pour la lookup offline.

**Gog** exécute les opérations Google dans son propre sandbox. L'envoi d'email nécessite une confirmation explicite (`/confirm`) — jamais exécuté silencieusement.

---

## Prérequis

- Node.js 20+ via NVM — `nvm install 22`
- openclaw : `npm install -g openclaw@latest`
- Docker 24+
- Python 3.10+ (pour Wiki_LM)
- `envsubst` : `apt install gettext`
- Bot Telegram (token via [@BotFather](https://t.me/botfather))
- Clé API **Euria/Infomaniak** (backend principal)
- Clé API **DeepSeek** (agent scout)

---

## Installation rapide

Voir `openclaw-config/INSTALL.md` pour la procédure complète. En résumé :

**1. Cloner** :

```bash
git clone https://github.com/mauceri/Secretarius
cd Secretarius
```

**2. Builder les images Docker** :

```bash
docker build -f openclaw-config/Dockerfile.tiron -t secretarius-tiron:latest .
docker build -f openclaw-config/Dockerfile.wiki  -t secretarius-wiki:latest  .
docker build -f openclaw-config/Dockerfile.gog   -t secretarius-gog:latest   .
```

**3. Installer** :

```bash
cd openclaw-config && bash install.sh
```

**4. Renseigner les secrets** dans `~/.openclaw/gateway.systemd.env` :

```
TELEGRAM_BOT_TOKEN=<token BotFather>
EURIA_API_KEY=<clé Euria 80 chars>
EURIA_PRODUCT_ID=<identifiant produit Infomaniak>
DEEPSEEK_API_KEY=<clé DeepSeek — agent scout uniquement>
GOG_ACCOUNT=<adresse gmail>
```

> Si `~/.config/secrets.env` est en place et sourcé par `.bashrc`, `install.sh` l'a déjà lu — vérifier simplement que les valeurs sont correctes.

**5. Copier le plugin derisk-deleg** (`openclaw plugins install .` échoue avec NVM) :

```bash
SRC=~/Secretarius/derisk-deleg
DST=~/.openclaw/extensions/derisk-deleg
mkdir -p "$DST" && cp -r "$SRC/dist" "$SRC/node_modules" "$SRC/openclaw.plugin.json" "$SRC/package.json" "$DST/"
```

**6. Démarrer le gateway** :

```bash
systemctl --user start openclaw-gateway
```

**7. Activer le plugin** dans l'UI (`http://localhost:18789`) :
→ Plugins → activer `derisk-deleg` → cocher `hooks: allowConversationAccess` → Restart

```bash
# 8. Appairer Telegram : envoyer /start au bot, puis
openclaw pairing approve telegram <CODE>
systemctl --user restart openclaw-gateway
```

---

## Changer de modèle

L'agent principal (`main`) utilise par défaut `Mistral-Small-4` (Euria). Pour basculer sur un autre modèle :

```bash
switch-model Qwen397    # Qwen3.5-397B — meilleur routage wiki, recommandé
switch-model Euria      # Mistral-Small-4 (défaut)
switch-model Qwen122    # Qwen3.5-122B — variante légère
systemctl --user restart openclaw-gateway
```

Modèles disponibles via Euria/Infomaniak :

| Alias | Modèle | Notes |
|-------|--------|-------|
| `Euria` | Mistral-Small-4-119B-2603 | Défaut — fiable, rapide |
| `Qwen397` | Qwen3.5-397B-A17B-FP8 | Recommandé : meilleur routage `/c`→wiki |
| `Qwen122` | Qwen3.5-122B-A10B-FP8 | Variante légère de Qwen397 |
| `Gemma4` | google/gemma-4-31B-it | — |
| `Nemotron3` | nvidia/Nemotron-3-Nano-30B-A3B | — |

> **Note santiago** : Qwen3.5-397B est indisponible sur ce compte Euria (product_id 109005). L'agent main tourne sur Mistral-Small-4.

L'agent `scout` utilise toujours DeepSeek (`deepseek-chat`) — non modifiable via `switch-model`.

---

## Mise à jour

```bash
git pull
cd openclaw-config
bash uninstall.sh --yes
bash install.sh
# Ré-activer le plugin derisk-deleg dans l'UI gateway
```

> `uninstall` + `install` est plus sûr que `--force` (évite l'effacement de l'entrée plugin).

---

## Structure du dépôt

```
Secretarius/
├── openclaw-config/
│   ├── INSTALL.md                      # Procédure d'installation détaillée
│   ├── install.sh                      # Génère ~/.openclaw/ via envsubst
│   ├── uninstall.sh                    # Désinstallation propre
│   ├── openclaw-slm.json.template      # Configuration OpenClaw (4 agents + plugin)
│   ├── gateway-slm.systemd.env.template
│   ├── openclaw-gateway.service        # Service systemd user
│   ├── Dockerfile.tiron                # Sandbox agent principal
│   ├── Dockerfile.wiki                 # Sandbox agent wiki
│   ├── Dockerfile.gog                  # Sandbox agent gog
│   ├── workspace/                      # Workspace Tiron (AGENTS.md, SOUL.md, skills/)
│   │   └── skills/                     # 15 skills déterministes
│   ├── workspace-wiki/                 # Workspace agent wiki
│   ├── workspace-scout/                # Workspace agent scout
│   ├── workspace-gog/                  # Workspace agent gog
│   └── switch-model                    # Changer le modèle de l'agent main
│
├── derisk-deleg/                       # Plugin OpenClaw (gog_* + wiki_* + /confirm)
│   ├── src/index.ts
│   └── openclaw.plugin.json
│
├── Wiki_LM/
│   ├── tools/                          # Pipeline : capture, ingest, query, kb_*
│   ├── tests/                          # Suite pytest (170+ tests, zéro réseau)
│   └── zim/                            # Fichiers ZIM Wikipedia FR (non versionnés)
│
└── docs/
    ├── Secretarius.md                  # Présentation complète
    └── architecture/                   # Documents d'architecture
```

Les données wiki (`raw/`, `wiki/`, `knowledge_base/`) vivent dans `~/Documents/Arbath/Wiki_LM/` et ne sont pas versionnées.

---

## Sécurité

Secretarius atténue les risques d'OpenClaw par :

- **Sandboxes Docker par agent** : chaque agent est isolé dans son propre conteneur. Tiron n'a accès qu'à son workspace et au répertoire `.gog-config` monté explicitement.
- **Scout** : tout contenu externe passe par un agent isolé (sans accès réseau direct depuis les autres agents) avant d'atteindre Tiron.
- **Confirmation obligatoire pour les envois** : `gog_send` prépare un brouillon ; seul `/confirm` de l'utilisateur déclenche l'envoi réel.
- **Allowlist d'outils par agent** : chaque agent ne voit que les outils qui lui sont nécessaires.

Ces mesures réduisent fortement la surface d'attaque sans l'éliminer complètement. Voir `docs/Secretarius.md` pour une analyse détaillée.
