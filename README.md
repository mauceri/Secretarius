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

```bash
# 1. Cloner
git clone https://github.com/mauceri/Secretarius
cd Secretarius

# 2. Variables d'environnement (lues automatiquement par install.sh)
#    Créer ~/.config/secrets.env avec :
#    TELEGRAM_BOT_TOKEN, EURIA_API_KEY, EURIA_PRODUCT_ID, DEEPSEEK_API_KEY, GOG_ACCOUNT

# 3. Builder les images Docker
docker build -f openclaw-config/Dockerfile.tiron -t secretarius-tiron:latest .
docker build -f openclaw-config/Dockerfile.wiki  -t secretarius-wiki:latest  .
docker build -f openclaw-config/Dockerfile.gog   -t secretarius-gog:latest   .

# 4. Installer
cd openclaw-config && bash install.sh

# 5. Installer le plugin derisk-deleg (copie manuelle si NVM — voir INSTALL.md §2)

# 6. Démarrer
systemctl --user start openclaw-gateway

# 7. Appairer Telegram
openclaw pairing approve telegram <CODE>
systemctl --user restart openclaw-gateway
```

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
