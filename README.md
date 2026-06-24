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

- openclaw, installé dans votre Node (NVM) avec `npm install -g openclaw`.
  **Ne pas utiliser l'installeur `curl … install-cli.sh`** : il place openclaw dans
  `~/.openclaw`, qui est le répertoire de config de Secretarius → l'uninstall
  supprimerait openclaw avec.
- Docker 24+
- Python 3.10+ (pour Wiki_LM)
- `envsubst` : `apt install gettext`
- `gog-bin` : binaire [gogcli](https://gogcli.sh/) (CLI Google Workspace) — télécharger le binaire Linux depuis [github.com/openclaw/gogcli](https://github.com/openclaw/gogcli/releases), renommer en `gog-bin` et placer à la racine du dépôt ; ou copier depuis une machine déjà configurée via `scp`
- Bot Telegram (token via [@BotFather](https://t.me/botfather))
- Clé API **Euria/Infomaniak** (backend principal)
- Clé API **DeepSeek** (agent scout)

---

## Installation

Procédure complète pour une première installation. Chaque étape indique le résultat attendu. (Pour mettre à jour une installation existante, voir « Mise à jour ».)

**1. Cloner le dépôt**

```bash
git clone https://github.com/mauceri/Secretarius
cd Secretarius
```

**2. Installer `gog-bin`**

Télécharger le binaire Linux de [gogcli](https://github.com/openclaw/gogcli/releases), le renommer `gog-bin`, le placer à la racine du dépôt (ou le copier via `scp` depuis une machine déjà configurée) :

```bash
ls -l gog-bin        # le fichier doit exister
```

**3. Créer le fichier de secrets** `~/.config/secrets.env`

```
TELEGRAM_BOT_TOKEN=<token BotFather>
EURIA_API_KEY=<clé Euria, 80 caractères>
EURIA_PRODUCT_ID=<identifiant produit Infomaniak>
DEEPSEEK_API_KEY=<clé DeepSeek — agent scout>
GOG_ACCOUNT=<adresse gmail>
```

**4. Construire les images Docker**

```bash
docker build -f openclaw-config/Dockerfile.tiron -t secretarius-tiron:latest .
docker build -f openclaw-config/Dockerfile.wiki  -t secretarius-wiki:latest  .
docker build -f openclaw-config/Dockerfile.gog   -t secretarius-gog:latest   .
```

→ attendu : `naming to ... secretarius-*:latest` pour chacune.

**5. Lancer l'installation**

```bash
./install.sh --env-file ~/.config/secrets.env
```

Répondre aux 4 questions (coffre Obsidian, nom de l'assistant, LLM, chemin OpenClaw). `install.sh` génère `~/.openclaw`, écrit un jeton de gateway cohérent, installe la commande `openclaw`, puis démarre le gateway.

→ attendu : `Installation terminée` et `token gateway réconcilié`.

**6. Copier le plugin derisk-deleg**

`openclaw plugins install .` échoue avec NVM ; on copie les fichiers (déjà construits dans le dépôt) :

```bash
SRC=~/Secretarius/derisk-deleg
DST=~/.openclaw/extensions/derisk-deleg
mkdir -p "$DST"
cp -r "$SRC/dist" "$SRC/node_modules" "$SRC/openclaw.plugin.json" "$SRC/package.json" "$DST/"
```

**7. Redémarrer pour charger le plugin**

```bash
./start.sh
```

→ attendu : `openclaw-gateway démarré`.

**8. Activer le plugin dans l'interface web**

Afficher le jeton de connexion :

```bash
grep '^OPENCLAW_GATEWAY_TOKEN=' ~/.openclaw/gateway.systemd.env
```

Ouvrir l'interface depuis un poste ayant accès réseau à la machine — directement (poste de bureau), via Tailscale (`https://<hôte>.<tailnet>.ts.net`) ou par tunnel SSH si la machine est sans écran. Mode **jeton** : coller la valeur affichée, **laisser le champ mot de passe vide**, Connecter.

À la **première connexion d'un navigateur**, l'interface affiche « Appairage de l'appareil requis » avec un identifiant. C'est normal (approbation unique). Approuvez-le sur l'hôte, puis reconnectez-vous :

```bash
openclaw devices approve <ID-affiché>
```

Puis dans l'interface : **Plugins** → activer `derisk-deleg` → cocher `allowConversationAccess` → **Restart**.

**9. Appairer Telegram**

Envoyer `/start` au bot, puis :

```bash
openclaw pairing approve telegram <CODE>
./start.sh
```

**10. Identifiants Google (gog)**

Pour que `/inbox` et l'envoi de mail fonctionnent, l'agent gog doit être authentifié. **Si gog est déjà installé et autorisé sur une autre machine Secretarius, ses identifiants sont directement réutilisables** (même chemin, refresh token inclus) — copiez-les vers cette machine :

```bash
scp -r <utilisateur>@<machine-source>:~/.openclaw/workspace/.gog-config/. \
       ~/.openclaw/workspace/.gog-config/
```

Sinon (première authentification depuis zéro), voir la section [Identifiants Google (gog)](#identifiants-google-gog) plus bas.

**11. Tester**

Sur Telegram : `/inbox`, puis un envoi de mail suivi de `/confirm`.

---

## Le jeton du gateway (à lire absolument)

Le gateway exige un jeton d'authentification. Trois règles, sinon l'interface web affiche « L'authentification ne correspond pas » :

1. **Le jeton à coller dans l'interface** est toujours celui-ci, et lui seul :
   ```bash
   grep '^OPENCLAW_GATEWAY_TOKEN=' ~/.openclaw/gateway.systemd.env
   ```
   Mode **jeton**, mot de passe **vide**.

2. **Utilisez toujours la commande `openclaw` telle quelle.** `install.sh` en fait un script (`~/.local/bin/openclaw`) qui charge ce jeton automatiquement. **Ne lancez jamais le binaire par son chemin complet** (`~/.nvm/.../openclaw`) : lancé sans son jeton, il en réécrit un nouveau, aléatoire, dans `openclaw.json`, et l'authentification cesse de fonctionner.

3. Si l'authentification a déjà été cassée (jeton réécrit), il suffit de relancer `./install.sh --env-file ~/.config/secrets.env` puis `./start.sh` : l'installation réaligne le jeton partout.

---

## Identifiants Google (gog)

L'agent gog s'authentifie via des identifiants stockés dans `~/.openclaw/workspace/.gog-config/` :

```
.gog-config/
├── keyring-password                       # mot de passe du keyring
└── gogcli/
    ├── credentials.json                   # OAuth client ID (application Desktop)
    └── keyring/
        ├── token:<compte>@gmail.com       # refresh token (l'autorisation effective)
        └── token:default:<compte>@gmail.com
```

Ces fichiers **ne sont pas versionnés** et sont perdus si `~/.openclaw` est supprimé (désinstallation).

**Si gog est déjà installé et autorisé sur une autre machine** (une installation Secretarius existante), **ses identifiants sont directement réutilisables** : ils se trouvent **au même chemin**, `~/.openclaw/workspace/.gog-config/`. Copiez ce répertoire vers la nouvelle machine plutôt que de refaire le flux OAuth — le compte Google y est déjà autorisé, le refresh token est inclus. (Note : `~/.openclaw` n'étant lisible que par son propriétaire, copiez-le pendant que l'instance source est installée, ou depuis une sauvegarde que vous en avez faite.)

```bash
# Depuis une machine source vers la cible (le point final copie aussi les fichiers cachés) :
scp -r <utilisateur>@<machine-source>:~/.openclaw/workspace/.gog-config/. \
       ~/.openclaw/workspace/.gog-config/

# Ou, si la sauvegarde est locale (ex. ~/gog-config-backup) :
mkdir -p ~/.openclaw/workspace/.gog-config
cp -a ~/gog-config-backup/. ~/.openclaw/workspace/.gog-config/
```

> **Conseil** : avant toute désinstallation, sauvegardez ce répertoire hors de `~/.openclaw` :
> `cp -a ~/.openclaw/workspace/.gog-config ~/gog-config-backup`

La cible doit contenir exactement l'arborescence ci-dessus. Aucun redémarrage du gateway n'est nécessaire (le conteneur gog monte ce répertoire à chaque exécution).

**Sinon (première authentification)** : placer `credentials.json` (OAuth client ID Desktop, depuis Google Cloud Console) dans `~/.openclaw/workspace/.gog-config/gogcli/`, puis lancer le flux d'autorisation gog (ouvre un navigateur pour autoriser le compte).

---

## Démarrage quotidien

Après une installation réussie, `start.sh` est le point d'entrée unique :

```bash
cd ~/Secretarius
./start.sh
```

Il vérifie le binaire openclaw, contrôle que `TELEGRAM_BOT_TOKEN` est renseigné, redémarre `openclaw-gateway` et `wiki-lm-server` (si activé).

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

> **Note** : Qwen3.5-397B n'est pas disponible sur tous les comptes Euria. Si l'agent ne répond pas après un `switch-model Qwen397`, revenir à `Euria` (Mistral-Small-4).

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
│   ├── openclaw.json.template          # Configuration OpenClaw (4 agents + plugin)
│   ├── gateway.systemd.env.template
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
