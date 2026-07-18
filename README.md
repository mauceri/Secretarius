# Secretarius `v0.4.0`

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

- **Node.js ≥ 22.22.3** (ou 24.x, ou 25.x) — via NVM. OpenClaw **refuse de démarrer**
  en dessous : le gateway boucle sur `Node.js >=22.22.3 … is required (current: vX)`.
- openclaw, installé dans votre Node (NVM) avec `npm install -g openclaw`.
  **Ne pas utiliser l'installeur `curl … install-cli.sh`** : il place openclaw dans
  `~/.openclaw`, qui est le répertoire de config de Secretarius → l'uninstall
  supprimerait openclaw avec.
  > **Piège Node + systemd — nvm ne suffit pas.** Le service gateway lance openclaw
  > via `#!/usr/bin/env node`, qui résout `node` par le **PATH système**
  > (`/usr/bin/node`) — **jamais** par nvm (nvm ne modifie que les shells interactifs,
  > pas les services systemd). Si le **Node système** est trop vieux, le gateway boucle
  > sur l'erreur `Node.js … required` **quoi que fasse nvm** (`nvm install/use/alias`
  > n'y changent rien). Vérifier le vrai coupable : `/usr/bin/node --version`.
  > Deux corrections fiables :
  > 1. **Mettre le Node système à niveau** (le plus propre) : installer Node ≥ 22.22.3
  >    en système (NodeSource : `curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash - && sudo apt install -y nodejs`), puis `/usr/bin/node --version` doit être valide.
  > 2. **Forcer un Node explicite dans le service** (sans toucher au système) via un
  >    drop-in — utile si le Node valide n'est qu'en nvm :
  >    ```bash
  >    NODE=~/.nvm/versions/node/v24.18.0/bin/node        # une version >= 22.22.3 installée
  >    OC=~/.nvm/versions/node/v24.18.0/bin/openclaw
  >    mkdir -p ~/.config/systemd/user/openclaw-gateway.service.d
  >    printf '[Service]\nExecStart=\nExecStart=%s %s gateway run\n' "$NODE" "$OC" \
  >      > ~/.config/systemd/user/openclaw-gateway.service.d/node.conf
  >    systemctl --user daemon-reload && systemctl --user restart openclaw-gateway
  >    ```
  >    (openclaw doit être installé sous ce Node : `nvm use 24 && npm install -g openclaw`.)
- Docker 24+ — avec **~40 Go d'espace libre** pour ses images (l'image `wiki`
  embarque BGE-M3 et pèse ~16 Go). `install.sh` construit l'image de base
  `openclaw-sandbox:bookworm-slim` si elle manque, puis les 3 images secretarius.
  > **VPS au disque juste.** Si `/` est trop petit, déplacez le data-root de
  > Docker sur un volume dédié **avant** l'install, sinon le build de `wiki`
  > remplit le disque :
  > ```bash
  > sudo systemctl stop docker docker.socket
  > sudo rsync -aP /var/lib/docker/ /mnt/<volume>/docker/
  > echo '{ "data-root": "/mnt/<volume>/docker" }' | sudo tee /etc/docker/daemon.json
  > sudo mv /var/lib/docker /var/lib/docker.old && sudo systemctl start docker
  > docker info | grep "Docker Root Dir"   # vérifier, puis: sudo rm -rf /var/lib/docker.old
  > ```
  > Vérifiez que le volume est dans `/etc/fstab` (option `nofail`) pour survivre au reboot.
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

> `install.sh` (étape 5) **construit désormais les trois images automatiquement**.
> Les commandes ci-dessous ne sont un repli que si l'install signale un échec de build
> (WARNING) — par exemple si Docker n'était pas encore accessible.

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

Répondre aux 4 questions (coffre Obsidian, nom de l'assistant, LLM, chemin OpenClaw). `install.sh` génère `~/.openclaw`, écrit un jeton de gateway cohérent, installe la commande `openclaw`, construit les 3 images, copie le plugin, installe le routeur `tiron-router`, puis démarre le gateway.

→ attendu : `Installation terminée` et `token gateway réconcilié`.

> **Cerveau distant (VPS sans GPU, ex. santiago).** Le « cerveau » de Tiron — le
> modèle qui alimente le provider `tiron-llm` et le routeur de commandes — est local
> par défaut (`http://127.0.0.1:8998`). Pour le servir ailleurs (Modal, ou le llama de
> sanroque via Tailscale), passez l'URL et la clé à l'install :
> ```bash
> TIRON_LLM_URL=https://<user>--tiron-llm-modal-serve.modal.run \
> TIRON_LLM_KEY=<clé Secret Modal> \
> BRAIN_MODAL_URL=https://<user>--tiron-llm-modal-serve.modal.run \
> ./install.sh --env-file ~/.config/secrets.env
> ```
> Voir `docs/components/modal.md` (déployer l'app Modal) et la section
> [Basculer le cerveau](#basculer-le-cerveau-de-tiron-local--modal).

**6. Copier le plugin derisk-deleg**

> `install.sh` (étape 5) **copie désormais le plugin automatiquement**. Les commandes
> ci-dessous ne sont un repli que si l'install signale un échec (WARNING).

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

> **Trouver l'URL Tailscale** (sur l'hôte) : `tailscale serve status` affiche directement l'URL si le gateway y est exposé (ex. `https://santiago.tailc69141.ts.net → http://127.0.0.1:18789`). Sinon, le nom complet de l'hôte (`<hôte>.<tailnet>.ts.net`) s'obtient par :
> ```bash
> tailscale status --json | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))"
> ```
> Si l'interface n'est pas encore exposée sur le tailnet d'une machine headless, l'y publier une fois : `tailscale serve --bg 18789`.

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

### Autoriser le compte (méthode recommandée)

`credentials.json` (OAuth client ID **Desktop**, créé dans Google Cloud Console) doit d'abord être présent dans `~/.openclaw/workspace/.gog-config/gogcli/`. Lancez ensuite l'autorisation **en shell**, via le script fourni — c'est plus fiable que la commande Telegram `/connecter` (qui passe par un pont FIFO fragile) :

```bash
cd ~/Secretarius
./gog-connect.sh cmauceri@gmail.com      # ou, si GOG_ACCOUNT est défini : ./gog-connect.sh
```

Déroulé : ouvrez l'URL Google affichée, connectez-vous et autorisez (Gmail + Drive + Agenda) ; Google redirige vers une URL `http://localhost:1/…` **qui ne charge pas** (normal) — copiez-la depuis la barre d'adresse et collez-la au prompt. Le token est écrit dans `.gog-config` ; l'agent gog le voit immédiatement (aucun redémarrage).

> **Un token par machine.** `--force-consent` donne à chaque machine son **propre** token. **Ne partagez pas un même `.gog-config` entre plusieurs machines en usage simultané** : Google fait tourner les refresh tokens, et les machines s'invalideraient mutuellement (« token révoqué/expiré »). Ré-autorisez chaque machine avec `./gog-connect.sh`.

### Réutiliser un `.gog-config` (dépannage d'une même machine)

Pour restaurer vite une instance après réinstallation **de la même machine**, le `.gog-config` est au même chemin et peut être recopié :

```bash
# Avant toute désinstallation, sauvegarder hors de ~/.openclaw :
cp -a ~/.openclaw/workspace/.gog-config ~/gog-config-backup
# Après réinstallation, restaurer :
mkdir -p ~/.openclaw/workspace/.gog-config
cp -a ~/gog-config-backup/. ~/.openclaw/workspace/.gog-config/
```

Aucun redémarrage du gateway n'est nécessaire (le conteneur gog monte ce répertoire à chaque exécution). Pour une **autre** machine, n'utilisez pas la copie : ré-autorisez avec `./gog-connect.sh` (token propre).

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

## Basculer le cerveau de Tiron (local ↔ Modal)

Distinct de `switch-model` (qui change le modèle **de conversation** de l'agent main) :
`switch-brain` change l'**endpoint du « cerveau »** — le modèle qui alimente le
provider `tiron-llm` et le routeur de commandes `tiron-router`. Utile quand une
machine n'a pas de GPU (VPS) et doit servir ce cerveau ailleurs.

Deux consommateurs sont repointés d'un coup (le provider dans `openclaw.json` **et**
le routeur via `~/.openclaw/tiron-router.env`), puis les services redémarrent :

```bash
./switch-brain.sh modal       # cerveau sur Modal
./switch-brain.sh sanroque    # cerveau = llama de sanroque (via Tailscale)
```

Les cibles nommées vivent dans **`~/.openclaw/brains.env`** (posé par `install.sh`,
éditable) :

```
BRAIN_SANROQUE_URL=http://100.100.126.7:8998        # IP Tailscale de sanroque
BRAIN_SANROQUE_KEY=
BRAIN_MODAL_URL=https://<user>--tiron-llm-modal-serve.modal.run
BRAIN_MODAL_KEY_FILE=~/.openclaw/secrets/tiron-llm-key
```

Prérequis pour la cible `modal` : l'app Modal déployée et la clé présente dans le
fichier `BRAIN_MODAL_KEY_FILE`. Déploiement de l'app, arrêt, coûts, chargement d'un
autre LLM : voir **`docs/components/modal.md`**.

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
├── install.sh, start.sh                # installation / démarrage (racine)
├── switch-brain.sh                     # bascule le cerveau LLM (local ↔ Modal)
├── gog-connect.sh                      # (ré)autorisation Google de l'agent gog (shell)
├── openclaw-config/
│   ├── INSTALL.md                      # Procédure d'installation détaillée
│   ├── install.sh                      # Génère ~/.openclaw/ via envsubst
│   ├── uninstall.sh                    # Désinstallation propre
│   ├── openclaw.json.template          # Configuration OpenClaw (4 agents + plugin)
│   ├── gateway.systemd.env.template
│   ├── openclaw-gateway.service        # Service systemd user
│   ├── tiron-router.service            # Service routeur de commandes (BGE-M3)
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
