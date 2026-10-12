# Secretarius `v0.4.0`

**Secretarius est une configuration sécurisée d'[OpenClaw](https://openclaw.dev) pour individus et petites structures**, enrichie d'une base de connaissances personnelle.

OpenClaw est un agent IA puissant, mais risqué dans sa configuration standard : il donne à l'IA un accès large à la machine hôte et reste vulnérable aux injections de prompt. Secretarius corrige cela par une architecture de sécurité en plusieurs couches, tout en ajoutant une mémoire documentaire durable.

Trois principes fondateurs :
- **Indépendance** : hébergé sur votre propre machine ou serveur.
- **Frugalité** : backend cloud léger (Infomaniak), pas de GPU requis.
- **Confidentialité** : vos données restent sous votre contrôle.

→ `docs/Secretarius.md` — présentation complète du projet  
→ `openclaw-config/INSTALL.md` — procédure d'installation détaillée

---

## Architecture

Quatre agents spécialisés, un plugin, des sandboxes Docker isolés :

```
Telegram ──► openclaw-gateway
                  │
                  ├── Tiron (main)   Infomaniak (Qwen3.5-397B ou Mistral-Small-4)
                  │       sandbox Docker secretarius-tiron
                  │       15 skills déterministes (/inbox /c /q /ingest …)
                  │
                  ├── wiki           Infomaniak (Mistral-Small-4)
                  │       sandbox Docker secretarius-wiki
                  │       outils Python Wiki_LM + ZIM Wikipedia FR offline
                  │
                  ├── scout          DeepSeek
                  │       fetch sécurisé de contenus externes (anti-injection)
                  │
                  └── gog            Infomaniak (Mistral-Small-4)
                          sandbox Docker secretarius-gog
                          CLI Google (email, agenda, drive)

Plugin derisk-deleg : fournit gog_* et wiki_* aux agents ; intercepte /confirm et /annuler
```

**Tiron** reçoit les demandes via Telegram. Il dispose de 15 commandes déterministes (aucune décision LLM : `/inbox`, `/q`, `/c`, `/ingest`, `/repondre`…) et délègue au sous-agent `wiki` pour les questions documentaires.

**Scout** reçoit les URLs à lire. Il fetche, nettoie, filtre le contenu avant de le transmettre — réduisant le risque d'injection de prompt indirecte.

**Wiki** gère la base de connaissances personnelle (capture, ingest, query). Il tourne dans un sandbox isolé avec accès en lecture seule aux fichiers ZIM Wikipedia FR pour la lookup offline.

**Gog** exécute les opérations Google dans son propre sandbox. L'envoi d'email nécessite une confirmation explicite (`/confirm`) — jamais exécuté silencieusement.

---

## Installation

Deux profils, un seul installateur (détails : `openclaw-config/INSTALL.md`) :

- **`leger`** (petit VPS, ex. santiago) : wiki, Scout, proxy obfusqué ; tous les modèles
  déportés (Qwen3-14B obfusqué sur Modal, DeepSeek pour le web), sans Laya ni modèle local ;
- **`lourd`** (machine type sanroque) : en plus Laya, courrier, cron, gog, Ollama local.

```bash
git clone https://github.com/mauceri/Secretarius.git ~/Secretarius   # emplacement imposé
cd ~/Secretarius
sudo ./amorcage-systeme.sh            # une fois : Docker, Node (NVM), openclaw, Tailscale, swap
# se reconnecter (groupe docker), puis :
./install.sh --profil leger --env-file ~/.config/secrets.env
```

L'installation se termine par une vérification bloquante des services, puis liste les
étapes de compte restantes (appairage Telegram, liaison Obsidian Sync). Le module `gog`
(profil lourd) exige le binaire `gog-bin` à la racine du dépôt (voir
[gogcli](https://github.com/openclaw/gogcli/releases)).

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

L'agent principal (`main`) utilise par défaut `Mistral-Small-4` (Infomaniak). Pour basculer sur un autre modèle :

```bash
switch-model Qwen397      # Qwen3.5-397B — meilleur routage wiki, recommandé
switch-model Infomaniak   # Mistral-Small-4 (défaut)
switch-model Qwen122      # Qwen3.5-122B — variante légère
systemctl --user restart openclaw-gateway
```

Modèles disponibles via Infomaniak :

| Alias | Modèle | Notes |
|-------|--------|-------|
| `Infomaniak` | Mistral-Small-4-119B-2603 | Défaut — fiable, rapide |
| `Qwen397` | Qwen3.5-397B-A17B-FP8 | Recommandé : meilleur routage `/c`→wiki |
| `Qwen122` | Qwen3.5-122B-A10B-FP8 | Variante légère de Qwen397 |
| `Gemma4` | google/gemma-4-31B-it | — |
| `Nemotron3` | nvidia/Nemotron-3-Nano-30B-A3B | — |

> **Note** : Qwen3.5-397B n'est pas disponible sur tous les comptes Infomaniak. Si l'agent ne répond pas après un `switch-model Qwen397`, revenir à `Infomaniak` (Mistral-Small-4).

L'agent `scout` utilise toujours DeepSeek (`deepseek-chat`) — non modifiable via `switch-model`.

---

## Cerveau de l'agent principal

Fixé par le profil (`MAIN_BRAIN`) : `modal` (Qwen3-14B obfusqué sur Modal, profil léger ;
clé de permutation et tokenizer restent locaux, cf. `~/obfuscator/docs/acces-openai.md`)
ou `ollama` (Qwen3-8B local, profil lourd). Infomaniak n'est jamais choisi
automatiquement ; `switch-model` permet d'y basculer à la main. Pour forcer un choix :

```bash
MAIN_BRAIN=ollama ./install.sh --profil lourd --env-file ~/.config/secrets.env --force
```

---

## Mise à jour

```bash
git pull
./install.sh --profil <leger|lourd> --env-file ~/.config/secrets.env
```

Sans `--force`, les fichiers existants (dont `openclaw.json`) sont conservés ;
`uninstall.sh --yes` puis `install.sh` repart de zéro.

---

## Structure du dépôt

```
Secretarius/
├── install.sh, start.sh                # installation / démarrage (racine)
├── gog-connect.sh                      # (ré)autorisation Google de l'agent gog (shell)
├── openclaw-config/
│   ├── INSTALL.md                      # Procédure d'installation détaillée
│   ├── install.sh                      # Génère ~/.openclaw/ via envsubst
│   ├── uninstall.sh                    # Désinstallation propre
│   ├── openclaw.json.template          # Configuration OpenClaw (4 agents + plugin)
│   ├── gateway.systemd.env.template
│   ├── openclaw-gateway.service        # Service systemd user
│   ├── tiron-router.service            # Service routeur de commandes (Laya)
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
