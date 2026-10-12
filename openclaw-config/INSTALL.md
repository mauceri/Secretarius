# Installation de Secretarius

Une installation se fait depuis zéro, à partir du dépôt et d'un seul fichier de secrets,
selon un **profil** (spec `docs/superpowers/specs/2026-10-12-installateur-profils-design.md`).

| Profil | Machine type | Modules | Modèles |
|---|---|---|---|
| `leger` | petit VPS (santiago : 4 cœurs, 7,6 Go) | `wiki scout obfuscateur` | agent `main` et notes du wiki : Qwen3-14B obfusqué (Modal) ; pages web et Scout : DeepSeek. Ni Laya, ni modèle local |
| `lourd` | machine type sanroque (≥ 16 Go) | `wiki scout obfuscateur laya courrier cron gog` | agent `main` et wiki : Qwen3-8B local (Ollama) ; routeur Laya |

Un module absent du profil ne laisse aucune trace : ni agent, ni outil autorisé, ni service.
Sans Laya (profil léger), seules les commandes tapées (`/q …`, `/c …`) sont comprises.

## 1. Amorçage du système (une fois, avec sudo)

```bash
git clone https://github.com/mauceri/Secretarius.git ~/Secretarius   # emplacement imposé
cd ~/Secretarius
sudo ./amorcage-systeme.sh
```

Le script installe Git, Python, gettext, curl et Docker (et ajoute l'utilisateur au
groupe `docker`), 4 Go de swap s'il n'y en a pas, Tailscale, NVM avec Node LTS,
`openclaw` et `obsidian-headless`, et active `loginctl enable-linger` pour que les
services tournent sans session ouverte. Ensuite :

1. se déconnecter puis se reconnecter (groupe `docker`) ;
2. `sudo tailscale up --hostname=<nom> --ssh`, puis désactiver l'expiration de la clé de
   la machine dans la console Tailscale.

> **Ne pas installer openclaw avec `curl … install-cli.sh`** : il se place dans
> `~/.openclaw`, qui est le répertoire de configuration de Secretarius.

> **Disque.** L'image Docker `wiki` embarque BGE-M3 (~16 Go). Sur un VPS au disque juste,
> déplacer le `data-root` de Docker sur un volume **avant** l'installation (voir README).

## 2. Fichier de secrets et réglages de la machine

Un seul fichier, par exemple `~/.config/secrets.env` (droits `600`), passé par `--env-file` :

```bash
TELEGRAM_BOT_TOKEN=<jeton>        # propre à CETTE instance : deux instances sur le même
                                  # robot se volent les messages (Conflict: getUpdates)
TELEGRAM_OWNER_ID=<identifiant>   # votre identifiant Telegram numérique (propriétaire)
ALOEPRI_API_KEY=<clé>             # proxy obfusqué sur Modal (module obfuscateur)
DEEPSEEK_API_KEY=<clé>            # Scout et résumé des pages web
WIKI_ZIM_DIR=/chemin/des/zim      # facultatif, voir § 3
WIKI_VAULT_MIRRORS=               # facultatif : Nom=chemin,… (sous-wikis par coffre)
# Facultatifs : INFOMANIAK_API_KEY, INFOMANIAK_PRODUCT_ID, GOG_ACCOUNT (profil lourd)
```

Toute variable du fichier l'emporte sur le profil (`MODULES`, `MAIN_BRAIN`, `WIKI_LLM`,
`COFFRE`).

## 3. Copie locale de Wikipédia (fichiers ZIM)

Les fichiers ZIM sont **fournis par l'utilisateur** : leur taille, leurs langues et leur
variante sont des choix de machine. Catalogue Kiwix :
<https://download.kiwix.org/zim/wikipedia/>. Variante `mini` pour une petite machine
(fr : ~3,4 Go, en : ~12 Go), `nopic` sinon (fr : ~13 Go).

L'installateur cherche le dossier dans cet ordre et dit lequel a servi :

1. `WIKI_ZIM_DIR` (dossier absent ou sans `.zim` → **échec**) ;
2. `~/Secretarius/Wiki_LM/zim/` ;
3. en `--interactive`, il demande le chemin (réponse vide = aucun) ;
4. sinon aucun : l'installation continue, l'enrichissement par Wikipédia est désactivé.

La consultation reste toujours hors ligne : jamais de repli vers l'API publique.

## 4. Installation

```bash
cd ~/Secretarius
./install.sh --profil leger --env-file ~/.config/secrets.env
```

Relancer la même commande ne casse rien : les fichiers existants sont conservés
(`--force` pour les régénérer). Le coffre par défaut est `~/Documents/<coffre du profil>`
(`Publication` pour le léger, `Secretarius` pour le lourd), créé s'il manque.

L'installation se termine par une **vérification bloquante** de chaque service installé
(gateway, serveur wiki, proxy obfusqué, garde d'injection, Scout, synchronisation du
coffre…). Un service muet fait échouer l'installation, avec son nom.

## 5. Étapes de compte (une fois)

Le résumé final liste celles qui restent :

- **Telegram** : envoyer `/start` au robot, puis `openclaw pairing approve telegram <CODE>`.
- **Obsidian Sync** du coffre (double authentification possible) :
  `ob login`, `ob sync-create-remote --name <coffre>` si le coffre distant n'existe pas,
  `ob sync-setup --vault <coffre> --path ~/Documents/<coffre>`, puis relancer `install.sh`,
  qui active alors `obsidian-sync@<coffre>`.
- **Google** (profil lourd, module gog) : copier `~/.openclaw/workspace/.gog-config/gogcli/`
  depuis une machine authentifiée, ou lancer `/connecter`.

Après le dépôt d'un plugin Obsidian dans le coffre :
`systemctl --user restart obsidian-sync@<coffre>`.

## 6. Vérification et dépannage

```bash
systemctl --user status openclaw-gateway wiki-lm-server obfuscator-proxy
journalctl --user -u openclaw-gateway -n 30
curl -s http://127.0.0.1:5051/health        # serveur wiki
curl -s http://172.17.0.1:8001/health       # proxy obfusqué (premier appel lent : réveil de Modal)
```

> **Jeton de la gateway.** Toute commande `openclaw` lancée sans `OPENCLAW_GATEWAY_TOKEN`
> régénère un jeton dans `openclaw.json` et casse l'authentification. Utiliser l'enveloppe
> `~/.local/bin/openclaw` installée par `install.sh`, qui charge le jeton. Jeton de la
> Control UI (mode jeton, mot de passe vide) :
> `grep '^OPENCLAW_GATEWAY_TOKEN=' ~/.openclaw/gateway.systemd.env`.

## Migration d'une installation antérieure (sanroque)

Les installations d'avant les profils ont deux proxys obfusqués
(`obfuscator-proxy` sur 127.0.0.1:8001, `obfuscator-proxy-docker` sur 172.17.0.1:8001) et
une synchro Obsidian lancée à la main. `install.sh` s'arrête tant que
`obfuscator-proxy-docker.service` existe. Pour migrer :

1. `systemctl --user disable --now obfuscator-proxy obfuscator-proxy-docker`, puis
   supprimer leurs deux unités de `~/.config/systemd/user/` ;
2. dans `~/.openclaw/openclaw.json`, faire pointer `models.providers.obfusque.baseUrl` sur
   `http://172.17.0.1:8001/v1` ;
3. arrêter la synchro `ob sync --continuous` lancée dans tmux ;
4. ajouter `TELEGRAM_OWNER_ID` et `ALOEPRI_API_KEY` au fichier de secrets, puis relancer
   `./install.sh --profil lourd --env-file …` (sans `--force` : `Wiki_LM/.env`, ses miroirs
   de coffres et le modèle du wiki sont conservés ; seules les clés manquantes sont ajoutées).

## 7. Désinstallation

```bash
bash ~/Secretarius/openclaw-config/uninstall.sh --yes
```

Arrête et retire tous les services que les modules peuvent poser, et `~/.openclaw`.
Le coffre, `~/obfuscator` et les fichiers ZIM sont conservés.
