---
title: Installateur à profils (léger, lourd) et première installation de santiago
date: 2026-10-12
statut: à relire par l'utilisateur
---

# Installateur à profils et première installation de santiago

## Intention

La V1 de Secretarius est acquise quand une installation **depuis zéro**, à partir du seul
dépôt et d'un fichier de secrets, réussit sans retouche à la main, selon deux profils :

- **léger** : tous les modèles sont déportés (Modal obfusqué, DeepSeek ; Infomaniak
  possible mais non retenu, jugé cher pour peu de gain). Ni Laya, ni modèle local. Cible :
  santiago (Hetzner, 4 cœurs, 7,6 Go sans swap, Ubuntu 26.04, sans écran, sans Docker),
  peut-être un Raspberry plus tard ;
- **lourd** : l'équivalent de sanroque (Laya, Ollama, courrier, cron, gog), éprouvé plus tard
  sur un VPS loué le temps des essais.

L'instance légère de santiago est **totalement déconnectée** de celle de sanroque : son
wiki part de zéro et ne contiendra que des contenus destinés à être publiés. Elle servira
de banc d'essai à une équipe d'agents Grok Bot (publication sur YouTube, Substack,
LinkedIn, site personnel) et, selon l'idée de l'utilisateur, de produit d'appel qui
aidera à spécifier grandeur nature la V2 puis la V3. Genitrix et la fonction publication
relèvent de la V2.

## Découpage du chantier

1. **Installateur à profils** : objet de cette spécification.
2. **Profil léger appliqué à santiago**, coffre `Publication` synchronisé : son essai de
   bout en bout est le critère de réussite de cette spécification (§ Vérification).
3. **Porte publique Tailscale Funnel et fiche de l'agent coordinateur Grok** : spécification
   séparée. Décisions déjà prises : Funnel (et non un sous-domaine ou Cloudflare) ; les agents
   Grok peuvent lire, déposer des brouillons dans le coffre et capturer dans le wiki ;
   un service distinct de `wiki-lm-server`, à liste blanche, exposé sur un port Funnel
   (8443 ou 10000), le 443 restant au tailnet ; clé Tailscale de santiago exemptée
   d'expiration.
4. **Profil lourd sur un VPS de test** : plus tard, même installateur.

## Décisions

| Sujet | Décision |
|---|---|
| Organisation | un seul `install.sh`, des profils qui fixent des modules (pas deux installateurs, pas Ansible) |
| Prérequis système | vérifiés seulement par `install.sh` ; installés par un script séparé `amorcage-systeme.sh` lancé une fois avec `sudo` |
| Routeur Laya | absent du profil léger ; derisk-deleg reconnaît lui-même les commandes tapées |
| Cerveau de l'agent `main` (léger) | Qwen3-14B obfusqué sur Modal |
| Ingestion du wiki (léger) | notes et textes : Qwen3-14B obfusqué (choix de démonstration) ; pages web : DeepSeek, par la branche existante `WIKI_INGEST_WEB_LLM_*` (`Wiki_LM/tools/wiki.py:404`) |
| Scout (léger) | inclus, garde d'injection compris ; l'agent `scout` reste sur DeepSeek |
| Courrier (léger) | prévu par l'installateur, non activé ; plus tard avec un compte Proton distinct (abonnement Business disponible) |
| Cron, gog (léger) | exclus ; la planification reste chez Grok Bot |
| Emplacement du dépôt | `~/Secretarius`, imposé et vérifié par les prérequis (les unités et scripts y font référence en dur ; le rendre libre n'apporte rien à la V1) |

## Architecture de l'installateur

```
amorcage-systeme.sh   (sudo, une fois)   Docker, Node (NVM), OpenClaw, Tailscale, swap
install.sh --profil leger|lourd --env-file <secrets>
  1. prérequis      vérifie, s'arrête au premier manquant avec la commande à lancer
  2. socle commun   venv, openclaw.json (gabarit), agent main, derisk-deleg,
                    robot Telegram, coffre, Wiki_LM/.env, obsidian-sync@<coffre>
  3. modules        module_<nom> pour chaque nom de $MODULES, dans l'ordre
  4. vérification   interroge chaque service installé ; échec bruyant au premier muet
```

### Profils

`profils/leger.conf` et `profils/lourd.conf`, sourcés après `install.conf` et avant
l'environnement : une variable d'environnement explicite garde la priorité.

| Variable | léger | lourd |
|---|---|---|
| `MODULES` | `wiki scout obfuscateur` | `wiki scout obfuscateur laya courrier cron gog` |
| `MAIN_BRAIN` | `modal` | `ollama` (ou auto selon la mémoire, comme aujourd'hui) |
| `WIKI_LLM` | `obfusque` | `local` |
| `COFFRE` | `Publication` | `Secretarius` |

L'ajout futur du courrier au profil léger tient dans la ligne `MODULES`.

### Modules

Chaque `module_<nom>` installe ses unités systemd, ses images Docker, son espace de travail
et **ses entrées dans `openclaw.json`** : agent, outils permis, outils refusés aux autres
agents. Règle : **un module absent ne laisse aucune trace**, ni agent déclaré, ni outil
autorisé, ni unité installée.

| Module | Contenu |
|---|---|
| `obfuscateur` | clone `~/obfuscator`, régénère la clé de permutation (déterministe), installe **un seul** proxy Qwen3-14B joignable de l'hôte et du pont Docker (sur sanroque deux instances, ~960 Mo chacune, pour le même modèle) ; fournisseur `obfusque` dans `openclaw.json` |
| `wiki` | `wiki-lm-server`, image `secretarius-wiki`, agent `wiki`, minuteur d'embeddings ; outils `wiki_*` |
| `scout` | garde d'injection, `scout-watcher`, agent `scout` (DeepSeek) |
| `laya` | `tiron-router`, FAQ de faits (`amorçage/faits.md`) |
| `courrier` | `tiron_mail`, relève, Proton Bridge, agent `consignes`, outils `mail_*` |
| `cron` | `tiron_cron`, webhook de livraison |
| `gog` | image `secretarius-gog` (échec de construction **bloquant**), agent `gog`, `/connecter` |

### Socle commun : ce qui change

- **derisk-deleg reconnaît les commandes tapées.** Si le message commence par une commande
  connue (`commandToAction` non nul), il l'exécute sans appeler le routeur, avec le contrôle
  d'argument manquant aujourd'hui fait par `router_service/server.py` (`NEEDS_ARG`,
  réponse « Usage : <cmd> <argument> »). Si le routeur n'est pas installé, un texte libre
  reçoit « Tapez une commande (voir /help). » au lieu de « Routeur local indisponible ».
  Le comportement vaut aussi pour le profil lourd : une commande tapée marche même routeur
  en panne. La présence du routeur est une option de configuration du plugin, écrite par
  le module `laya`.
- **Synchronisation du coffre** : unité `obsidian-sync@.service` (instance = nom du coffre)
  à la place de `ob sync --continuous` lancé à la main dans tmux. Une synchronisation
  continue déjà lancée ne détecte pas toujours un plugin déposé dans `.obsidian`
  (`docs/components/obsidian.md`, « Pièges connus ») ; d'où :
  - l'unité commence par une synchronisation ponctuelle
    (`ExecStartPre=ob sync --path <coffre>`), puis passe en continu
    (`ExecStart=ob sync --continuous --path <coffre>`) ;
  - « couper et relancer » devient `systemctl --user restart obsidian-sync@<coffre>`, que
    tout script de déploiement d'un plugin Obsidian lance lui-même à la fin.
- **Plus d'étape manuelle d'après installation** : l'entrée `derisk-deleg` (activée,
  `hooks.allowConversationAccess`) figure déjà dans le gabarit ; le message d'après
  installation qui demande de l'activer dans l'interface est à vérifier puis retirer, ainsi
  que tout consentement de plugin requis par OpenClaw 2026.8 (à écrire dans la
  configuration).
- **Vérification finale** bloquante, en remplacement des avertissements non bloquants.

### Écarts de reproductibilité corrigés

Recensés dans le guide de reprise (`Secretarius_dev/reprise-secretarius.md`), chacun
corrigé dans le module concerné :

- copie locale de Wikipédia : chemin des ZIM posé par l'installation (variable du module
  `wiki`) ; sur santiago, le volume Hetzner qui porte déjà les ZIM en et fr. Sans ZIM,
  le mode hors ligne reste forcé : jamais de repli silencieux vers l'API publique ;
- `WIKI_VAULT_MIRRORS` posée par l'installation (vide par défaut) ;
- `_DEFAULT_KB_DIR` (`Wiki_LM/tools/kb_update.py`) dérivé de `WIKI_PATH` au lieu d'Arbath ;
- image gog : échec de construction bloquant, obtention du binaire documentée ;
- services absents de l'installation actuelle : proxy obfuscateur dans tous les profils,
  relève du courrier, Proton Bridge, enveloppe `~/.local/bin/openclaw`, synchronisation
  du coffre ;
- DeepSeek n'est plus le défaut de `install.conf` ni de `Wiki_LM/.env` : ce sont les
  profils qui fixent les modèles.

## Budget de mémoire du profil léger

Estimé d'après les mesures de sanroque : gateway ~550 Mo, proxy obfuscateur ~960 Mo,
garde d'injection ~650 Mo, `wiki-lm-server` ~250 Mo au repos. Soit 2,5 à 3 Go au repos,
4,5 à 5 Go quand BGE-M3 (recherche sémantique, chargé à la première requête) est en
mémoire et qu'un conteneur tourne. Le mode hybride est conservé ; la mémoire sera mesurée
sur santiago avant toute autre décision. `amorcage-systeme.sh` ajoute 4 Go de swap.

## Vérification

### Tests automatiques (machine de développement)

- les suites existantes : Python (Wiki_LM, tiron_mail, tiron_cron, router_service) et les
  tests TypeScript de derisk-deleg ;
- derisk-deleg : commande tapée exécutée sans appel au routeur ; argument manquant ; texte
  libre sans routeur installé ; texte libre avec routeur (inchangé) ;
- installateur : pour chaque profil, génération de `openclaw.json` dans un répertoire
  temporaire, puis contrôle que :
  - un module absent ne laisse ni agent ni outil autorisé (ex. aucun `mail_*`, aucun agent
    `gog` ni `consignes` dans le profil léger) ;
  - aucune valeur propre à sanroque (chemin `/home/mauceric`, coffre `Arbath`) ne subsiste.

Pas de simulation complète en machine virtuelle ou en conteneur : trop de systemd et de
Docker dans Docker pour un gain modeste. santiago, puis le VPS de test, en tiennent lieu.

### Essai de bout en bout (critère de réussite)

1. **Léger sur santiago, depuis zéro** : `amorcage-systeme.sh`, puis
   `install.sh --profil leger --env-file secrets.env`. La vérification finale passe. Sur le
   second robot Telegram : `/wikistatus`, `/c <texte>`, `/ingest`, `/q` aboutissent ;
   `/source <url>` passe par Scout ; la page produite apparaît dans le coffre `Publication`
   sur la tablette de l'utilisateur.
2. **Idempotence** : relancer `install.sh` ne modifie rien et ne casse rien.
3. **Désinstallation** : `uninstall.sh`, adapté aux profils, laisse la machine propre.
4. **Lourd sur VPS de test** : morceau 4, hors de cette spécification.

## Hors périmètre

- Porte Funnel et fiche du coordinateur Grok (morceau 3).
- Profil lourd sur VPS (morceau 4), Raspberry (images ARM à vérifier le moment venu).
- Courrier sur santiago, cron, gog sur le profil léger.
- Isolation des réponses du wiki par coffre : sans objet, le wiki de santiago ne contient
  que du publiable.
