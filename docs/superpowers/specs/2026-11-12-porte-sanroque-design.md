# Porte Grok sur sanroque, isolée dans un conteneur — conception

Statut : à relire (2026-11-12). Suite de `2026-10-13-porte-grok-design.md` (porte
sur santiago). santiago est remis à zéro et hiberné (décision du 2026-11-06) ; tout
se fait désormais depuis sanroque, machine de travail de l'utilisateur. Publier la
porte par Funnel rend sanroque joignable depuis Internet : la porte doit donc être
isolée du reste de la machine.

## Décisions

| Question | Décision |
|---|---|
| Coffre vu par Grok | **Publication** (comme sur santiago), jamais Secretarius |
| Isolation | **Conteneur Docker** + règle de pare-feu `DOCKER-USER` |
| Modèle du wiki Publication | **qwen3:8b local via Ollama** (file partagée avec les `/q` de l'utilisateur) |
| Port Funnel | **10000** (443 pris par la route Tailscale du gateway, 8443 par le gateway) |
| Jetons | **Nouveaux** ; agent `orchestrateur`, seul agent sur santiago |

Pourquoi un conteneur plutôt qu'un utilisateur système : plusieurs services de
l'hôte écoutent sur `127.0.0.1` sans authentification, dont le wiki Secretarius
(5051). Un utilisateur dédié pourrait encore les joindre ; un conteneur sur un
réseau Docker ne voit pas le `127.0.0.1` de l'hôte.

## Architecture

```
Grok (cloud) ──HTTPS──> Funnel :10000 (tailscaled, sanroque)
                          └─> 127.0.0.1:5052 ──> conteneur « porte-grok »
                                                   ├─ porte (FastMCP, :5052)
                                                   └─ serveur wiki Publication (Flask, :5051 interne)
                                                        └─> Ollama hôte :11434 (qwen3:8b)
```

Un seul conteneur, deux processus. La porte appelle le wiki sur `127.0.0.1:5051`
**du conteneur** ; le 5051 de l'hôte (wiki Secretarius) lui est invisible.

### Image

- Base `ubuntu:24.04` + `python3.12` : `/usr/bin/python3.12`, la cible exacte des
  liens du `.venv` de l'hôte (Python 3.12.3).
- Aucune dépendance Python installée dans l'image : le `.venv` est monté au même
  chemin absolu. Un changement de code se prend par simple redémarrage.
- Point d'entrée : un script qui lance le serveur wiki puis la porte, et s'arrête
  si l'un des deux meurt (le redémarrage revient à systemd).

### Montages (rien d'autre)

| Hôte | Conteneur | Mode |
|---|---|---|
| `~/Documents/Publication` | `/coffre` | lecture-écriture |
| `~/Secretarius/Wiki_LM` (code, dont `.venv`) | même chemin | lecture seule |
| `~/.cache/huggingface` (`bge-m3`) | `/hf` | lecture seule |
| `~/.config/porte-grok/agents` | `/etc/porte/agents` | lecture seule |

Ni `~/.config/secrets.env`, ni `~/.ssh`, ni les coffres Secretarius et Arbath.
Les fichiers dérivés du wiki Publication vivent dans `/coffre/Wiki_LM/`
(`WIKI_PATH`), donc dans le montage du coffre.

Rien ne doit être écrit dans le code monté, sinon le montage en lecture seule
casse. Vu le 2026-11-12 : le cache BM25 et les embeddings suivent `WIKI_PATH`
(`tools/search.py`, `wiki_paths.embeddings_dir`). Le plan vérifie les autres
écritures (`wiki_cache.db`, journaux, `tags/`) sur les chemins réels.
`PYTHONDONTWRITEBYTECODE=1` évite les `__pycache__`.

### Durcissement

`--user 1000:1000` (les fichiers écrits appartiennent à l'utilisateur, `ob sync`
n'a pas de souci de droits), `--read-only` hors montages + `--tmpfs /tmp`,
`--cap-drop ALL`, `--security-opt no-new-privileges`, `HF_HUB_OFFLINE=1`.

### Réseau

- Réseau Docker dédié `porte-grok` (pont nommé `br-porte`).
- Port publié uniquement sur `127.0.0.1:5052`.
- Ollama joint par `host-gateway` sur 11434 (Ollama écoute déjà sur toutes les
  interfaces).
- Script versionné `openclaw-config/porte-pare-feu.sh`, lancé par l'utilisateur avec
  `sudo` : dans `DOCKER-USER`, depuis `br-porte`, accepter les connexions établies et
  `hôte:11434`, rejeter tout le reste (autres services de l'hôte en `0.0.0.0` :
  SSH, Samba, VNC, 4096, 9000, 9001, 9091, 19530 ; Internet). Grok étant dans le
  cloud, la porte n'a besoin d'aucune sortie Internet. Le script doit aussi poser la
  règle `INPUT` correspondante : le trafic d'un conteneur vers une IP de l'hôte passe
  par `INPUT`, pas par `FORWARD`/`DOCKER-USER`.
- La persistance de la règle après redémarrage (unité système ou
  `iptables-persistent`) est tranchée dans le plan, avec l'accord de l'utilisateur.

### Lancement et publication

- Unité systemd utilisateur `openclaw-config/porte-grok-conteneur.service` :
  `docker run --rm --name porte-grok …`. L'unité `porte-grok.service` actuelle reste
  celle du profil santiago ; l'installateur n'est pas modifié dans cette étape.
- Variables dans `~/.config/porte-grok/porte.env` (chmod 600) : `PORTE_COFFRE=/coffre`,
  `PORTE_AGENTS=/etc/porte/agents`, `PORTE_WIKI_URL=http://127.0.0.1:5051`,
  `WIKI_PATH=/coffre/Wiki_LM`, `WIKI_LLM_BACKEND=ollama`, adresse d'Ollama par
  `host-gateway`, `WIKI_VAULT_MIRRORS=` vide.
- Funnel : `tailscale funnel --bg --https=10000 http://127.0.0.1:5052`. L'utilisateur
  doit peut-être autoriser Funnel pour sanroque, une fois, dans la console Tailscale.
- Embeddings des nouvelles pages : timer utilisateur qui lance l'embedding par
  `docker exec porte-grok …`, même fréquence que `wiki-lm-embed` sur santiago.

### Données dérivées

Restaurées depuis `~/Documents-sauvegardes/santiago-20261106/santiago.tar.gz`
(embeddings, `.ingest_state.json`, `raw/.ingested`, `wiki_bm25_cache.json`,
`wiki_cache.db`) vers `~/Documents/Publication/`, sans écraser un fichier plus
récent déjà présent.

## Jetons et fiche

- Nouveau fichier `~/.config/porte-grok/agents` ; `porte-jeton ajouter orchestrateur`
  affiche le jeton une fois. L'utilisateur le copie avec la nouvelle adresse dans la
  configuration de Grok.
- `docs/grok/fiche-coordinateur.md` : remplacer santiago et 8443 par sanroque et
  `https://sanroque.<tailnet>.ts.net:10000/mcp`. Rien d'autre.

## Vérification

1. Restauration : nombre d'embeddings = nombre de pages du wiki Publication.
2. Conteneur lancé : `curl 127.0.0.1:5052/health` répond ; une `/r` par la porte
   renvoie des pages de Publication (et aucune de Secretarius).
3. Isolation, depuis le conteneur, **avant** puis **après** la règle de pare-feu :
   `secrets.env` illisible (absent) ; écriture dans le code refusée ; un fichier écrit
   dans `/coffre` appartient à l'uid 1000. Avant la règle, l'hôte en `0.0.0.0`
   (SSH) est joignable (preuve que le test voit ce que la règle bloquera) ; après,
   SSH, Samba, Internet échouent et Ollama passe. Dans les deux cas, le
   `127.0.0.1:5051` et Proton Bridge de l'hôte sont injoignables.
4. Funnel, depuis l'extérieur : sans jeton → 401 ; avec jeton → liste des outils.
5. Essai réel depuis Grok, par l'utilisateur.

Confirmation de l'utilisateur avant chaque `systemctl start/stop/enable/restart`,
`docker run`, commande Funnel et `git push`.

## Retour arrière

`systemctl --user stop porte-grok-conteneur` et `tailscale funnel --https=10000 off`
ferment tout ; le wiki Secretarius et le reste de sanroque ne sont pas touchés.
Le script de pare-feu sait aussi retirer ses règles (`porte-pare-feu.sh retirer`).

## Hors périmètre (suites)

- Exécution des blocs `/c` déposés par Grok dans Publication : plugin
  `wikilm-capture` à configurer sur buho pour ce coffre, pointé vers ce wiki.
- Intégration à l'installateur (profil « porte en conteneur »).
- Spec 4 (profil lourd sur VPS), à revoir avec la nouvelle stratégie.
