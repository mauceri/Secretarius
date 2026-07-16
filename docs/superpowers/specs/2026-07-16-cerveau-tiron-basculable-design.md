# Cerveau Tiron basculable (sanroque ↔ Modal) + install.sh turnkey

- Date : 2026-07-16
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Problème

Une installation santiago (VPS sans GPU) avec le cerveau de Tiron sur Modal
n'est pas turnkey. Le câblage Modal (chantier précédent, `TIRON_LLM_URL`/
`TIRON_LLM_KEY`) configure le provider `openclaw.json`, mais :
- le **routeur `tiron-router`** (BGE-M3 + classification), dont dépend chaque
  commande via `callRouter(8999)`, n'est **pas installé** par `install.sh` (unité
  manuelle, sanroque uniquement, avec `Requires=slm-llama_cpp.service` = serveur
  llama local absent sur santiago) ;
- les 3 **images sandbox** (`secretarius-gog/tiron/wiki`) ne sont pas construites ;
- la **copie du plugin `derisk-deleg`** est manuelle (install.sh l'imprime).

Besoin exprimé : pouvoir **basculer indifféremment le cerveau de Tiron sur
sanroque (via Tailscale) ou sur Modal**, à chaud, sur une install en route.

## Objectif

1. `install.sh` installe le **routeur** (versionné, sans dépendance llama locale,
   câblé au cerveau initial via `TIRON_LLM_URL/KEY`).
2. Une commande **`switch-brain <sanroque|modal>`** repointe le provider + le
   routeur vers l'endpoint choisi et redémarre, sans réinstaller.
3. `install.sh` devient turnkey : construit les 3 images sandbox, automatise la
   copie du plugin.

## Architecture — principe

Le routeur (BGE-M3, classification) tourne **toujours sur la machine**
(santiago ou sanroque), depuis le venv `Wiki_LM/.venv` (déjà torch 2.12 +
transformers 5.12 via `sentence-transformers`). Ce que l'on bascule, c'est
l'**endpoint LLM (phi-4)** que le routeur ET le provider interrogent :
- `sanroque` = `http://100.100.126.7:8998` (IP Tailscale de sanroque, pas de clé) ;
- `modal` = `https://<user>--tiron-llm-modal-serve.modal.run` + clé
  (`~/.openclaw/secrets/tiron-llm-key`).

## Décisions actées

| Sujet | Décision |
|-------|----------|
| Où tourne le routeur | Sur la machine, venv `Wiki_LM/.venv` (torch/transformers déjà présents) |
| Dépendance llama locale | **Retirée** de l'unité routeur (`Requires=slm-llama_cpp` supprimé) — santiago n'en a pas ; le llama est l'endpoint distant (`TIRON_LLAMA_BASE`) |
| Bascule | **Commande à chaud** `switch-brain` (script hôte, lancé en SSH) — pas depuis Telegram (édite systemd + openclaw.json, opération hôte) |
| Cible « sanroque » | Son llama joignable via **Tailscale** (`100.100.126.7:8998`) |
| Images sandbox | `install.sh` construit les 3 (gog/tiron/wiki), en douceur (WARNING si échec) |
| Copie plugin | `install.sh` l'automatise (aujourd'hui seulement imprimée) |

## Composants

### 1. Unité routeur versionnée — `openclaw-config/tiron-router.service`

```ini
[Unit]
Description=Tiron router service (BGE-M3 gate + dispatch)
After=network.target
[Service]
Type=simple
WorkingDirectory=%h/Secretarius
EnvironmentFile=%h/.openclaw/tiron-router.env
ExecStart=%h/Secretarius/Wiki_LM/.venv/bin/python -m router_service.server
Restart=on-failure
RestartSec=5
[Install]
WantedBy=default.target
```
Différences vs l'unité sanroque actuelle : **plus de `Requires=slm-llama_cpp`**,
python du **venv Wiki_LM** (pas `/usr/bin/python3`), `EnvironmentFile` dédié.

### 2. Fichier d'environnement routeur — `~/.openclaw/tiron-router.env`

Écrit par `install.sh` et par `switch-brain` :
```
TIRON_LLAMA_BASE=<url sans /v1>
TIRON_LLAMA_KEY=<clé ou vide>
WIKI_PATH=<chemin wiki>   # pour la FAQ (faits.md)
```

### 3. Registre de cerveaux — `~/.openclaw/brains.env`

Posé par `install.sh` (valeurs par défaut, éditable) :
```
BRAIN_SANROQUE_URL=http://100.100.126.7:8998
BRAIN_SANROQUE_KEY=
BRAIN_MODAL_URL=https://<user>--tiron-llm-modal-serve.modal.run
BRAIN_MODAL_KEY_FILE=~/.openclaw/secrets/tiron-llm-key
```

### 4. Commande `switch-brain` — `switch-brain.sh` (racine du dépôt)

`switch-brain <sanroque|modal>` :
1. lit l'URL + clé du cerveau cible depuis `brains.env` (clé Modal lue du fichier) ;
2. **provider** : `openclaw.json` → `tiron-llm.baseUrl = <url>/v1`, `apiKey = <clé｜"local">` ; puis `cp` vers `.bak` (anti-clobber) ;
3. **routeur** : réécrit `~/.openclaw/tiron-router.env` (`TIRON_LLAMA_BASE=<url>`, `TIRON_LLAMA_KEY=<clé>`) ;
4. `systemctl --user restart openclaw-gateway tiron-router` ;
5. affiche « Cerveau actif : <nom> (<url>) ».
Garde-fou : nom inconnu → usage + sortie 1, sans rien toucher.

### 5. Modifications `install.sh` / `openclaw-config/install.sh`

- Installer `tiron-router.service`, écrire `tiron-router.env` (depuis
  `TIRON_LLM_URL/KEY`) et `brains.env`, `daemon-reload` + enable + start.
- **Automatiser la copie du plugin** `derisk-deleg` (le bloc actuellement imprimé).
- **Construire les 3 images** `docker build -f openclaw-config/Dockerfile.{gog,tiron,wiki} -t secretarius-{gog,tiron,wiki}:latest` ; échec → WARNING + commande manuelle (pattern existant).

## Flux (bascule à chaud)

`switch-brain modal` → lit `brains.env` → `openclaw.json` provider repointé +
`.bak` → `tiron-router.env` réécrit → restart gateway + tiron-router → le routeur
recharge (BGE-M3, ~20 s) et interroge désormais Modal. `switch-brain sanroque` =
symétrique vers `100.100.126.7:8998`.

## Gestion d'erreur

- `switch-brain` : nom inconnu → usage, exit 1. Clé Modal absente alors que la
  cible est `modal` → erreur explicite, exit 1 (ne pas basculer avec une clé vide).
- `install.sh` : build image ou création venv en échec → **WARNING** non bloquant
  + commande manuelle (jamais d'arrêt brutal de l'install).

## Tests

- **`switch-brain` (unitaire, sans restart)** : sur des fichiers temporaires
  (`openclaw.json` + `tiron-router.env` + `brains.env` factices), vérifier que
  `switch-brain sanroque|modal` écrit les bonnes valeurs (baseUrl avec `/v1`,
  env sans `/v1`, clé résolue), et qu'un nom inconnu ne touche rien (exit 1).
  Mécanisme : `switch-brain` prend les chemins via variables d'environnement
  surchargeables (`OPENCLAW_JSON`, `ROUTER_ENV`, `BRAINS_ENV`) pour être testable
  hors système.
- **E2E** : sur une machine réelle, `switch-brain modal` → `/q` répond via Modal
  (logs Modal) ; `switch-brain sanroque` → `/q` répond via sanroque ; retour.

## Critères de succès

1. `install.sh` sur une machine neuve installe le routeur (venv Wiki_LM, câblé au
   cerveau initial), les 3 images, le plugin — sans étape manuelle pour ces points.
2. `switch-brain modal` puis `switch-brain sanroque` basculent le cerveau à chaud,
   provider **et** routeur repointés, `/q` fonctionne dans les deux cas.
3. Aucune régression sanroque (le cerveau local par défaut `127.0.0.1:8998` reste
   possible via `brains.env`/`TIRON_LLM_URL`).

## Hors périmètre

- `switch-brain` depuis Telegram (opération hôte ; extension future éventuelle).
- Déploiement/gestion de l'app Modal elle-même (couvert par `docs/components/modal.md`).
- Purge des conteneurs sandbox orphelins (nettoyage ponctuel séparé).
- Le service `slm-llama_cpp` (llama local sanroque) reste hors dépôt/hors install
  (spécifique sanroque ; santiago vise un endpoint distant).
- **Concurrence sur wiki partagé** : sanroque et santiago partagent **déjà** le même
  vault Obsidian synchronisé (un seul coffre). Le verrou d'ingestion
  (`.ingest_state.json`) est **local à chaque machine** et ne coordonne pas entre
  elles → deux ingestions simultanées sur les deux machines pourraient se marcher
  dessus. Géré par **discipline** (une seule ingestion à la fois, sur une seule
  machine) ; les lectures (`/q`) et captures (`/c`) restent sûres. **Non traité dans
  ce chantier** (accepté par l'utilisateur 2026-07-16).

## Point à confirmer au plan

- Sur santiago, `pip install sentence-transformers` peut tirer un **torch CUDA**
  inutile (pas de GPU) ; vérifier/forcer un torch CPU pour alléger le venv (index
  `--extra-index-url` CPU), sinon accepter le poids. Détail d'implémentation.
