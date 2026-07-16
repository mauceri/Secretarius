# Câblage Modal du cerveau Tiron + documentation composant

- Date : 2026-07-16
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude
- Précède : `2026-07-15-tiron-modal-deploiement-design.md` (mesure de faisabilité,
  exécutée : app `tiron_modal/`, mesures L4 dans son README)

## Objectif

Rendre le cerveau Modal de Tiron réellement utilisable — déploiement persistant,
endpoint protégé, bascule local↔Modal validée de bout en bout — puis documenter
le tout dans une fiche composant `docs/components/modal.md`.

C'est l'étape « câblage » que la spec du 2026-07-15 renvoyait à septembre,
avancée suite au succès de la mesure (63,8 tok/s génération sur L4).

## Décisions actées

| Sujet | Décision |
|-------|----------|
| Local | Reste l'état nominal ; `slm-llama_cpp.service` (8998) intouché ; Modal = alternative ponctuelle (démo, secours, VPS) |
| Bascule | Procédure manuelle documentée (pas de script, pas de second provider) |
| Cold start | Accepté : le 1er message après ~5 min d'inactivité peut échouer (503/timeout routeur 30 s) ; documenter « réessayer », curl de réveil mentionné comme astuce |
| Auth | `--api-key` de llama-server, clé dans un Secret Modal ; Bearer côté consommateurs |
| Installation | Variables `TIRON_LLM_URL`/`TIRON_LLM_KEY` pour pointer un cerveau distant dès l'installation (cas VPS sans GPU, ex. santiago) |
| Confidentialité | TLS + api-key suffisent pour transport et accès ; les prompts sont en clair chez Modal → note dans la doc + entrée backlog « confidentialité renforcée » |

## Modifications de code

Trois modifications, toutes petites :

1. **`tiron_modal/app.py`** : attacher un `modal.Secret` nommé `tiron-llm-api-key`
   (variable `LLAMA_API_KEY`) à la fonction `serve()` et ajouter
   `--api-key $LLAMA_API_KEY` à la ligne de lancement de `llama-server`.

2. **`router_service/server.py`** : si l'env `TIRON_LLAMA_KEY` est définie,
   ajouter l'en-tête `Authorization: Bearer <clé>` à la requête vers
   `TIRON_LLAMA_BASE`. Variable absente = comportement actuel inchangé
   (~3 lignes).

3. **Installation** (`install.conf`, `install.sh`, templates) :
   - `install.conf` : `TIRON_LLM_URL` (défaut `http://127.0.0.1:8998`) et
     `TIRON_LLM_KEY` (défaut vide), surchargeables par l'environnement.
   - `install.sh` : substitue ces valeurs dans les deux templates
     `openclaw.json*.template` (provider `tiron-llm` : `baseUrl` = URL + `/v1`,
     `apiKey` si clé non vide).
   - Défauts = comportement actuel ; une installation VPS passe
     `TIRON_LLM_URL=https://…modal.run` et la clé.
   - Constat : `tiron-router.service` n'est **pas** géré par `install.sh`
     (unité créée à la main sur sanroque, absente des installations VPS
     actuelles) → son édition reste dans la procédure de bascule manuelle,
     pas dans l'installation.

Piège de format : le provider OpenClaw attend l'URL **avec** `/v1`, le routeur
**sans** (il ajoute lui-même `/v1/chat/completions`).

## Procédure de bascule (à documenter, pas à coder)

Aller (local → Modal) :
1. `modal deploy tiron_modal/app.py` (si pas déjà déployé) ; noter l'URL.
2. `~/.openclaw/openclaw.json` : provider `tiron-llm` → `baseUrl` =
   `https://…modal.run/v1`, `apiKey` = clé du Secret.
3. `tiron-router.service` : `TIRON_LLAMA_BASE` = URL Modal (sans `/v1`),
   `TIRON_LLAMA_KEY` = clé ; `systemctl --user daemon-reload`.
4. Restart : gateway OpenClaw + `tiron-router.service`.

Attention : l'unité a `Requires=slm-llama_cpp.service` — ne pas stopper le
service local pendant que le cerveau est sur Modal, sinon le routeur tombe
aussi (à documenter dans la fiche).

Retour (Modal → local) : valeurs locales (`http://127.0.0.1:8998`, pas de clé),
mêmes restarts. Optionnel : `modal app stop tiron-llm-modal`.

## Vérification E2E (critères de succès)

1. `modal deploy` → verify : curl avec clé = complétion valide (200),
   curl sans clé = 401.
2. Bascule aller → verify : une commande Tiron réelle via Telegram (p. ex. `/q`)
   répond, et la requête apparaît dans `modal app logs`.
3. Bascule retour → verify : même commande OK via le service local 8998.

## Fiche `docs/components/modal.md`

Pattern des fiches existantes (frontmatter `tags`/`date`, Rôle, Prérequis,
Usage, Notes). Contenu :

- **Rôle** : cerveau Tiron de secours/démo sur GPU serverless (Modal, L4).
- **Prérequis** : compte Modal, CLI dans `~/modal-venv`, Volume `tiron-models`,
  Secret `tiron-llm-api-key`.
- **Démarrer / stopper** : `modal deploy` (persistant, URL stable),
  `modal app stop tiron-llm-modal`, `modal serve` (dev, éphémère),
  `modal app list` / `modal app logs`.
- **Basculer le cerveau de Tiron** : procédure aller/retour ci-dessus + le cas
  installation VPS (`TIRON_LLM_URL`/`TIRON_LLM_KEY`).
- **Charger un nouveau LLM** : `modal volume put tiron-models <gguf> /<gguf>`,
  ajuster `BASE`/`LORA`/`CTX`/`GPU` dans `app.py`, redéployer. Compatibilité
  version GGUF ↔ release llama.cpp de l'image : point de vigilance connu.
- **Coûts et comportement** : scale-to-zero après 5 min, cold start (503 pendant
  le chargement, 1er message peut échouer, astuce curl de réveil), tarif L4
  (~0,80 $/h facturé à l'usage), mesures du 2026-07-15 (252 tok/s prompt,
  63,8 tok/s génération, ~0,9 s/req à chaud).
- **Confidentialité** : TLS + api-key = transport et accès protégés ; les
  prompts sont traités en clair dans l'infrastructure Modal ; règle d'usage :
  pas de contenu sensible quand le cerveau est sur Modal.

## Backlog

Ajouter au backlog d'idées : « Confidentialité renforcée Secretarius↔Modal » —
étudier confidential computing / GPU TEE, ou filtrage-anonymisation des prompts
avant envoi. Aucune implémentation dans ce périmètre.

## Hors périmètre

- Script de bascule automatique, second provider `tiron-llm-modal`.
- `min_containers=1` (conteneur chaud permanent).
- vLLM / hot-swap LoRA.
- Toute mitigation de confidentialité au-delà de la note documentaire.

## Risques

- **Timeout routeur (30 s) vs cold start** : assumé (décision cold start).
- **Provider OpenClaw et `apiKey`** : vérifier que le provider
  `openai-completions` transmet bien `apiKey` en Bearer ; sinon, ajuster
  (champ `headers` ou équivalent) — à confirmer à l'implémentation.
- **santiago** : la substitution d'installation doit rester inerte pour les
  installations existantes (défauts = valeurs actuelles).
