# Recouvrement fonctionnel prod → SLM — design

Date : 2026-06-18
Branche cible : travail SLM (archi par intention)

## Objectif

Porter sur la version SLM (commandes déterministes) les fonctionnalités de la
version prod (outils MCP v0.1.0) encore manquantes, afin que le SLM puisse
remplacer la prod. Ce lot couvre les **6 fonctionnalités débloquées** (sans
re-consentement OAuth), précédées d'une **refonte d'image** isolant gog dans une
image dédiée `secretarius-gog` (étape 0). **Calendar** est hors périmètre (scope non consenti, 403
vérifié 2026-06-18). **Drive en lecture** s'est révélé déjà consenti (recherche
testée OK 2026-06-18) → `drive_search` intégré ; drive download/upload restent
hors lot (transfert de fichier à concevoir, scope write non vérifié).

## Périmètre — 6 commandes

| Commande | Outil plugin | Agent · op | Nature |
|----------|--------------|-----------|--------|
| `/chercher <requête>` | `gog_search` | gog · `search` (op déjà mappée) | lecture |
| `/lire <id>` | `gog_get` | gog · `get` (nouvelle) | lecture, contenu non fiable |
| `/drive <requête>` | `gog_drive_search` | gog · `drive_search` (nouvelle) | lecture, contenu non fiable |
| `/repondre <id> <texte>` | `gog_reply` | gog · `reply` (nouvelle) | écriture → brouillon + /confirm |
| `/tags` | `wiki_tags` | wiki · `tags` (nouvelle CLI) | lecture |
| `/kbupdate` | `wiki_kb_update` | wiki · `kb_update` (nouvelle CLI) | écriture → async background |

Noms de commandes validés avec l'utilisateur (2026-06-18).

## Étape 0 — image dédiée `secretarius-gog`

Refonte de sécurité (moindre privilège) à faire **avant** les commandes gog.
Aujourd'hui l'agent `gog` **et** `main` partagent `secretarius-tiron:latest`, qui
n'ajoute à la base que le binaire gog (`Dockerfile.tiron` = base + `gog-bin` +
`gog-wrapper`). Résultat : le conteneur de `main` embarque inutilement gog.

- Créer `openclaw-config/Dockerfile.gog` = contenu actuel de `Dockerfile.tiron`
  (`FROM openclaw-sandbox:bookworm-slim` + COPY `gog-bin` + COPY `gog-wrapper.sh`).
  Build → image `secretarius-gog:latest`.
- Réduire `Dockerfile.tiron` à `FROM openclaw-sandbox:bookworm-slim` (base nue,
  nom conservé pour évolution future de `main`). Rebuild `secretarius-tiron:latest`.
- Basculer l'agent `gog` dans `openclaw.json` : image `secretarius-tiron:latest`
  → `secretarius-gog:latest`. `main` reste sur `secretarius-tiron`. Restart gateway.
- Vérifier : l'agent gog exécute toujours `gog` (creds montés, `/inbox` OK) ;
  le conteneur de `main` n'a plus le binaire gog.

Aucun impact fonctionnel attendu : `main` délègue déjà à l'agent gog, il n'exécute
jamais gog lui-même.

## Patron commun (déjà prouvé)

skill `command-dispatch:tool` (`disable-model-invocation`, `command-arg-mode:raw`)
→ outil plugin (`derisk-deleg`) → `api.runtime.subagent.run` vers l'agent dédié
→ op dans l'AGENTS.md de l'agent + allow global `tools.sandbox.tools.allow`
+ deny du tool chez chaque sous-agent.
Réf. patron : `docs/architecture/spec-architecture-par-intention.md` et la mémoire
`project-intention-architecture`.

## Conception par commande

### gog (4) — pas de rebuild d'image
L'agent gog exécute déjà le binaire `gog`. Commandes prod de référence :
`gog gmail search <q> --max 10`, `gog gmail get <id>`, `gog gmail reply <id> --body <texte>`,
`gog drive search <q> --max 10`.

- **`/chercher`** : op `search` déjà présente dans `workspace-gog/AGENTS.md`.
  N'ajouter que l'outil `gog_search` + la skill `/chercher`. `delegateGog(api,"search",requête)`.
- **`/lire`** : ajouter l'op `get` à `workspace-gog/AGENTS.md`
  (`gog gmail get <argument>`). Outil `gog_get` + skill `/lire`. Le corps du mail
  est externe → **non fiable** : `main` l'encadre `<UNTRUSTED>` comme pour `/q` et
  `/source`. (La prod le passe en plus par l'injection-guard `_screen` ; durcissement
  de parité noté comme suite optionnelle, hors de ce lot.)
- **`/drive`** : ajouter l'op `drive_search` à `workspace-gog/AGENTS.md`
  (`gog drive search "<argument>" --max 10`). Outil `gog_drive_search` + skill `/drive`.
  Noms de fichiers externes → encadrés `<UNTRUSTED>` par `main`, comme `/lire`.
- **`/repondre`** : écriture. Réutiliser le mécanisme de brouillon existant.
  - Généraliser `pendingSend` en union discriminée :
    `{kind:"send", to, subject, body}` | `{kind:"reply", messageId, body}`.
  - `gog_reply <id> <texte>` (argMode raw : premier token = id, reste = corps) prépare
    un `pendingReply`, n'envoie pas, répond « Brouillon de réponse prêt. /confirm ».
  - **Le même `/confirm`** envoie selon `kind` (pour `reply` : op `reply` à l'agent gog) ;
    **le même `/annuler`** abandonne. TTL 10 min inchangé. Un nouveau brouillon
    (send ou reply) écrase l'ancien.
  - Ajouter l'op `reply` à `workspace-gog/AGENTS.md` (`gog gmail reply <id> --body <texte>`).

### wiki (2) — rebuild de l'image `Dockerfile.wiki`
L'agent wiki exécute une `wiki.py` allégée (ops capture/ingest/status/query/_ingest_worker).
`tags` et `kb_update` n'existent que dans le `mcp_server.py` prod ; il faut les
ajouter à la CLI `wiki.py` puis reconstruire l'image.

- **`/tags`** : ajouter l'op `tags` à `wiki.py` → appelle `collect_tags(wiki_root/"wiki")`,
  renvoie `{"tags":[…]}`. Lecture rapide, synchrone. Outil `wiki_tags` + skill `/tags`.
  AGENTS.md wiki : op `tags` (sans argument).
- **`/kbupdate`** : ajouter à `wiki.py` les ops `kb_update` et `_kb_update_worker`
  (worker async, sur le modèle de `ingest`/`_ingest_worker`) → appelle `update_kb(…)`
  sur le dernier clustering. L'agent wiki lance `_kb_update_worker` en
  `exec background: true` et répond une seule fois « Mise à jour du KB lancée en
  arrière-plan ». Outil `wiki_kb_update` + skill `/kbupdate`. AGENTS.md wiki : op
  `kb_update` avec la procédure async dédiée (calquée sur celle d'ingestion).

## Configuration

- `tools.sandbox.tools.allow` (global) : ajouter `gog_search, gog_get, gog_drive_search,
  gog_reply, wiki_tags, wiki_kb_update`.
- Deny chez les sous-agents : ajouter les 6 outils au `deny` de **chaque** sous-agent
  (wiki, scout, gog) — règle établie : tout sous-agent denie tous les outils
  d'orchestration, sinon il les voit et les appelle au lieu de faire son travail (boucle).
- Skills dans `~/.openclaw-slm/workspace/skills/` : `chercher, lire, drive, repondre, tags, kbupdate`.

## Build / déploiement

- Images : build `secretarius-gog` (Dockerfile.gog) + rebuild `secretarius-tiron`
  (base nue) à l'étape 0 ; rebuild `secretarius-wiki` (Dockerfile.wiki) après modif
  de `wiki.py` à l'étape 2.
- Plugin : `npm run build` → `openclaw --profile slm plugins install . --force` →
  `systemctl --user restart openclaw-gateway-slm`.

## Ordre d'exécution

0. Image dédiée `secretarius-gog` (Dockerfile.gog, tiron réduit à la base, bascule
   de l'agent gog) → vérifier `/inbox` toujours OK sur la nouvelle image.
1. gog : `/chercher` → `/lire` → `/drive` → `/repondre` (pas de rebuild image gog
   pour les commandes — l'image gog est figée à l'étape 0).
2. wiki : `/tags` → `/kbupdate` (rebuild image wiki).

Chaque commande **testée E2E via Telegram en session neuve** avant la suivante
(biais de session SLM : tester les skills en session neuve).

## Critères de succès (vérifiables)

- Image : `secretarius-gog:latest` existe ; l'agent gog y tourne et `/inbox` répond ;
  `docker run --rm secretarius-tiron:latest which gog` ne trouve plus le binaire.
- `/chercher motclé` → liste réelle de mails correspondants.
- `/lire <id>` (id issu de /inbox ou /chercher) → contenu du mail, encadré non fiable.
- `/drive motclé` → liste réelle de fichiers Drive correspondants.
- `/repondre <id> bonjour` → « Brouillon de réponse prêt. /confirm » ; `/annuler`
  → message d'abandon exact ; re-faire → `/confirm` → réponse réellement envoyée
  (vérifier réception).
- `/tags` → liste réelle des tags du KB.
- `/kbupdate` → « Mise à jour du KB lancée en arrière-plan » ; le KB reflète le
  dernier clustering après le run.

## Hors périmètre

- Calendar (events/create/delete) : scope OAuth non consenti (403 vérifié 2026-06-18,
  re-consentement utilisateur requis) + arguments structurés (create/delete) → voie
  conversation, lot séparé.
- Drive download/upload : transfert de fichier vers/depuis le sandbox à concevoir ;
  upload = scope write non vérifié. Lot séparé. (drive_search est dans ce lot.)
- Durcissement injection-guard sur le corps des mails lus (`/lire`) et les résultats
  drive (`/drive`) : parité prod optionnelle, suite possible.
