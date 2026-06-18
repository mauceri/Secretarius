# Recouvrement fonctionnel prod → SLM — design

Date : 2026-06-18
Branche cible : travail SLM (archi par intention)

## Objectif

Porter sur la version SLM (commandes déterministes) les fonctionnalités de la
version prod (outils MCP v0.1.0) encore manquantes, afin que le SLM puisse
remplacer la prod. Ce lot couvre **6 fonctionnalités débloquées** + une commande
de **mise en service OAuth autonome** (`/connecter`), précédées d'une **refonte
d'image** isolant gog dans une image dédiée `secretarius-gog` (étape 0). Les
**commandes** calendar (`/agenda`) et drive download/upload restent hors lot ;
mais le **scope** calendar est consenti dès `/connecter` (un seul consentement).

## Périmètre — 7 commandes

| Commande | Outil plugin | Agent · op | Nature |
|----------|--------------|-----------|--------|
| `/connecter` | `gog_connect_start` | gog · `auth_start` (nouvelle, async multi-tours) | mise en service OAuth |
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
  (`FROM openclaw-sandbox:bookworm-slim` + COPY `gog-bin` + COPY `gog-wrapper.sh`)
  **+ COPY `gog-auth-bridge.sh`** (pont OAuth, voir `/connecter`).
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

### gog — mise en service OAuth : `/connecter` (multi-tours)
Cas à part : l'autorisation Google est interactive et doit être faisable par
l'utilisateur final **sans shell ni assistance**, via Telegram. Le flux gog
`auth add --manual` est un **process unique interactif** (affiche une URL, attend
le collage de l'URL de redirection sur stdin) — vérifié : pas de découpage url/code.
Il faut donc un **pont stateful** entre deux messages Telegram. Contrainte : le
plugin (Node, gateway) ne peut pas spawn de process (scanner `child_process`) ;
le pont vit donc côté agent gog (exec background) + file-drop dans `/gog-config`.

Script `gog-auth-bridge.sh` (embarqué dans l'image `secretarius-gog`) :
1. lance `gog auth add <email> --manual --force-consent --services gmail,drive,calendar` ;
2. capture l'URL d'autorisation affichée → l'écrit dans `/gog-config/auth_url` ;
3. attend (poll) `/gog-config/auth_response`, puis l'injecte sur stdin de gog ;
4. au succès, écrit `/gog-config/auth_done` (et nettoie les fichiers temporaires).

**Tour 1** — `/connecter` → outil `gog_connect_start` → délègue op `auth_start` à
l'agent gog, qui lance le script en `exec background: true`. L'outil lit
`/gog-config/auth_url`, l'envoie dans le chat, arme l'état `pendingAuth`.
**Tour 2** — l'utilisateur consent sur son téléphone et recolle l'URL de
redirection ; le hook `before_agent_reply` (même mécanisme que `/confirm`), voyant
`pendingAuth` actif, écrit le message dans `/gog-config/auth_response`. Le script
finalise → token écrit. Réponse : « Compte Google connecté. »

Décisions : scopes `gmail,drive,calendar` (un seul consentement, valide l'utilisateur
2026-06-18) ; `--force-consent` pour garantir un refresh token frais.
**Effets de bord à connaître** : un `/connecter` **réécrit le token** (régénère le
refresh token) — tester avec ce fait en tête ; l'URL de redirection recollée contient
le code d'autorisation (secret) et transite par Telegram (canal privé du propriétaire,
acceptable). État `pendingAuth` avec TTL (ex. 10 min) ; un nouvel `/connecter` écrase
le précédent.

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

### wiki (2) — sans rebuild d'image (bind ro)
L'agent wiki exécute une `wiki.py` allégée (ops capture/ingest/status/query/_ingest_worker).
`tags` et `kb_update` n'existent que dans le `mcp_server.py` prod ; il faut les
ajouter à la CLI `wiki.py`. **Pas de rebuild** : `tools/` est monté en bind ro
(`/home/mauceric/Secretarius/Wiki_LM/tools:/wiki-tools:ro`), `kb_tags.py`/`kb_update.py`
y sont déjà et leurs deps (`frontmatter`, `numpy`) sont dans l'image. Modifier
`wiki.py` sur l'hôte suffit ; le prochain `exec python3 /wiki-tools/wiki.py` lit la
nouvelle version (sessions wiki fraîches à chaque délégation).

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

- `tools.sandbox.tools.allow` (global) : ajouter `gog_connect_start, gog_search,
  gog_get, gog_drive_search, gog_reply, wiki_tags, wiki_kb_update`.
- Deny chez les sous-agents : ajouter les 7 outils au `deny` de **chaque** sous-agent
  (wiki, scout, gog) — règle établie : tout sous-agent denie tous les outils
  d'orchestration, sinon il les voit et les appelle au lieu de faire son travail (boucle).
- État plugin : généraliser `pendingSend` (send|reply) ; ajouter `pendingAuth` (TTL).
  Hook `before_agent_reply` étendu : intercepte aussi le retour OAuth quand `pendingAuth` actif.
- Skills dans `~/.openclaw-slm/workspace/skills/` : `connecter, chercher, lire, drive,
  repondre, tags, kbupdate`.

## Build / déploiement

- Images : build `secretarius-gog` (Dockerfile.gog) + rebuild `secretarius-tiron`
  (base nue) à l'étape 0 uniquement. **Aucun rebuild wiki** : `wiki.py` est monté
  en bind ro, la modif prend effet au prochain appel.
- Plugin : `npm run build` → `openclaw --profile slm plugins install . --force` →
  `systemctl --user restart openclaw-gateway-slm`.

## Ordre d'exécution

0. Image dédiée `secretarius-gog` (Dockerfile.gog, tiron réduit à la base, bascule
   de l'agent gog) → vérifier `/inbox` toujours OK sur la nouvelle image.
1. gog : `/connecter` → `/chercher` → `/lire` → `/drive` → `/repondre` (pas de
   rebuild image gog pour les commandes — l'image gog est figée à l'étape 0).
2. wiki : `/tags` → `/kbupdate` (modif `wiki.py`, sans rebuild — bind ro).

Chaque commande **testée E2E via Telegram en session neuve** avant la suivante
(biais de session SLM : tester les skills en session neuve).

## Critères de succès (vérifiables)

- Image : `secretarius-gog:latest` existe ; l'agent gog y tourne et `/inbox` répond ;
  `docker run --rm secretarius-tiron:latest which gog` ne trouve plus le binaire.
- `/connecter` → l'URL d'autorisation arrive dans le chat ; après consentement et
  recollage de l'URL de redirection → « Compte Google connecté » ; `gog auth list`
  montre un token frais ; `/agenda`-équivalent (`gog calendar events`) ne renvoie
  plus 403 (scope calendar effectivement accordé).
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

- Calendar — **commandes** (`/agenda`, events/create/delete) hors lot : arguments
  structurés (create/delete) → voie conversation, lot séparé. NB : le **scope**
  calendar est consenti dès `/connecter`, donc le futur lot calendar n'exigera pas
  de re-consentement.
- Drive download/upload : transfert de fichier vers/depuis le sandbox à concevoir ;
  upload = scope write non vérifié. Lot séparé. (drive_search est dans ce lot.)
- Durcissement injection-guard sur le corps des mails lus (`/lire`) et les résultats
  drive (`/drive`) : parité prod optionnelle, suite possible.
