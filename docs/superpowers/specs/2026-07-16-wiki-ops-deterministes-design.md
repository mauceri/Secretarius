# Ops wiki déterministes — le plugin exécute wiki.py sans orchestrateur LLM

- Date : 2026-07-16
- Statut : design validé, en attente d'exécution
- Auteur : Christian Mauceri + Claude

## Problème

Les commandes wiki (`/c`, `/q`, `/ingest`, `/wikistatus`, …) sont déléguées à un
**agent LLM** (Euria Mistral-119B, l'« agent wiki ») via
`delegateWiki → api.runtime.subagent.run("op: <op> | <arg>")`. Cet agent est censé
exécuter `wiki.py <op>` et reformuler le JSON, mais il a la **latitude de ne pas le
faire** : sur `/q tee gpu` (2026-07-16) il a **inventé** une réponse (« infrastructure
matérielle du cluster », liens `[[hardware-cluster-inventory]]` bidon) sans jamais
lancer `wiki.py query` — alors que `wiki.py query "tee gpu"` renvoie une synthèse
correcte. Même famille de dérapage que « Ingestion terminée » sur une capture.

Ces 6 ops (`capture`, `query`, `status`, `ingest`, `tags`, `kb_update`) sont
**purement mécaniques** : un appel `wiki.py` qui renvoie un JSON. La génération
(synthèse `query`, résumé `ingest`) se fait **dans `wiki.py`** (backend Euria via
`OPENAI_BASE_URL`), pas dans l'agent orchestrateur. L'agent LLM n'ajoute donc que de
la latitude d'erreur.

## Objectif

Rendre les 6 ops wiki **déterministes** : le plugin `derisk-deleg` exécute
`wiki.py <op>` directement dans le **sandbox wiki** (même isolation qu'aujourd'hui),
capture le JSON et le formate. L'orchestrateur LLM disparaît pour ces ops.

## Périmètre

**Inclus** : les 6 ops de `delegateWiki` (`capture`, `query`, `status`, `ingest`,
`tags`, `kb_update`).

**Exclus** : `delegateGog` (inbox/search/drive/repondre) — OAuth + envoi sensible via
`/confirm`, chaîne déjà validée, chantier distinct. Aucun changement à `wiki.py`
(le code des ops est correct ; c'est l'orchestration qui change).

## Décisions actées

| Sujet | Décision |
|-------|----------|
| Isolation | `wiki.py` reste dans le **sandbox `secretarius-wiki`** (image/montages/env inchangés), lancé via le SDK sandbox |
| Agent wiki | **Conservé comme profil de sandbox** (source unique de la config Docker dans `agents.list[wiki].sandbox`) ; son rôle LLM disparaît, son `AGENTS.md` d'orchestration devient caduc |
| Conteneur | **Une `sessionKey` stable** → un seul conteneur wiki réutilisé (fin de la fuite de conteneurs orphelins, une par op aujourd'hui) |
| Génération | Euria reste moteur **dans `wiki.py`** (synthèse/résumé), intouché |
| gog | Inchangé |

## Architecture

Nouvelle fonction dans `derisk-deleg/src/index.ts` :

```
runWikiOp(api, op: string, arg: string): Promise<string>
  1. Résout le sandbox wiki : resolveSandboxContext({config, sessionKey})
     → SandboxBackendHandle (image secretarius-wiki, montages wiki + euria-key).
  2. handle.runShellCommand({ command: ["python3","/wiki-tools/wiki.py",op,arg], ... })
  3. Parse stdout (JSON de wiki.py). Formate selon l'op (voir plus bas).
  4. Renvoie le message utilisateur (string).
```

Elle **remplace `delegateWiki`** partout où il est appelé :
- **le hook `before_agent_reply`** — pour les ops ayant une commande slash
  (`/c`→capture, `/q`→query, `/wikistatus`→status, `/ingest`→ingest) ;
- **les 6 outils enregistrés** `wiki_capture` / `wiki_query` / `wiki_status` /
  `wiki_ingest` / `wiki_tags` / `wiki_kb_update` (que l'agent principal peut appeler ;
  `tags` et `kb_update` n'ont pas de commande slash → uniquement via ces outils).

`delegateGog` et `runAndRead` restent inchangés pour gog. `ingest` et `kb_update`
lancent leur propre worker en arrière-plan **dans `wiki.py`** (déjà le cas depuis le
correctif du worker détaché) ; `runWikiOp` ne fait que lancer et rapporter le `status`.

## Formatage par op (déterministe, remplace la reformulation de l'agent)

| op | JSON `wiki.py` | message |
|----|----------------|---------|
| `query` | `{synthesis, references}` ou `{error}` | `synthesis` verbatim (déjà markdown prêt) ; ou l'`error` |
| `capture` | `{files: [...]}` | « Capturé : `<fichiers>` (en file d'attente pour ingestion). » |
| `ingest` | `{status, queued}` | `launched`→« Ingestion lancée en arrière-plan. » ; `nothing_to_do`→« Rien à ingérer. » ; `already_running`→« Ingestion déjà en cours. » |
| `status` | `{running, last_run, pending, blocked_files}` | ligne sobre : en cours / dernier run / N en attente / bloqués |
| `tags` | (liste) | mise en forme sobre |
| `kb_update` | `{status, …}` | mise en forme sobre |

Règle générale : si le JSON contient `error`, le surfacer **verbatim** ; ne jamais
inventer.

## Gestion d'erreur

- Exit non-nul, stdout vide, ou stdout non-JSON → `Erreur wiki : <stderr tronqué>`
  (déterministe, pas d'invention).
- Timeout du `runShellCommand` (ex. 120 s) → message « L'opération wiki a expiré. »

## Flux (exemple `/q`)

`/q tee gpu` → hook `before_agent_reply` → `callRouter` → `{ok, /q, "tee gpu"}` →
`runWikiOp(api, "query", "tee gpu")` → exec sandbox `wiki.py query "tee gpu"` →
stdout `{synthesis, references}` → renvoie `synthesis` → `{handled:true, reply}`.
**Aucun tour de LLM.** Idem `/c`, `/ingest`, `/wikistatus`.

## Confidentialité

Le résultat wiki revient **directement à l'utilisateur** via le hook, sans transiter
par un LLM orchestrateur → le contenu du wiki (non fiable) ne devient jamais des
*instructions* pour un modèle. Le risque d'injection de prompt sur ce chemin
disparaît. La synthèse reste produite par Euria *dans* `wiki.py` (inchangé). La
frontière `<UNTRUSTED>` côté Tiron ne concernait que le passage par un LLM, supprimé
ici.

## Tests

- **TDD unitaire** : fonctions de formatage `formatWikiResult(op, json) → string`
  (pures : une par op + le cas `error`). C'est le cœur testable.
- **Sandbox exec mocké** en unitaire (injecter un faux `runShellCommand` renvoyant un
  JSON connu) → vérifier que `runWikiOp` parse et formate correctement, et gère
  exit≠0 / stdout non-JSON.
- **E2E réel** (verify) : `/q tee gpu` renvoie la vraie synthèse ; `/c #x <url>` fait
  une capture (`.url` dans `raw/`) ; `/ingest` renvoie `launched` et le worker tourne ;
  `/wikistatus` renvoie l'état. Vérifier qu'aucun conteneur wiki n'est spawné par op
  (réutilisation).

## Critères de succès

1. `/q tee gpu` via Telegram renvoie la synthèse issue de `wiki.py query` (plus
   d'invention), confirmé par les logs (exec `wiki.py`, pas de tour de LLM wiki).
2. `/c`, `/ingest`, `/wikistatus` déterministes et corrects.
3. Un seul conteneur wiki réutilisé (plus de prolifération orpheline).
4. Tous les tests verts, sortie propre.

## Inconnue à lever au plan (exploration en tête d'implémentation)

Le fil exact côté plugin pour : obtenir l'`OpenClawConfig`, construire/retrouver la
`sessionKey` du sandbox wiki, et créer/réutiliser le `SandboxBackendHandle`. L'API est
localisée (`plugin-sdk/sandbox` : `resolveSandboxContext`,
`SandboxBackendHandle.runShellCommand`, `getSandboxBackendManager`,
`SandboxBackendFactory`), mais le câblage précis (comment `api` expose la config, quel
`sessionKey` cible le sandbox `agents.list[wiki]`) est à confirmer par une courte
exploration avant d'écrire `runWikiOp`. Repli si l'API sandbox s'avère impraticable
depuis un plugin : `runPluginCommandWithTimeout` avec `docker exec`/`docker run` ciblé
sur `secretarius-wiki` (moins propre, à éviter).

## Déploiement

Build du plugin (`derisk-deleg`) + copie vers `~/.openclaw/extensions/derisk-deleg` +
`restart openclaw-gateway`. Allègement de l'`AGENTS.md` wiki (orchestration caduque)
optionnel, non bloquant.

## Hors périmètre

- gog déterministe (chantier distinct).
- Purge des 14 conteneurs orphelins actuels (nettoyage ponctuel, séparé).
- Bascule de l'ingestion sur phi-4 local (autre chantier, cf. `[[project_ingestion_phi4_passages]]`).
