# Diagnostic et observabilité — Secretarius (architecture par intention)

> Document rédigé pour l'**architecture cible** : dispatch déterministe
> `commande → skill → agent`, un skill par agent, outils **bakés dans l'image**
> de chaque agent, et une voie conversationnelle où l'intention se résout
> toujours en une **commande nommée + confirmation** avant toute action.
> Les chemins de fichiers sont ceux réellement utilisés par l'instance SLM ;
> on révisera ce document quand l'architecture cible sera en place.

## 0. Conventions

- `$SLM` = racine de l'instance SLM : `~/.openclaw-slm`.
- `$KB` = base de connaissances wiki : `~/Documents/Arbath/Wiki_LM`.
- Les conteneurs d'agents n'ont que `python3` (pas de `python`).
- Service systemd : `openclaw-gateway-slm` (utilisateur). La **prod** est une
  instance distincte (`~/.openclaw`, service `openclaw-gateway`) — ne pas confondre.

## 1. Modèle mental

Une requête utilisateur traverse, dans l'ordre :

```
Telegram (ou autre canal)
   │  message entrant
   ▼
Gateway ──► Dispatcher de commandes (déterministe)
   │             │  /commande → skill du même nom → agent cible
   │             │  commande inconnue → réponse fixe « pas de skill pour /xxx »
   │             │  message en langage naturel → voie conversation
   ▼             ▼
Orchestrateur Tiron (agent `main`)
   │  sessions_spawn(agentId=…, task="op: <op> | <arg>")
   ▼
Sous-agent isolé (wiki, scout, …) — image autonome, outils bakés
   │  exec d'un outil dans le bac à sable → résultat JSON
   ▼
Artefacts métier (KB wiki, mail envoyé, événement agenda…)
```

**Principe d'or du diagnostic :** chaque étage laisse une trace distincte. Un
incident se diagnostique en **descendant les étages** jusqu'à trouver l'écart
entre ce qui est *affirmé* et ce qui s'est *réellement produit*.

## 2. Les couches de traces

| # | Couche | Où | Ce qu'on y trouve |
|---|--------|-----|-------------------|
| 1 | Gateway / canal | `journalctl --user -u openclaw-gateway-slm` ; `/tmp/openclaw/openclaw-AAAA-MM-JJ.log` | démarrage, connexion du bot, messages entrants/sortants, erreurs globales |
| 2 | Dispatcher de commandes | log gateway (lignes de dispatch) | quelle commande a matché quel skill ; refus « commande inconnue » |
| 3 | Orchestrateur Tiron | `$SLM/agents/main/sessions/<sid>.jsonl` (+ `.trajectory.jsonl`) | message reçu, skill/agent choisi, tâche déléguée, réponse relayée |
| 4 | Sous-agents | `$SLM/agents/<agentId>/sessions/<sid>.jsonl` | commandes `exec` réelles, `toolResult`, sortie finale de l'agent |
| 5 | Registre des délégations | `$SLM/subagents/runs.json` | `task` déléguée, agent cible, `outcome {status, timing}` |
| 6 | Artefacts métier | ex. wiki : `$KB/raw/`, `$KB/.ingested`, `$KB/.ingest_state.json`, `$KB/log.md`, `$KB/wiki/` | preuve concrète qu'une opération a (ou non) abouti |

> ⚠️ `runs.json` ne contient **que** le statut et les horodatages d'un run,
> **pas le texte du résultat**. Le texte vit dans la session du sous-agent (couche 4).

## 3. Tracer une requête de bout en bout

Les clés de corrélation, dans l'ordre où on les récupère :

1. **`message_id`** (canal) → trouve la requête dans la session `main`.
2. **`sessionId`** de `main` → fichier `$SLM/agents/main/sessions/<sid>.jsonl`.
   Le `.trajectory.jsonl` donne la séquence d'événements (`model.completed`, etc.).
3. **commande / skill** déclenché (couche 2) → confirme le routage déterministe.
4. **`sessions_spawn`** → renvoie `runId` + `childSessionKey` (dans la session `main`).
5. **`childSessionKey`** → via `$SLM/agents/<agentId>/sessions/sessions.json`,
   retrouve le **fichier de session du sous-agent** (couche 4).
6. **`outcome`** du run → `runs.json` (cherche le `runId` ou le `controllerSessionKey`).
7. **artefact métier** (couche 6) → preuve finale.

**Astuce livraison asynchrone :** la réponse d'un sous-agent revient à Tiron par
une **annonce push**, dans un **2ᵉ tour autonome** (un 2ᵉ `model.completed` dans
le `.trajectory.jsonl` de `main`, sans message utilisateur). Un appel CLI
*one-shot* (`openclaw agent --json`) rend la main au `sessions_yield` avec
`payloads` vide : **c'est normal**, le résultat arrive ensuite. Ne jamais
conclure « non livré » sur la seule sortie one-shot ; vérifier le 2ᵉ tour.

## 4. Analyser un sous-agent

1. Localiser sa session (étape 5 ci-dessus), ou la plus récente :
   ```
   ls -t $SLM/agents/<agentId>/sessions/*.jsonl | grep -v trajectory | head -1
   ```
2. Lire le JSONL. Chaque ligne est un message ; le `content` est une liste de
   blocs : `text` (raisonnement / réponse), `toolUse` (appel d'outil, avec
   `input.command` pour un `exec`), `toolResult` (sortie réelle de l'outil).
3. Extraire trois choses :
   - **ce que l'agent a exécuté** (`toolUse` → `command`),
   - **ce que l'outil a renvoyé** (`toolResult`),
   - **ce que l'agent a affirmé** (dernier bloc `text` `assistant`).

## 5. Vérité-terrain : ne jamais croire le récit de l'agent

C'est la règle la plus importante. Un SLM **confabule** volontiers un succès :
il annonce « ingestion terminée avec succès » alors que l'outil a échoué ou
n'a jamais tourné. **Toujours croiser le récit avec une preuve indépendante :**

| L'agent affirme… | Preuve à exiger |
|------------------|-----------------|
| « capturé » | un fichier dans `$KB/raw/*.url` daté du moment |
| « ingéré avec succès » | l'entrée dans `$KB/.ingested` **et** la page `$KB/wiki/sources/src-*.md` datée du moment (pas une page ancienne homonyme) |
| « X ressources ingérées » | `outcome.status` + `last_run` du `status` réel ; attention : un compteur **n'attribue pas** le résultat à une URL précise |
| « mail envoyé » | la trace côté `gog` / l'élément dans « Envoyés » |

Piège classique observé : un `status` renvoie `{"ingested": 1, "pending": 1}`
(le `1 ingéré` = un run **antérieur**, le `1 en attente` = la nouvelle URL).
L'agent attribue le compteur à la dernière URL et déclare un faux succès. La
seule parade est la preuve métier (couche 6).

## 6. Vérifier le dispatch déterministe (architecture cible)

- **Routage** : la commande `/x` a-t-elle invoqué le **skill `x`** ? Le dispatch
  ne doit **jamais** dépendre d'une décision du modèle. Si une commande part
  vers le mauvais agent, le défaut est dans le **dispatcher**, pas dans le prompt.
- **Commande inconnue** : `/inconnue` doit produire une **réponse fixe**
  (« pas de skill pour /inconnue »), sans passer par le LLM.
- **Voie conversation** : une demande en langage naturel ne doit **jamais**
  déclencher une action directement. Elle se résout en une **commande proposée +
  confirmation**. Vérifier qu'une étape de confirmation précède toute écriture
  externe (mail/agenda/drive).

## 7. Recettes ciblées (une question → un endroit)

- **Quel skill/agent a traité le dernier message ?**
  → log gateway (couche 2) + `task` du dernier `sessions_spawn` dans la session `main`.
- **Qu'a réellement exécuté l'agent X ?**
  → sa dernière session (couche 4), blocs `toolUse`/`toolResult`.
- **Le succès annoncé est-il réel ?**
  → artefact métier (couche 6), jamais le texte de l'agent.
- **Pourquoi une commande n'a rien fait ?**
  → vérifier dans l'ordre : dispatch (couche 2) → tâche déléguée (couche 3) →
  exec du sous-agent (couche 4) → artefact (couche 6). L'écart apparaît à un
  étage précis.

## 8. Signatures d'échec courantes

| Symptôme | Cause probable | Où confirmer |
|----------|----------------|--------------|
| Faux succès (annoncé OK, artefact absent) | confabulation du SLM | couche 6 |
| Opération droppée (capture sans ingest) | tâche bundlée par l'orchestrateur / règle anti-enchaînement de l'agent | `task` en couche 3 |
| Mauvais routage (commande → mauvais agent) | en arch cible : impossible (dispatch déterministe). Si observé : dispatcher mal configuré, ou routage encore confié au LLM | couche 2 |
| `NO_REPLY` | course de l'annonce push : le résultat est arrivé après la réponse | `outcome` (couche 5) |
| `command not found` | binaire absent de l'image (ex. `python` au lieu de `python3`) | `toolResult` en couche 4 |
| Réponse vide au `yield` | normal en CLI one-shot ; le résultat arrive au 2ᵉ tour | `.trajectory.jsonl` de `main` |

## 9. Lire un transcript JSONL (format)

Petit extracteur réutilisable :

```python
import json, sys
for l in open(sys.argv[1]):
    if not l.strip():
        continue
    m = json.loads(l).get("message", json.loads(l))
    role = m.get("role", "")
    c = m.get("content")
    if isinstance(c, list):
        for b in c:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t in ("toolUse", "tool_use"):
                inp = b.get("input") or {}
                print("EXEC:", json.dumps(inp.get("command") or inp, ensure_ascii=False)[:200])
            elif t == "toolResult":
                cc = b.get("content")
                print("  ->", (cc if isinstance(cc, str) else json.dumps(cc, ensure_ascii=False))[:200])
            elif t == "text" and b.get("text", "").strip():
                print(f"[{role}] " + b["text"][:200].replace("\n", " "))
```

## 10. Hygiène

- Le log dédié tourne par date : `/tmp/openclaw/openclaw-AAAA-MM-JJ.log`.
- Les sessions s'accumulent dans `agents/*/sessions/` ; trier par date (`ls -t`).
- `runs.json` peut être nettoyé : un run absent n'est pas une preuve d'échec,
  croiser avec la session de l'agent et l'artefact métier.
- Toujours diagnostiquer en **session neuve** quand on teste un skill modifié :
  l'historique d'une session persistante peut l'emporter sur le prompt à jour.
