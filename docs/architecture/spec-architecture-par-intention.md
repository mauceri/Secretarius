# Spec — Architecture par intention (Secretarius)

> Statut : **proposition de conception, 2026-06-15** (archive). L'architecture par
> intention décrite ici a depuis été largement implémentée — voir l'état en service
> dans `Secretarius.md`.
> Remplace le routage « par jugement du LLM » par un **dispatch déterministe
> commande → outil**, et clarifie le rôle des fichiers d'instruction.

## 1. Problème

Le routage et le respect du contrat d'opération **par jugement du LLM** sont non
fiables, même avec un grand modèle (Qwen3.5-397B). Cas réel (incident *marp*) :

- `/c <url>` routé tantôt vers `wiki`, tantôt vers `scout` ;
- tâche déléguée « capturer **et** ingérer » alors que `/c` = capture seule ;
- ingestion jamais lancée, puis **faux succès** rapporté (confabulation au statut).

**Cause racine.** Dans OpenClaw, les *skills* sont **rendus dans le system prompt**
(`renderSkills`, `SkillsForSystemPrompt`) : ils sont **advisory**, le LLM décide.
De plus, l'AGENTS.md de l'orchestrateur disait littéralement d'envoyer une
« question en langage naturel » à l'agent wiki — contredisant le contrat
`op: <op> | <arg>` du skill. Le déterminisme ne peut pas venir des skills.

## 2. Principes

1. **Déterminisme par commande → outil**, pas par jugement du LLM.
2. **1 commande ↔ 1 skill (user-invocable) ↔ 1 outil cible.**
3. **Commande inconnue → réponse fixe** (« pas de skill pour /xxx »).
4. **Chaque agent a son skill** ; ses **outils sont bakés dans son image** (pas de bind).
5. **Deux voies** : (a) **commande** = déterministe ; (b) **conversation** =
   l'intention se **résout en une commande proposée + confirmation**, et ne
   déclenche **jamais** une action directement.
6. **Supprimer le routage par intention ad hoc** (gog passe par des commandes).

## 3. Mécanique OpenClaw retenue (vérifiée sur 6.1)

Schéma natif (`types-*.d.ts`) :

```ts
type SkillCommandDispatchSpec = {
  kind: "tool";       // route la commande vers un outil
  toolName: string;   // nom d'un outil de l'agent (AnyAgentTool.name)
  argMode?: "raw";    // transmet la chaîne d'arguments brute, sans parsing
};
// SkillCommandSpec.dispatch?: SkillCommandDispatchSpec  // "Optional deterministic dispatch behavior"
```

- **Dispatch déterministe** : une skill *user-invocable* avec `command-name` et
  `dispatch: { kind: "tool", toolName, argMode: "raw" }` fait que `/<cmd> <args>`
  **appelle directement l'outil `toolName` avec les args bruts, sans le modèle**.
- **Activation** : `commands.nativeSkills` (global) et
  `channels.<provider>.commands.nativeSkills` (par canal).
- **Refus déterministe** : `unknownCommand` natif.
- **Voie confirmation** : flux natif `/approve <id> <decision>` + `execApprovals`
  (déjà configuré pour Telegram), et hooks `HookBeforeToolCall` (`ask`/`deny`).

**Contrainte clé.** La cible est un **outil** et `argMode: "raw"` passe une
**chaîne brute**. `sessions_spawn` (déléguer à un sous-agent) attend des arguments
**structurés** `{agentId, task}` → **non ciblable directement** en mode raw.
Donc chaque commande doit cibler un **outil qui accepte une chaîne brute** :
un **outil custom** (petit plugin) qui parse les args et exécute la délégation.

## 4. Composants

- **Orchestrateur Tiron** (agent `main`).
- **Outils custom** (plugin, exposés comme `AnyAgentTool`) — un par opération.
  Chacun parse la chaîne brute et **encapsule** la délégation à l'agent isolé
  (ou l'appel `gog`). C'est le « handler » déterministe.
- **Agents isolés** : `wiki`, `scout` — image autonome, **outils métier bakés**.
- **Skills user-invocable** — un par commande, surtout du frontmatter
  (`command-name` + `dispatch`). Le corps sert la voie conversation (description
  d'intention pour le classifieur / l'aide).
- **Classifieur d'intention** (petit modèle local, entraîné sur le corpus seed,
  cf. `corpus-intentions-seed.md`) — pour la **voie conversation** uniquement :
  langage naturel → intention → **commande proposée**. Jamais exécuteur.

## 5. Taxonomie commande ↔ skill ↔ outil ↔ agent

| Commande | Skill | Outil cible (custom) | Agent / effet | Confirmation |
|----------|-------|----------------------|---------------|--------------|
| `/c` | `c` | `wiki_capture` | `wiki` (capture) | non (local, réversible) |
| `/ingest` | `ingest` | `wiki_ingest` | `wiki` (file d'attente) | non |
| `/wiki-status` | `wiki-status` | `wiki_status` | `wiki` | non |
| `/q` | `q` | `wiki_query` | `wiki` | non (lecture) |
| `/source` | `source` | `source_read` | `scout` (lecture externe) | non (lecture) |
| `/mail` | `mail` | `gog_mail` | `gog` | **oui** (envoi) |
| `/agenda` | `agenda` | `gog_calendar` | `gog` | **oui** (écriture) |
| `/drive` | `drive` | `gog_drive` | `gog` | **oui** (écriture/partage) |
| `/help` | (natif) | — | Tiron direct | — |
| *(inconnu)* | — | — | `unknownCommand` → refus fixe | — |

## 6. Voie conversation (langage naturel)

- Le **classifieur** (petit modèle entraîné sur le corpus seed) mappe un message
  NL vers une **intention** (les 10 du corpus), donc vers une **commande candidate**.
- **Invite de confirmation** : avant d'exécuter la commande inférée, afficher
  **« Exécuter `/x <args>` ? (Oui/Non) »** et n'exécuter qu'après *oui*. C'est le
  garde-fou de la voie conversation (la classification peut se tromper — une
  telle invite aurait intercepté le bundling *marp*).
- **Politique URL nue** (sans verbe ni `/c`) : cas ambigu → **décision à figer**
  (défaut proposé : `wiki_capture`, ou demander). À encoder dans le corpus.
- Le LLM orchestrateur ne **route** plus ; il **propose** une commande et
  **relaie** des résultats d'outils déterministes.

### 6.1 Politique d'approbation configurable (interrupteur)

L'invite Oui/Non est **optionnelle**, pilotée par une **politique d'approbation**
à deux niveaux qui se combinent :

- **Par commande** (défaut encodé dans la spec de commande, `confirm: true|false`) :
  les commandes d'**action** (mail/agenda/drive/ingest) confirment ; les **lectures**
  (status/query/source) non.
- **Interrupteur global** par-dessus, drapeau de config
  `conversation.confirmResolvedCommand: always | actions-only | never` :
  - `always` — confirmer **toute** commande inférée (recommandé en phase de mise
    au point : on voit la proposition du classifieur avant exécution) ;
  - `actions-only` — confirmer seulement les actions (usage courant) ;
  - `never` — exécution directe (mode expert).
  Modifiable à chaud via `/config set` (propriétaire) ou `/debug set` (runtime).

**Primitives natives** : l'invite et la résolution réutilisent le flux
d'approbation natif (`/approve <id> <decision>` + `execApprovals`) et la décision
**`ask`** du hook `HookBeforeToolCall`.

**Portée** : la confirmation s'applique à la **voie conversation** (intention
*inférée*). Dans la voie **commande** (`/x` tapé explicitement), l'intention est
certaine → pas de confirmation, sauf action externe qui garde son `/approve`.

## 7. Rôle des fichiers d'instruction (lève l'incohérence relevée)

| Fichier | Contient | Ne contient PAS |
|---------|----------|------------------|
| `AGENTS.md` (Tiron) | agents/outils disponibles, routine de session, principe zéro-initiative, voie confirmation | le contrat d'op (devenu déterministe via dispatch) |
| `SOUL.md` (Tiron) | personnalité + règles absolues (Scout pour lecture externe, **carve-out** wiki) | logique de routage |
| `SKILL.md` (par commande) | frontmatter `command-name` + `dispatch` ; description d'intention pour la voie conversation | instructions d'exécution fragiles |
| `AGENTS.md` (agent wiki/scout) | comment l'agent exécute ses ops via ses outils **bakés** | — |
| `TOOLS.md` | spécificités d'environnement | — |

Avec le dispatch déterministe, le contrat d'opération **n'est plus** de la prose
fragile dupliquée : il est **encodé** dans le `dispatch` du skill + l'outil custom.

## 8. Outils bakés dans l'image (remplace le bind)

Chaque agent embarque ses outils dans son image (Dockerfile `COPY`), supprimant
le bind hôte et `dangerouslyAllowExternalBindSources`. Image autonome et
portable (VPS), et cohérent avec le futur `agent-plugin` / `create-agent`.

## 9. Sécurité

- **Voie confirmation** (`/approve` + `execApprovals`) sur toute écriture externe.
- **Scout obligatoire** pour le contenu externe : l'**ingest** doit faire son
  fetch **via Scout** (anti-injection) — point #3 **encore ouvert** (aujourd'hui
  l'agent wiki fetche en direct, cf. roadmap). Dépendance à résoudre.
- Le déterminisme **réduit** la surface d'injection (le LLM ne décide plus du
  routage ni de l'op).

## 10. Étapes d'implémentation (esquisse, à valider)

1. **Spike** : une skill user-invocable triviale avec `dispatch: {kind:tool,
   toolName:"…"}` → vérifier que `/cmd args` appelle l'outil **sans le modèle**.
   → verify : la trace montre l'appel d'outil, pas un tour de modèle.
2. **Outils custom** (plugin) : commencer par `wiki_*` (capture/ingest/status/query).
   → verify : chaque outil exécute l'op réelle et renvoie un JSON fidèle.
3. **Skills par commande** + `dispatch` ; activer `nativeSkills`.
   → verify : routage déterministe sur les 10 cas du harnais.
4. **Voie confirmation** pour `gog`.
   → verify : aucune écriture externe sans `/approve`.
5. **Bake** des outils dans les images.
   → verify : suppression du bind + `dangerouslyAllowExternalBindSources`.
6. **Classifieur d'intention** (corpus seed) pour la voie conversation.
   → verify : taux de classification sur jeu de test tenu à l'écart.

## 11. Points ouverts (à vérifier avant/pendant l'étape 1-2)

- **Écriture d'un plugin d'outils** pour l'instance : faisabilité et coût.
- **Un outil custom peut-il spawner un sous-agent** (`sessions_spawn` en interne)
  pour atteindre `wiki`/`scout`, ou doit-il appeler le conteneur autrement ?
- **Permissions** : l'outil cible doit être autorisé pour l'agent `main`
  (`tools.sandbox.allow`).
- **`argMode`** : seul `"raw"` est documenté — confirmer qu'il n'y a pas de mode
  structuré ; sinon le parsing reste dans l'outil custom.
- **Politique URL nue** (capture vs demander).
- **#3 Scout/ingest** : refondre le fetch d'ingest pour passer par Scout.

## 12. Résultats du spike (2026-06-15)

Spike réalisé : skill *user-invocable* `spikeping` avec `command-dispatch: tool`,
testée via Telegram (le CLI `agent --message` **n'exécute pas** les slash-commands).

- ✅ **Dispatch déterministe confirmé** : `/spikeping` ciblant `sessions_list` a
  renvoyé le **JSON brut de l'outil**, sans tour de modèle ni prose.
- ✅ Frontmatter validé : `user-invocable: true`, `command-dispatch: tool`,
  `command-tool: <nom>`, `command-arg-mode: raw`. L'outil reçoit
  `{ command: "<args bruts>", commandName, skillName }`.
- ⚠️ **Surface d'outils restreinte au dispatch** : `command-tool: exec` →
  « Tool not available: exec ». Disponibles : `gog, read, sessions_list,
  sessions_spawn, sessions_yield`. Nos outils custom devront être **enregistrés
  dans cette surface autorisée**.
- ⚠️ **Délégation sous-agent** : `sessions_spawn` est disponible mais **structuré**
  (`{agentId, task}`), incompatible avec `argMode: raw`. → **outil custom requis**
  qui reçoit `{command}` et appelle `sessions_spawn` en interne.

**Dé-risque plugin (2026-06-15) — résultat partiel.** Le SDK simple
`defineToolPlugin` (`openclaw/plugin-sdk/tool-plugin`) expose un contexte d'outil
(`OpenClawPluginToolContext`) avec : config, `fsPolicy`/`workspaceDir`/`agentDir`,
`agentId`, `sessionKey`/`sessionId`, `activeModel`, `browser`, `deliveryContext`,
`hasAuthForProvider`/`resolveApiKeyForProvider`, `requesterSenderId`, `sandboxed`.
**Pas de méthode pour spawner un sous-agent.** Conséquences :
- **Capture (`/c`)** : faisable **déterministe et autonome** par un outil (écrire le
  `.url` via `fsPolicy`/`workspaceDir`) — sans sous-agent ni LLM. Plus simple que
  l'archi actuelle.
- **Ingest / query / source / gog** : nécessitent l'agent isolé ; le SDK d'outil
  simple ne peut pas y déléguer. → **plugin plus lourd** (`definePlugin` / SDK
  runtime — expose-t-il l'orchestration ? à vérifier) **ou pont** (HTTP
  `tools-invoke` du gateway, ou file-drop pour l'ingest async).

**Fork RÉSOLU (2026-06-16).** Le SDK complet (`definePlugin`) expose
l'orchestration via **`api.runtime.subagent`** :
`run({ sessionKey, message, deliver })` → `{ runId }`, puis `waitForRun({ runId,
timeoutMs })` et `getSessionMessages({ sessionKey, limit })`. **Les plugins non-
trusted peuvent lancer des sous-agents** (seul l'override provider/model exige
`plugins.entries.<id>.subagent.allowModelOverride: true`).

Chemin complet retenu (natif, sans pont HTTP/file-drop) :
- **Plugin complet** : ses outils ferment sur `api`, donc `execute` appelle
  `api.runtime.subagent.run(...)`. (Le SDK simple `defineToolPlugin` n'a pas
  `api.runtime` — d'où le blocage du 15/06.)
- `/c <args>` → skill `command-dispatch: tool` → outil `wiki_capture` déterministe
  → délègue à `wiki` avec **`message: "op: capture | <args>"` construit par le code**.
- Corrige (1) routage **et** contrat d'op (le tool bâtit le message exact ;
  Tiron ne free-forme plus), en **préservant** l'isolation du sous-agent.

**CONFIRMÉ par spike (2026-06-16).** Chaîne E2E prouvée via Telegram :
`/derisk` → dispatch déterministe → outil de plugin custom `derisk_wiki_status`
→ `api.runtime.subagent.run(...)` → sous-agent **wiki** exécutant `op: status`.
Un outil de plugin complet (`definePluginEntry` + `register(api)`, execute fermant
sur `api`) accède bien à `api.runtime.subagent`. Notes d'impl : import depuis
`openclaw/plugin-sdk/core` ; install via `openclaw plugins install <dir>` ;
outil à déclarer dans `tools.sandbox.tools.allow`.

**E2E impeccable après polissage (2026-06-16)** : `/derisk` → `status=ok` + le vrai
JSON de `wiki.py status` relayé. Règles de conception confirmées :
- `waitForRun → {status}` (pas le texte) ; lire le résultat via `getSessionMessages`
  (dernier message `role:"assistant"`).
- **Disponibilité au dispatch = allowlist GLOBALE** `tools.sandbox.tools.allow`
  (l'allow par-agent de `main` ne suffit pas).
- **Outil d'orchestration à `deny` pour chaque sous-agent** (sinon le sous-agent
  appelle le tool au lieu de faire son travail → boucle/timeout). Ex. :
  `agents.list[wiki].tools.sandbox.tools.deny:["<tool>"]`.

## Références
- `docs/architecture/diagnostic-et-observabilite.md` — analyse logs/sous-agents.
- `docs/architecture/corpus-intentions-seed.md` — corpus pour le classifieur.
- Mémoire : `project-intention-architecture`, `project-slm-to-prod-roadmap`.
