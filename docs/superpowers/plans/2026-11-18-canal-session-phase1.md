# Canal Session — phase 1 — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** converser avec Tiron, taper toutes les commandes et envoyer des pièces jointes
depuis Session, Telegram restant intact.

**Architecture :** un démon Bun `session-broker` (transport Session.js, 127.0.0.1, jeton)
et un plugin de canal OpenClaw `session` qui s'y abonne en SSE, transmet le texte à `main`
et dépose les pièces jointes dans `Arbath/Réception/`. `derisk-deleg` répond en texte brut
sur ce canal.

**Tech Stack :** Bun 1.2.4 + `@session.js/client` 0.0.57 + `@session.js/bun-network` 1.0.15
+ `bun:sqlite` + `bun test` ; plugin TypeScript/Node 24 + SDK `openclaw` 2026.8.2 + vitest ;
`derisk-deleg` (vitest) ; `openclaw-config` (pytest).

**Spec :** `docs/superpowers/specs/2026-11-18-canal-session-design.md` (phase 1 = § 6.1).

## Global Constraints

- Phrase secrète : variable `SESSION_BOT_MNEMONIC`, **seulement** dans `~/.config/secrets.env`.
  Jamais dans un dépôt, un journal, une sortie de commande, l'environnement du gateway ou un
  conteneur.
- Jeton broker ↔ plugin : `TIRON_SESSION_BROKER_TOKEN` (préfixe `TIRON_` = transmis au
  gateway par `/maj`). Broker sur `127.0.0.1` seulement ; toute route exige
  `Authorization: Bearer <jeton>`, `/health` compris.
- Liste blanche `allowFrom` (identifiants Session complets, 66 caractères, préfixe `05`) :
  au départ celui de l'utilisateur seul. Hors liste : ignoré, sans réponse ni accusé de
  lecture, journalisé avec 8 caractères. Le broker refuse d'envoyer hors liste.
- Journaux : jamais le texte d'un message (longueur seulement), identifiants tronqués à 8.
- Versions épinglées exactes (`0.0.57`, `1.0.15`, `openclaw` `2026.8.2`), fichiers de
  verrouillage commités.
- Pièces jointes reçues : `~/Documents/Arbath/Réception/` (variable `TIRON_RECEPTION_DIR`
  si définie, comme `tiron_cron/livraison.py`).
- Telegram : aucune modification de son comportement.
- `systemctl --user start/stop/enable/restart`, `openclaw plugins install`, édition de
  `~/.openclaw/openclaw.json` : **demander confirmation à l'utilisateur** avant.
- `git add` avec chemins explicites ; jamais `Wiki_LM/.env.bak-*` ni
  `gen_corpus/laya_train_chargédnasKaggle.jsonl`.

## Review Focus

1. Expéditeur hors liste blanche avec pièce jointe : rien n'est écrit sur disque, aucun
   accusé de lecture, rien transmis — test en tâche 2.
2. Gateway redémarré pendant que des messages arrivent : les événements non acquittés sont
   rejoués, et le plugin ne les traite qu'une fois — tests en tâches 2 et 6.
3. Nom de pièce jointe hostile (`../../.bashrc`, `/etc/passwd`, vide, 300 caractères) :
   reste dans `Réception/`, n'écrase rien — test en tâche 6.
4. Broker injoignable au démarrage du gateway : le compte réessaie avec délai croissant
   sans faire tomber le gateway, et s'arrête proprement sur `abortSignal` — test en tâche 6.
5. Réponse de Tiron contenant `&`, `<b>` : arrive telle quelle sur Session (pas `&amp;`) —
   test en tâche 7.

## Structure des fichiers

```
session-broker/                 # démon Bun (nouveau)
  package.json, bun.lock, tsconfig.json
  src/config.ts                 # chargement config + jeton
  src/http.ts                   # serveur HTTP (routes, auth, SSE)
  src/inbound.ts                # traitement d'un message Session entrant
  src/outbound.ts               # envoi, typing, réessais
  src/store.ts                  # SQLite : dédoublonnage + file non acquittée
  src/init.ts                   # commande init (identité)
  src/main.ts                   # assemblage (Session.js réel)
  test/*.test.ts
  bin/session-broker            # enveloppe : extrait 2 clés de secrets.env
  systemd/session-broker.service
openclaw-session/               # plugin de canal (nouveau)
  package.json, package-lock.json, tsconfig.json, openclaw.plugin.json
  index.ts, setup-entry.ts
  src/channel.ts                # ChannelPlugin
  src/broker-client.ts          # client HTTP/SSE du broker
  src/monitor.ts                # boucle startAccount, aiguillage
  src/depot.ts                  # dépôt des pièces jointes
  src/*.test.ts
derisk-deleg/src/wiki-ops.ts, index.ts, aide.ts, jeton-ops.ts   # modifiés
openclaw-config/openclaw.json.template, profil_config.py, test_profil_config.py  # modifiés
```

---

### Task 1: broker — configuration, authentification, santé

**Files:**
- Create: `session-broker/package.json`, `bun.lock`, `tsconfig.json`, `src/config.ts`,
  `src/http.ts`, `test/http.test.ts`, `test/config.test.ts`

**Interfaces:**
- Produces:
  - `type BrokerConfig = { port: number; allowFrom: string[]; stateDir: string; token: string }`
  - `loadConfig(path: string, env: Record<string,string|undefined>): BrokerConfig` — lit
    `~/.config/session-broker/config.json` (`port`, `allowFrom`, `stateDir` facultatif,
    défaut `~/.local/state/session-broker`) ; `token` = `env.TIRON_SESSION_BROKER_TOKEN`.
    Lève si jeton absent, `allowFrom` vide, ou entrée ne correspondant pas à `/^05[0-9a-f]{64}$/`.
  - `createServer(deps: HttpDeps): { fetch(req: Request): Promise<Response> }` (handler
    `Bun.serve`) ; `HttpDeps` est complété par les tâches 2 et 3. Tâche 1 : route
    `GET /health` → `{ ok: true, lastPollOkAt, lastSendOkAt, sessionId }`.

- [ ] **Step 1:** `bun init`, puis `bun add @session.js/client@0.0.57 @session.js/bun-network@1.0.15 @session.js/keypair@1.0.8 @session.js/mnemonic@1.0.6` (versions exactes dans `package.json`, sans `^`).
- [ ] **Step 2: tests qui échouent**
  - `config.test.ts` : `loadConfig` lève sans jeton ; lève sur `allowFrom: []` ; lève sur
    `"05abc"` ; rend `stateDir` par défaut.
  - `http.test.ts` : `GET /health` sans en-tête → 401 ; mauvais jeton → 401 ; bon jeton →
    200 avec `ok: true` ; route inconnue → 404.
- [ ] **Step 3:** `cd session-broker && bun test` → FAIL (modules absents).
- [ ] **Step 4:** implémenter `config.ts`, `http.ts` (comparaison de jeton à temps constant).
- [ ] **Step 5:** `bun test` → PASS.
- [ ] **Step 6:** commit `feat(session-broker): config, auth et /health`.

### Task 2: broker — réception (liste blanche, dédoublonnage, pièces jointes, SSE)

**Files:**
- Create: `src/store.ts`, `src/inbound.ts`, `test/inbound.test.ts`, `test/sse.test.ts`
- Modify: `src/http.ts`

**Interfaces:**
- Consumes: `BrokerConfig`, `createServer`.
- Produces:
  - `type InboundEvent = { id: string; from: string; timestamp: number; text: string; attachments: { path: string; name: string; contentType: string; size: number; error?: "trop-grande" }[] }` — une pièce de plus de `maxAttachmentBytes` (config, défaut 10 Mo) n'est pas écrite : `path: ""`, `error: "trop-grande"`.
  - `interface SessionLike { markMessageAsRead(p: { from: string; timestamp: number }): Promise<void>; getFile(att: unknown): Promise<File>; sendMessage(p: { to: string; text: string }): Promise<{ messageHash: string; timestamp: number }>; showTypingIndicator?(p:{conversation:string}):Promise<void> }` — adapter les noms aux méthodes réelles de 0.0.57 (`node_modules/@session.js/client/dist/instance/*.d.ts`) et le noter dans le ledger.
  - `class Store` (`bun:sqlite`, `<stateDir>/broker.sqlite`) : `seen(id): boolean`,
    `enqueue(evt: InboundEvent)`, `pending(): InboundEvent[]`, `ack(id)`.
  - `handleSessionMessage(msg, deps: { session: SessionLike; store: Store; cfg: BrokerConfig; publish(evt: InboundEvent): void; log(line: string): void }): Promise<"dropped" | "duplicate" | "queued">`
  - Routes : `GET /events` (SSE : à la connexion, rejoue `pending()`, puis diffuse ;
    `event: message`, `data: <InboundEvent JSON>`) ; `POST /ack {id}` → 204.

- [ ] **Step 1: tests qui échouent** (`inbound.test.ts`, faux `SessionLike`)
  - `expéditeur hors liste → "dropped"`, `publish` non appelé, `markMessageAsRead` non
    appelé, `getFile` non appelé, aucun fichier dans `<stateDir>/media`, journal sans le texte.
  - `même id deux fois → "queued" puis "duplicate"`.
  - `pièce jointe autorisée → fichier dans <stateDir>/media (mode 0600, dossier 0700), path dans l'événement, markMessageAsRead appelé une fois`.
  - `texte vide + pièce jointe → queued, text ""`.
  - `pièce de 11 Mo → aucun fichier écrit, error "trop-grande" dans l'événement`.
  - `sse.test.ts` : événement en file, client SSE connecté → le reçoit ; sans `/ack`,
    reconnexion → le reçoit à nouveau ; après `/ack` → plus rejoué. `/events` sans jeton → 401.
- [ ] **Step 2:** `bun test` → FAIL.
- [ ] **Step 3:** implémenter `store.ts`, `inbound.ts`, routes SSE/ack.
- [ ] **Step 4:** `bun test` → PASS.
- [ ] **Step 5:** commit `feat(session-broker): réception, liste blanche, SSE avec rejeu`.

### Task 3: broker — envoi et saisie

**Files:**
- Create: `src/outbound.ts`, `test/outbound.test.ts`
- Modify: `src/http.ts`

**Interfaces:**
- Consumes: `SessionLike`, `BrokerConfig`.
- Produces:
  - `sendText(to: string, text: string, deps: { session: SessionLike; cfg: BrokerConfig; sleep?(ms:number):Promise<void>; log(l:string):void }): Promise<{ messageId: string; timestamp: number }>` — refuse (`NotAllowedError`) si `to` hors `allowFrom` ; 3 essais (délais 1 s, 3 s) puis lève.
  - Routes : `POST /send {to,text}` → 200 `{messageId,timestamp}` | 403 hors liste | 502
    échec après essais ; `POST /typing {to,on}` → 204 (403 hors liste).
  - Met à jour `lastSendOkAt` exposé par `/health`.

- [ ] **Step 1: tests qui échouent** : `/send` vers un identifiant hors liste → 403 et
  `sendMessage` jamais appelé ; deux échecs puis succès → 200, 3 appels ; trois échecs → 502 ;
  journal ne contient pas le texte.
- [ ] **Step 2:** `bun test` → FAIL.
- [ ] **Step 3:** implémenter.
- [ ] **Step 4:** `bun test` → PASS.
- [ ] **Step 5:** commit `feat(session-broker): envoi avec réessais, saisie`.

### Task 4: broker — identité, assemblage, service

**Files:**
- Create: `src/init.ts`, `src/main.ts`, `test/init.test.ts`, `bin/session-broker`,
  `systemd/session-broker.service`, `README.md` (installation en 6 lignes)

**Interfaces:**
- Produces:
  - `initIdentity(secretsPath: string, gen = () => encode(generateSeedHex())): { sessionId: string }` — refuse si `SESSION_BOT_MNEMONIC=` déjà présent ; ajoute `SESSION_BOT_MNEMONIC=…` et `TIRON_SESSION_BROKER_TOKEN=<32 octets hex>` ; conserve le mode du fichier (0600 si créé).
  - CLI : `bun src/main.ts init` (n'imprime que l'identifiant Session) ; `bun src/main.ts serve`.
  - `serve` : `await ready`, `Session` réel, `setMnemonic(m, "Tiron")`, `Poller`,
    `on("message", m => handleSessionMessage(...))`, `acceptConversationRequest` pour un
    expéditeur autorisé si la méthode l'exige (à vérifier en essai réel), `Bun.serve({ hostname: "127.0.0.1", port, fetch })`, `lastPollOkAt` mis à jour à chaque cycle de poll réussi.
  - `bin/session-broker` (bash) : lit **uniquement** `SESSION_BOT_MNEMONIC` et
    `TIRON_SESSION_BROKER_TOKEN` dans `~/.config/secrets.env` (`grep '^CLÉ=' | cut -d= -f2-`),
    les exporte, `exec ~/.bun/bin/bun <dépôt>/session-broker/src/main.ts serve`.
  - Unité utilisateur : `ExecStart=%h/Secretarius/session-broker/bin/session-broker`,
    `Restart=on-failure`, `RestartSec=10`, `NoNewPrivileges=yes`.

- [ ] **Step 1: tests qui échouent** (`init.test.ts`, fichier temporaire, générateur
  injecté) : ajoute les deux clés ; second appel lève et ne modifie pas le fichier ; la
  sortie (`sessionId`) ne contient pas la phrase secrète ; mode 0600 conservé.
- [ ] **Step 2:** `bun test` → FAIL. **Step 3:** implémenter. **Step 4:** `bun test` → PASS.
- [ ] **Step 5:** vérif. statique : `grep -n "secrets.env" bin/session-broker` ne montre que
  deux `grep '^SESSION_BOT_MNEMONIC='` / `grep '^TIRON_SESSION_BROKER_TOKEN='`.
- [ ] **Step 6:** commit `feat(session-broker): init d'identité, assemblage, unité systemd`.
- [ ] **Step 7 (avec l'utilisateur) :** `bun src/main.ts init` ; l'utilisateur écrit
  `~/.config/session-broker/config.json` avec son identifiant ; **après accord**, installer
  l'unité et `systemctl --user enable --now session-broker` ; `curl -s -H "Authorization: Bearer $(grep '^TIRON_SESSION_BROKER_TOKEN=' ~/.config/secrets.env | cut -d= -f2-)" 127.0.0.1:<port>/health` → `ok: true`. L'utilisateur écrit au bot : son message passe à « lu ».

### Task 5: plugin — paquet, compte, sécurité, client du broker

**Files:**
- Create: `openclaw-session/package.json`, `package-lock.json`, `tsconfig.json`,
  `openclaw.plugin.json`, `index.ts`, `setup-entry.ts`, `src/channel.ts`,
  `src/broker-client.ts`, `src/channel.test.ts`, `src/broker-client.test.ts`

**Interfaces:**
- Consumes: routes des tâches 1-3.
- Produces:
  - Config `channels.session` (schéma dans `openclaw.plugin.json`, `additionalProperties: false`) :
    `{ enabled: boolean, brokerUrl: string, dmPolicy: "allowlist", allowFrom: string[] }`.
  - `resolveSessionAccount(cfg, accountId?): { accountId: "default"; brokerUrl: string; allowFrom: string[]; dmPolicy: "allowlist"; token: string; enabled: boolean; configured: boolean }` — `token` lu dans `process.env.TIRON_SESSION_BROKER_TOKEN`.
  - `inspectAccount` sans lire le jeton (statut `available|missing` seulement).
  - `class BrokerClient(url, token, fetchImpl?)` : `events(signal): AsyncIterable<InboundEvent>`
    (parse SSE), `ack(id)`, `send(to, text): Promise<{messageId:string}>`, `typing(to, on)`,
    `health()`.
  - `sessionPlugin` via `createChatChannelPlugin` (`openclaw/plugin-sdk/channel-core`), id
    `session`, `security.dm` = `{ channelKey: "session", defaultPolicy: "allowlist", resolveAllowFrom: a => a.allowFrom }`, pas de `pairing`, `outbound.attachedResults.sendText` → `BrokerClient.send`, `textChunkLimit` 4000 (à vérifier contre Session en tâche 8), pas de capacités live. Modèle : `@openclaw/signal@2026.8.2`, `dist/channel-*.js` (objet `signalPlugin`).
  - `package.json` : `devDependencies.openclaw` = `2026.8.2` exact ; `peerDependencies.openclaw` = `>=2026.8.2`.

- [ ] **Step 1: tests qui échouent** : `resolveSessionAccount` lit `allowFrom` ;
  `inspectAccount` rend `configured:false` sans `brokerUrl` et n'expose jamais le jeton ;
  `BrokerClient` (serveur HTTP local de test) envoie `Authorization: Bearer`, parse deux
  événements SSE, remonte 403 de `/send` en erreur typée.
- [ ] **Step 2:** `npx vitest run` → FAIL. **Step 3:** implémenter. **Step 4:** PASS.
- [ ] **Step 5:** `npm run build` puis `openclaw plugins validate --entry ./dist/index.js` → valide.
- [ ] **Step 6:** commit `feat(openclaw-session): plugin de canal, compte, client du broker`.

### Task 6: plugin — boucle de réception, aiguillage, dépôt

**Files:**
- Create: `src/monitor.ts`, `src/depot.ts`, `src/monitor.test.ts`, `src/depot.test.ts`
- Modify: `src/channel.ts` (ajout `gateway.startAccount`, `heartbeat.sendTyping/clearTyping`)

**Interfaces:**
- Consumes: `BrokerClient`, `InboundEvent`, `resolveSessionAccount`.
- Produces:
  - `deposerPiecesJointes(evt: InboundEvent, dir: string, now: Date): { note: string; fichiers: string[] }` — nom = `YYYY-MM-DD HHmm <nom assaini>` (`path.basename`, caractères `/\\:\0` et contrôles retirés, 120 caractères max, `piece-jointe` si vide) ; suffixe ` (2)`, ` (3)`… si existe ; note `YYYY-MM-DD HHmm Session.md` avec en-tête YAML `origine: session`, `date`, `expediteur: <8 car.>`, le texte du message, puis `![[<fichier>]]` par pièce ; déplace (`rename`, sinon copie + suppression) depuis le dossier du broker.
  - `traiterEvenement(evt, deps: { allowFrom: string[]; seen: Set<string>; depot: typeof deposerPiecesJointes; receptionDir: string; dispatchTexte(evt): Promise<void>; repondre(to: string, text: string): Promise<void>; ack(id: string): Promise<void> }): Promise<"ignore"|"doublon"|"depot"|"texte">` — défense en profondeur : hors `allowFrom` → `"ignore"` (ack quand même, rien d'autre) ; `id` déjà vu → `"doublon"` + ack ; pièces jointes → dépôt, réponse `Reçu : <n> pièce(s) jointe(s) déposée(s) dans Réception/`, ack, **pas** de `dispatchTexte` (une pièce `error: "trop-grande"` n'est pas déposée et la réponse ajoute `<n> pièce(s) trop grande(s), non reçue(s)`) ; sinon `dispatchTexte` puis ack.
  - `startAccount(ctx)` : boucle `for await (evt of client.events(ctx.abortSignal))` ; sur
    erreur de connexion, délai croissant (1 s → 60 s max) avec `sleepWithAbort` ; sortie
    propre sur abort ; statut via `createAccountStatusSink`.
  - `dispatchTexte` : construire le contexte et appeler `runChannelInboundEvent` comme le
    plugin Signal (`dist/monitor-*.js`, l. ~640-1000 : `resolveAgentRoute`,
    `buildChannelInboundEventContext`, `createChannelMessageReplyPipeline` avec `typing.start` →
    `client.typing`, `delivery.deliver` → `client.send`, `record.updateLastRoute`). Messages
    directs seulement : pas de groupes, réactions, historique de groupe, approbations natives.
    Le canal doit apparaître comme `messageProvider = "session"` dans les crochets, et
    `access.commands.authorized` vrai pour un expéditeur de `allowFrom`.

- [ ] **Step 1: tests qui échouent**
  - `depot.test.ts` : noms `../../.bashrc`, `/etc/passwd`, `""`, 300 caractères → fichiers
    tous sous `dir`, aucun écrasement (deux dépôts du même nom → ` (2)`) ; note contient le
    texte et un `![[…]]` par fichier ; `TIRON_RECEPTION_DIR` respecté par l'appelant.
  - `monitor.test.ts` : expéditeur hors liste → `"ignore"`, ni dépôt, ni dispatch, ni
    réponse ; même `id` rejoué → `"doublon"`, dispatch une seule fois ; pièce jointe →
    `"depot"`, réponse exacte, pas de dispatch ; texte → `"texte"` ; boucle : `events` qui
    lève deux fois puis rend un événement → traité, délais croissants ; abort → la boucle
    rend la main.
- [ ] **Step 2:** `npx vitest run` → FAIL. **Step 3:** implémenter. **Step 4:** PASS.
- [ ] **Step 5:** `npm run build && openclaw plugins validate --entry ./dist/index.js`.
- [ ] **Step 6:** commit `feat(openclaw-session): réception, dépôt des pièces jointes`.

### Task 7: derisk-deleg — texte brut sur Session

**Files:**
- Modify: `derisk-deleg/src/wiki-ops.ts` (type `WikiOpRegime`, `formatWikiResult`),
  `derisk-deleg/src/index.ts:778`, `derisk-deleg/src/aide.ts`, `derisk-deleg/src/jeton-ops.ts:2-3`
- Test: `derisk-deleg/src/wiki-ops.test.ts`, `index.test.ts`, `aide.test.ts`

**Interfaces:**
- Produces: `WikiOpRegime = "brief" | "full" | "plain"` ; `"plain"` = même contenu que
  `"brief"` sans `escapeHtml`. `index.ts` : `messageProvider === "webchat"` → `"full"`,
  `=== "session"` → `"plain"`, sinon `"brief"`.

- [ ] **Step 1: tests qui échouent** : `formatWikiResult("query", {brief: "a & <b>", history_path: "p"}, "plain")` → `"a & <b>\n\np"` ; le même en `"brief"` reste échappé (test existant) ; crochet avec `messageProvider: "session"` sur `/q` → `runWikiOp` appelé avec `"plain"` ; `AIDE` contient `Session`.
- [ ] **Step 2:** `cd derisk-deleg && npx vitest run` → FAIL. **Step 3:** implémenter ;
  `aide.ts` : titre « Commandes Secretarius (Telegram, Session) », test « tient dans un
  message » conservé ; commentaire de `jeton-ops.ts` : « canal direct (Telegram, Session) ».
- [ ] **Step 4:** `npx vitest run` → PASS (toute la suite) ; `npm run build`.
- [ ] **Step 5:** commit `feat(derisk-deleg): texte brut sur le canal Session`.

### Task 8: configuration, installation, essai réel

**Files:**
- Modify: `openclaw-config/openclaw.json.template`, `openclaw-config/profil_config.py`
  (si les canaux y sont générés), `openclaw-config/test_profil_config.py`

**Interfaces:**
- Produces (modèle et config vivante) : `channels.session = { enabled: true, brokerUrl: "http://127.0.0.1:<port>", dmPolicy: "allowlist", allowFrom: ["<id utilisateur>"] }` (dans le modèle : variable de substitution, comme `TELEGRAM_OWNER_ID`) ; binding `{ agentId: "main", match: { channel: "session", accountId: "default" } }` ; `commands.ownerAllowFrom` += `"session:<id>"` ; plugin `openclaw-session` autorisé comme `derisk-deleg`.

- [ ] **Step 1: test qui échoue** `test_canal_session_lie_a_main` : le modèle rendu
  contient le binding session→main, `channels.session.dmPolicy == "allowlist"`, et un
  `ownerAllowFrom` commençant par `session:` ; Telegram inchangé.
- [ ] **Step 2:** `python -m pytest openclaw-config -q` → FAIL. **Step 3:** modifier.
  **Step 4:** PASS. **Step 5:** commit `feat(config): canal Session lié à main`.
- [ ] **Step 6 (avec accord) :** sauvegarde `~/.openclaw/openclaw.json` →
  `openclaw.json.avant-session` ; édition vivante ; `/maj` (ajoute
  `TIRON_SESSION_BROKER_TOKEN` au gateway) ; `openclaw plugins install ~/Secretarius/openclaw-session --force --accept-capabilities` ;
  `openclaw plugins install ~/Secretarius/derisk-deleg --force --accept-capabilities` ;
  redémarrage `openclaw-gateway`.
- [ ] **Step 7 : vérifs avant l'utilisateur** : `openclaw gateway call channels.status --params '{"probe":true}'` montre `session` connecté ; `grep -c SESSION_BOT_MNEMONIC ~/.openclaw/gateway.systemd.env` → `0` ; aucun journal (`journalctl --user -u session-broker`, `/tmp/openclaw/openclaw-*.log`) ne contient la phrase secrète ni le texte d'un message d'essai.
- [ ] **Step 8 : essai de l'utilisateur, téléphone hors Tailscale (données mobiles, Tailscale coupé)** :
  `/aide` ; `/c une idée` ; `/c <url>` ; `/q …` ; `/source https://example.org` ; une action
  à `/confirm` ; une photo avec légende → `Arbath/Réception/` (fichier + note) et réponse
  « Reçu : 1 pièce(s) jointe(s)… » ; Telegram fonctionne toujours. Puis un **second compte
  Session** écrit au bot → aucune réponse, message non lu, ligne « ignoré » dans le journal.
  Mesurer la longueur maximale acceptée par Session et ajuster `textChunkLimit` si besoin.
- [ ] **Step 9:** fiche mémoire datée (état, commits, reste : phases 2 et 3).
