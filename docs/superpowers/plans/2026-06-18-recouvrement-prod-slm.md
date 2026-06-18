# Recouvrement prod → SLM — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Porter sur le SLM (commandes déterministes) 7 fonctionnalités prod manquantes (`/connecter /chercher /lire /drive /repondre /tags /kbupdate`), précédées de l'isolation de gog dans une image `secretarius-gog`.

**Architecture:** Patron prouvé — skill `command-dispatch:tool` → outil du plugin `derisk-deleg` → `api.runtime.subagent.run` vers l'agent dédié (gog ou wiki) + op dans son `AGENTS.md` + allow global / deny sous-agents. gog isolé dans sa propre image. wiki sans rebuild (tools/ en bind ro). `/connecter` médie le flux OAuth interactif via un pont file-drop.

**Tech Stack:** TypeScript (plugin OpenClaw, build tsc, tests vitest), Python 3 (wiki.py CLI), shell (bridge OAuth), Docker (image gog), JSON (openclaw.json), Markdown (skills, AGENTS.md).

**Référence spec :** `docs/superpowers/specs/2026-06-18-recouvrement-prod-slm-design.md`

---

## Conventions de vérification

- **Tests unitaires** (vitest) : seulement pour la logique pure (`src/parse.ts`).
- **Vérification E2E** : via Telegram, **en session neuve** (`/new`) à chaque fois (biais de session SLM). Le bot SLM est `@secretarius_tiron_bot`.
- **Cycle de déploiement plugin** (après toute modif de `src/`) :
  ```bash
  cd ~/secretarius-plugin-spike/derisk-deleg && npm run build \
    && openclaw --profile slm plugins install . --force \
    && systemctl --user restart openclaw-gateway-slm
  ```
- **Édition config** : `~/.openclaw-slm/openclaw.json` (faire un backup `.bak-<motif>` avant chaque édition), suivie d'un `systemctl --user restart openclaw-gateway-slm`.

## File Structure

| Fichier | Rôle | Action |
|---------|------|--------|
| `~/Secretarius/openclaw-config/Dockerfile.gog` | image dédiée gog (base + gog + bridge) | créer |
| `~/Secretarius/openclaw-config/gog-auth-bridge.sh` | pont OAuth `--manual` | créer |
| `~/Secretarius/openclaw-config/Dockerfile.tiron` | image main, réduite à la base nue | modifier |
| `~/secretarius-plugin-spike/derisk-deleg/src/parse.ts` | helpers purs (`parseReply`) | créer |
| `~/secretarius-plugin-spike/derisk-deleg/src/parse.test.ts` | tests des helpers | créer |
| `~/secretarius-plugin-spike/derisk-deleg/src/index.ts` | tools + hooks + état pending | modifier |
| `~/Secretarius/Wiki_LM/tools/wiki.py` | ops CLI `tags`, `kb_update`, `_kb_update_worker` | modifier |
| `~/.openclaw-slm/workspace-gog/AGENTS.md` | ops gog `get`/`drive_search`/`reply`/`auth_start` | modifier |
| `~/.openclaw-slm/workspace-wiki/AGENTS.md` | ops wiki `tags`/`kb_update` | modifier |
| `~/.openclaw-slm/workspace/skills/{connecter,chercher,lire,drive,repondre,tags,kbupdate}/SKILL.md` | 7 skills | créer |
| `~/.openclaw-slm/openclaw.json` | image gog, allow/deny des 7 outils | modifier |

---

# Phase 0 — Image dédiée `secretarius-gog`

### Task 0.1 : Créer `Dockerfile.gog` et le bridge OAuth

**Files:**
- Create: `~/Secretarius/openclaw-config/Dockerfile.gog`
- Create: `~/Secretarius/openclaw-config/gog-auth-bridge.sh`

- [ ] **Step 1 : Écrire le pont OAuth**

`~/Secretarius/openclaw-config/gog-auth-bridge.sh` :
```sh
#!/bin/sh
# Pont OAuth pour /connecter : lance `gog auth add --manual`, expose l'URL
# d'autorisation dans un fichier, attend l'URL de redirection recollée par
# l'utilisateur, l'injecte sur stdin de gog, puis signale la fin.
# Tout passe par $XDG_CONFIG_HOME (bind /gog-config), lu/écrit par le plugin.
set -e
CFG="${XDG_CONFIG_HOME:-/gog-config}"
EMAIL="${1:?usage: gog-auth-bridge.sh <email>}"
RAW="$CFG/auth_raw"; URL="$CFG/auth_url"; RESP="$CFG/auth_response"; DONE="$CFG/auth_done"
rm -f "$RAW" "$URL" "$RESP" "$DONE"
FIFO="$CFG/.auth_fifo"; rm -f "$FIFO"; mkfifo "$FIFO"

# gog lit l'URL de redirection sur stdin (FIFO), écrit ses prompts dans RAW.
gog auth add "$EMAIL" --manual --force-consent \
  --services gmail,drive,calendar < "$FIFO" > "$RAW" 2>&1 &
GOGPID=$!

# Attendre l'URL d'autorisation Google, puis l'exposer.
i=0
while ! grep -qE 'https://accounts\.google\.com/[^ ]+' "$RAW" 2>/dev/null; do
  sleep 0.5; i=$((i+1)); [ $i -gt 120 ] && { echo "timeout url" > "$DONE"; kill $GOGPID 2>/dev/null; exit 1; }
done
grep -oE 'https://accounts\.google\.com/[^ ]+' "$RAW" | head -1 > "$URL"

# Attendre la réponse de l'utilisateur (déposée par le plugin), puis l'injecter.
i=0
while [ ! -f "$RESP" ]; do
  sleep 1; i=$((i+1)); [ $i -gt 600 ] && { echo "timeout response" > "$DONE"; kill $GOGPID 2>/dev/null; exit 1; }
done
cat "$RESP" > "$FIFO"
if wait $GOGPID; then echo ok > "$DONE"; else echo error > "$DONE"; fi
rm -f "$FIFO"
```

> NOTE pour l'implémenteur : le format exact de sortie de `gog auth add --manual`
> (où apparaît l'URL, ce qu'attend stdin) est à **caractériser en Task 1.7 avant**
> de finaliser ce script. Lancer une caractérisation contrôlée et ajuster les
> motifs `grep` si besoin. Ne pas régénérer le vrai token pendant la caractérisation
> (tuer le process après lecture de l'URL, sans rien coller).

- [ ] **Step 2 : Écrire `Dockerfile.gog`**

`~/Secretarius/openclaw-config/Dockerfile.gog` :
```dockerfile
FROM openclaw-sandbox:bookworm-slim
COPY --chmod=755 gog-bin /usr/local/bin/gog-bin
COPY --chmod=755 openclaw-config/gog-wrapper.sh /usr/local/bin/gog
COPY --chmod=755 openclaw-config/gog-auth-bridge.sh /usr/local/bin/gog-auth-bridge
```

- [ ] **Step 3 : Vérifier le contexte de build (présence de `gog-bin`)**

Run: `ls -la ~/Secretarius/gog-bin ~/Secretarius/openclaw-config/gog-wrapper.sh`
Expected : les deux fichiers existent (contexte de build = `~/Secretarius`, comme pour Dockerfile.tiron). Si `gog-bin` est ailleurs, ajuster le `COPY` en conséquence.

- [ ] **Step 4 : Commit**

```bash
cd ~/Secretarius && git add openclaw-config/Dockerfile.gog openclaw-config/gog-auth-bridge.sh \
  && git commit -m "feat(image): Dockerfile.gog + pont OAuth gog-auth-bridge.sh"
```

### Task 0.2 : Réduire `Dockerfile.tiron` à la base nue et builder les images

**Files:**
- Modify: `~/Secretarius/openclaw-config/Dockerfile.tiron`

- [ ] **Step 1 : Réduire `Dockerfile.tiron`**

Remplacer tout le contenu par :
```dockerfile
FROM openclaw-sandbox:bookworm-slim
```

- [ ] **Step 2 : Builder les deux images**

Run :
```bash
cd ~/Secretarius
docker build -f openclaw-config/Dockerfile.gog   -t secretarius-gog:latest   .
docker build -f openclaw-config/Dockerfile.tiron -t secretarius-tiron:latest .
```
Expected : deux builds réussis.

- [ ] **Step 3 : Vérifier l'isolation**

Run :
```bash
docker run --rm secretarius-gog:latest   sh -c 'command -v gog gog-auth-bridge'
docker run --rm secretarius-tiron:latest sh -c 'command -v gog || echo "gog absent (attendu)"'
```
Expected : l'image gog liste `/usr/local/bin/gog` et `/usr/local/bin/gog-auth-bridge` ; l'image tiron affiche « gog absent (attendu) ».

- [ ] **Step 4 : Commit**

```bash
cd ~/Secretarius && git add openclaw-config/Dockerfile.tiron \
  && git commit -m "feat(image): tiron réduit à la base nue (gog isolé hors de main)"
```

### Task 0.3 : Basculer l'agent gog sur la nouvelle image

**Files:**
- Modify: `~/.openclaw-slm/openclaw.json`

- [ ] **Step 1 : Backup**

Run: `cp ~/.openclaw-slm/openclaw.json ~/.openclaw-slm/openclaw.json.bak-gog-image`

- [ ] **Step 2 : Changer l'image de l'agent gog**

Dans `~/.openclaw-slm/openclaw.json`, agent d'`id: "gog"`, sous `sandbox.docker.image` :
remplacer `"secretarius-tiron:latest"` par `"secretarius-gog:latest"`.
(Laisser l'agent `main` sur `secretarius-tiron:latest`.)

- [ ] **Step 3 : Restart**

Run: `systemctl --user restart openclaw-gateway-slm && sleep 3 && systemctl --user is-active openclaw-gateway-slm`
Expected : `active`.

- [ ] **Step 4 : Vérification E2E**

Telegram (`/new`) : `/inbox`
Expected : liste réelle des emails (l'agent gog tourne sur `secretarius-gog`, creds montés OK).

- [ ] **Step 5 : Commit (config versionnée si applicable)**

Si la config est versionnée dans un dépôt, committer ; sinon, le backup `.bak-gog-image` suffit comme point de retour. Noter la bascule.

---

# Phase 1 — Commandes gog

### Task 1.1 : Helper pur `parseReply` (TDD)

**Files:**
- Create: `~/secretarius-plugin-spike/derisk-deleg/src/parse.ts`
- Create: `~/secretarius-plugin-spike/derisk-deleg/src/parse.test.ts`

- [ ] **Step 1 : Écrire le test (échoue)**

`src/parse.test.ts` :
```ts
import { describe, it, expect } from "vitest";
import { parseReply } from "./parse";

describe("parseReply", () => {
  it("sépare l'id du corps", () => {
    expect(parseReply("18ab body de la réponse")).toEqual({
      messageId: "18ab",
      body: "body de la réponse",
    });
  });
  it("refuse sans corps", () => {
    expect(parseReply("18ab")).toBeNull();
    expect(parseReply("18ab   ")).toBeNull();
  });
  it("refuse une chaîne vide", () => {
    expect(parseReply("")).toBeNull();
  });
  it("conserve les espaces internes du corps", () => {
    expect(parseReply("X  deux  espaces")).toEqual({
      messageId: "X",
      body: "deux  espaces",
    });
  });
});
```

- [ ] **Step 2 : Lancer le test (échoue)**

Run: `cd ~/secretarius-plugin-spike/derisk-deleg && npx vitest run src/parse.test.ts`
Expected : FAIL (`Cannot find module './parse'`).

- [ ] **Step 3 : Implémenter `src/parse.ts`**

```ts
// Helpers purs (testables) pour les commandes du plugin derisk-deleg.

// /repondre <id> <texte> : premier token = id, reste = corps (espaces conservés).
export function parseReply(
  raw: string,
): { messageId: string; body: string } | null {
  const s = (raw ?? "").trim();
  const sp = s.indexOf(" ");
  if (sp < 1) return null;
  const messageId = s.slice(0, sp);
  const body = s.slice(sp + 1).trim();
  if (!messageId || !body) return null;
  return { messageId, body };
}
```

- [ ] **Step 4 : Lancer le test (passe)**

Run: `cd ~/secretarius-plugin-spike/derisk-deleg && npx vitest run src/parse.test.ts`
Expected : PASS (4 tests).

- [ ] **Step 5 : Commit**

```bash
cd ~/secretarius-plugin-spike/derisk-deleg && git add src/parse.ts src/parse.test.ts \
  && git commit -m "feat(plugin): parseReply helper + tests"
```

### Task 1.2 : Outils gog lecture (`gog_search`, `gog_get`, `gog_drive_search`)

**Files:**
- Modify: `~/secretarius-plugin-spike/derisk-deleg/src/index.ts`

- [ ] **Step 1 : Ajouter les trois outils**

Dans `register(api)`, après l'outil `gog_inbox` (vers la ligne 179), insérer :
```ts
    api.registerTool({
      name: "gog_search",
      description:
        "Rechercher des emails Gmail (délègue 'op: search' à l'agent gog).",
      parameters: Type.Object({
        command: Type.Optional(Type.String({ description: "Requête Gmail." })),
      }),
      async execute(_id: string, params: { command?: string }) {
        const q = (params?.command ?? "").trim();
        if (!q) return { content: [{ type: "text", text: "Usage: /chercher <requête>" }] };
        const out = await delegateGog(api, "search", q);
        return { content: [{ type: "text", text: out.slice(0, 1800) }] };
      },
    });

    api.registerTool({
      name: "gog_get",
      description:
        "Lire le contenu d'un email Gmail par son id (délègue 'op: get' à l'agent gog).",
      parameters: Type.Object({
        command: Type.Optional(Type.String({ description: "L'id du message." })),
      }),
      async execute(_id: string, params: { command?: string }) {
        const id = (params?.command ?? "").trim();
        if (!id) return { content: [{ type: "text", text: "Usage: /lire <id>" }] };
        const out = await delegateGog(api, "get", id);
        return { content: [{ type: "text", text: out.slice(0, 1800) }] };
      },
    });

    api.registerTool({
      name: "gog_drive_search",
      description:
        "Rechercher des fichiers Google Drive (délègue 'op: drive_search' à l'agent gog).",
      parameters: Type.Object({
        command: Type.Optional(Type.String({ description: "Requête Drive." })),
      }),
      async execute(_id: string, params: { command?: string }) {
        const q = (params?.command ?? "").trim();
        if (!q) return { content: [{ type: "text", text: "Usage: /drive <requête>" }] };
        const out = await delegateGog(api, "drive_search", q);
        return { content: [{ type: "text", text: out.slice(0, 1800) }] };
      },
    });
```

- [ ] **Step 2 : Build + validate**

Run: `cd ~/secretarius-plugin-spike/derisk-deleg && npm run build`
Expected : exit 0.

- [ ] **Step 3 : Commit**

```bash
git add src/index.ts && git commit -m "feat(plugin): outils gog_search/gog_get/gog_drive_search"
```

### Task 1.3 : Outil `gog_reply` + état `pendingSend` généralisé (send|reply)

**Files:**
- Modify: `~/secretarius-plugin-spike/derisk-deleg/src/index.ts`

- [ ] **Step 1 : Importer `parseReply` et généraliser l'état**

En tête de `index.ts`, après les imports existants :
```ts
import { parseReply } from "./parse";
```
Remplacer la déclaration de `pendingSend` (lignes ~76-78) par une union discriminée :
```ts
type Pending =
  | { kind: "send"; to: string; subject: string; body: string; ts: number }
  | { kind: "reply"; messageId: string; body: string; ts: number };
let pending: Pending | null = null;
```

- [ ] **Step 2 : Adapter `gog_send` au nouvel état**

Dans l'outil `gog_send`, remplacer l'affectation `pendingSend = {...}` par :
```ts
        pending = {
          kind: "send",
          to: params.to,
          subject: params.subject,
          body: params.body,
          ts: Date.now(),
        };
```

- [ ] **Step 3 : Ajouter l'outil `gog_reply`**

Après `gog_send` :
```ts
    api.registerTool({
      name: "gog_reply",
      description:
        "Prépare un brouillon de RÉPONSE à un email (n'envoie PAS). Indiquer ensuite à l'utilisateur de taper /confirm.",
      parameters: Type.Object({
        command: Type.Optional(
          Type.String({ description: "Args bruts : <id> <texte de la réponse>." }),
        ),
      }),
      async execute(_id: string, params: { command?: string }) {
        const parsed = parseReply(params?.command ?? "");
        if (!parsed) {
          return { content: [{ type: "text", text: "Usage: /repondre <id> <texte>" }] };
        }
        pending = { kind: "reply", messageId: parsed.messageId, body: parsed.body, ts: Date.now() };
        return {
          content: [
            {
              type: "text",
              text: `📧 Brouillon de réponse prêt (non envoyé) :\n• En réponse à : ${parsed.messageId}\n• Corps : ${parsed.body}\n\nTapez /confirm pour envoyer (valable 10 min), ou /annuler pour abandonner.`,
            },
          ],
        };
      },
    });
```

- [ ] **Step 4 : Adapter le hook `/confirm` et `/annuler`**

Remplacer le corps des branches `/confirm` et `/annuler` (lignes ~222-250) par une gestion selon `pending.kind` :
```ts
      if (cmd === "/confirm") {
        if (!pending) {
          return { handled: true, reply: { text: "Rien à confirmer (aucun brouillon en attente)." } };
        }
        if (Date.now() - pending.ts > PENDING_TTL_MS) {
          pending = null;
          return { handled: true, reply: { text: "Brouillon expiré (plus de 10 min) — rien envoyé. Recomposez si besoin." } };
        }
        const p = pending;
        pending = null;
        const out =
          p.kind === "send"
            ? await delegateGog(api, "send", `to=${p.to}; subject=${p.subject}; body=${p.body}`)
            : await delegateGog(api, "reply", `id=${p.messageId}; body=${p.body}`);
        return { handled: true, reply: { text: out.slice(0, 1800) } };
      }

      if (cmd === "/annuler") {
        if (!pending) {
          return { handled: true, reply: { text: "Rien à annuler (aucun brouillon en attente)." } };
        }
        const dest = pending.kind === "send" ? pending.to : `réponse à ${pending.messageId}`;
        pending = null;
        return {
          handled: true,
          reply: { text: `Brouillon abandonné (était destiné à ${dest}). Rien n'a été envoyé.` },
        };
      }
```

- [ ] **Step 5 : Étendre le garde-fou `before_tool_call`**

Le garde-fou bloque déjà `reply` (la regex `/(send|reply|forward|delete|…)/`). Vérifier seulement, aucune modif nécessaire (la branche `reply` côté agent gog est autorisée car `agentId === "gog"`).

- [ ] **Step 6 : Build**

Run: `cd ~/secretarius-plugin-spike/derisk-deleg && npm run build && npx vitest run`
Expected : build exit 0, tests PASS.

- [ ] **Step 7 : Commit**

```bash
git add src/index.ts && git commit -m "feat(plugin): gog_reply + état pending généralisé (send|reply)"
```

### Task 1.4 : Ops gog dans `AGENTS.md` (get, drive_search, reply)

**Files:**
- Modify: `~/.openclaw-slm/workspace-gog/AGENTS.md`

- [ ] **Step 1 : Étendre le tableau op → commande gog**

Dans le tableau de `workspace-gog/AGENTS.md`, ajouter les lignes :
```
| `get` | `gog gmail get <argument> --json` |
| `drive_search` | `gog drive search "<argument>" --max 10 --json` |
| `reply` | `gog gmail reply <id> --body <body>` (l'argument fournit `id=…; body=…`) |
```

- [ ] **Step 2 : Mettre à jour la note Contraintes**

Remplacer la ligne « Gmail : lecture (`inbox`/`search`) et envoi… » par :
```
- Gmail : lecture (`inbox`/`search`/`get`), envoi (`send`) et réponse (`reply`),
  send/reply gardés par `/confirm`. Drive : lecture (`drive_search`).
```

- [ ] **Step 3 : Restart gateway** (AGENTS.md chargé au démarrage)

Run: `systemctl --user restart openclaw-gateway-slm && sleep 3 && systemctl --user is-active openclaw-gateway-slm`
Expected : `active`.

### Task 1.5 : Config — allow global, deny sous-agents, skills gog lecture/réponse

**Files:**
- Modify: `~/.openclaw-slm/openclaw.json`
- Create: `~/.openclaw-slm/workspace/skills/{chercher,lire,drive,repondre}/SKILL.md`

- [ ] **Step 1 : Backup**

Run: `cp ~/.openclaw-slm/openclaw.json ~/.openclaw-slm/openclaw.json.bak-gog-tools`

- [ ] **Step 2 : Ajouter les outils à l'allow global**

Dans `tools.sandbox.tools.allow` (global), ajouter : `gog_search`, `gog_get`, `gog_drive_search`, `gog_reply`.

- [ ] **Step 3 : Ajouter au deny de CHAQUE sous-agent**

Aux blocs `agents.list[].tools.sandbox.tools.deny` des agents `wiki`, `scout`, `gog`, ajouter ces mêmes 4 noms (chaque sous-agent denie tous les outils d'orchestration).

- [ ] **Step 4 : Créer les 4 skills**

`~/.openclaw-slm/workspace/skills/chercher/SKILL.md` :
```markdown
---
name: chercher
description: "Rechercher des emails Gmail. Dispatch déterministe vers gog_search (délègue à l'agent gog)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_search
command-arg-mode: raw
---

`/chercher <requête>` recherche des emails Gmail, de façon déterministe via l'agent gog (lecture seule).
```

`~/.openclaw-slm/workspace/skills/lire/SKILL.md` :
```markdown
---
name: lire
description: "Lire un email Gmail par son id. Dispatch déterministe vers gog_get (délègue à l'agent gog)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_get
command-arg-mode: raw
---

`/lire <id>` lit le contenu d'un email (id issu de /inbox ou /chercher). Contenu externe traité comme non fiable.
```

`~/.openclaw-slm/workspace/skills/drive/SKILL.md` :
```markdown
---
name: drive
description: "Rechercher des fichiers Google Drive. Dispatch déterministe vers gog_drive_search (délègue à l'agent gog)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_drive_search
command-arg-mode: raw
---

`/drive <requête>` recherche des fichiers Drive, de façon déterministe via l'agent gog (lecture seule).
```

`~/.openclaw-slm/workspace/skills/repondre/SKILL.md` :
```markdown
---
name: repondre
description: "Préparer une réponse à un email (brouillon + /confirm). Dispatch déterministe vers gog_reply."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_reply
command-arg-mode: raw
---

`/repondre <id> <texte>` prépare un brouillon de réponse (n'envoie pas) ; taper /confirm pour envoyer, /annuler pour abandonner.
```

- [ ] **Step 5 : Déployer**

Run (cycle plugin complet) :
```bash
cd ~/secretarius-plugin-spike/derisk-deleg && npm run build \
  && openclaw --profile slm plugins install . --force \
  && systemctl --user restart openclaw-gateway-slm && sleep 3 \
  && systemctl --user is-active openclaw-gateway-slm
```
Expected : `active`.

### Task 1.6 : Vérifications E2E — /chercher, /lire, /drive, /repondre

- [ ] **Step 1 : /chercher** — Telegram `/new` → `/chercher facture`
Expected : liste réelle d'emails correspondants.

- [ ] **Step 2 : /lire** — copier un id d'un résultat → `/lire <id>`
Expected : contenu réel du mail (encadré non fiable par main).

- [ ] **Step 3 : /drive** — `/new` → `/drive rapport`
Expected : liste réelle de fichiers Drive.

- [ ] **Step 4 : /repondre + /annuler** — `/new` → `/repondre <id> Bonjour, bien reçu.` → attendre « Brouillon de réponse prêt » → `/annuler`
Expected : « Brouillon abandonné (était destiné à réponse à <id>). Rien n'a été envoyé. »

- [ ] **Step 5 : /repondre + /confirm** — refaire `/repondre <id> …` → `/confirm`
Expected : réponse réellement envoyée (vérifier dans le fil Gmail).

### Task 1.7 : `/connecter` — caractériser le flux puis implémenter le pont

**Files:**
- Modify: `~/Secretarius/openclaw-config/gog-auth-bridge.sh` (ajuster motifs si besoin)
- Modify: `~/secretarius-plugin-spike/derisk-deleg/src/index.ts`
- Modify: `~/.openclaw-slm/workspace-gog/AGENTS.md`
- Modify: `~/.openclaw-slm/openclaw.json`
- Create: `~/.openclaw-slm/workspace/skills/connecter/SKILL.md`

- [ ] **Step 1 : Caractériser la sortie de `gog auth add --manual`**

Lancer, en contrôlé sur l'hôte (NE PAS finaliser — tuer après lecture de l'URL pour ne pas régénérer le token) :
```bash
export XDG_CONFIG_HOME=~/.openclaw-slm/workspace/.gog-config
export GOG_KEYRING_BACKEND=file
export GOG_KEYRING_PASSWORD="$(cat $XDG_CONFIG_HOME/keyring-password)"
timeout 8 gog auth add cmauceri@gmail.com --manual --force-consent --services gmail,drive,calendar 2>&1 | head -40 || true
```
Noter : où apparaît l'URL `https://accounts.google.com/...`, le texte du prompt
attendant le retour, et la forme attendue (URL de redirection complète vs code).
**Ajuster** les motifs `grep` et le contenu écrit dans le FIFO de `gog-auth-bridge.sh`
si la réalité diffère du script de Task 0.1.

- [ ] **Step 2 : Rebuild de l'image gog si le bridge a changé**

Si `gog-auth-bridge.sh` a été modifié :
```bash
cd ~/Secretarius && docker build -f openclaw-config/Dockerfile.gog -t secretarius-gog:latest .
```

- [ ] **Step 3 : Ajouter l'op `auth_start` à `workspace-gog/AGENTS.md`**

Ajouter au tableau :
```
| `auth_start` | **cas spécial async** : `gog-auth-bridge <email>` lancé en `exec background: true` ; répondre « Autorisation lancée. » et s'arrêter. |
```
Et une note : pour `auth_start`, l'agent fait un **seul** `exec background: true` sur
`gog-auth-bridge cmauceri@gmail.com`, puis répond une fois « Autorisation lancée. ».

- [ ] **Step 4 : Ajouter `pendingAuth`, l'outil `gog_connect_start`, et l'interception du retour**

Dans `index.ts`, après la déclaration de `pending` :
```ts
const AUTH_TTL_MS = 10 * 60 * 1000;
const GOG_CFG = `${process.env.HOME}/.openclaw-slm/workspace/.gog-config`;
let pendingAuth: { ts: number } | null = null;
```
Importer `fs`/`path` en tête : `import { readFileSync, writeFileSync, existsSync, rmSync } from "node:fs";`

Outil `gog_connect_start` (après `gog_reply`) :
```ts
    api.registerTool({
      name: "gog_connect_start",
      description:
        "Démarre l'autorisation Google (délègue 'op: auth_start' à l'agent gog), puis renvoie l'URL d'autorisation à coller dans le navigateur.",
      parameters: Type.Object({
        command: Type.Optional(Type.String({ description: "Inutilisé." })),
      }),
      async execute(_id: string, _params: { command?: string }) {
        try { rmSync(`${GOG_CFG}/auth_url`); } catch {}
        try { rmSync(`${GOG_CFG}/auth_response`); } catch {}
        try { rmSync(`${GOG_CFG}/auth_done`); } catch {}
        await delegateGog(api, "auth_start", "");
        // Attendre que le bridge expose l'URL (poll court).
        let url = "";
        for (let i = 0; i < 60; i++) {
          if (existsSync(`${GOG_CFG}/auth_url`)) { url = readFileSync(`${GOG_CFG}/auth_url`, "utf8").trim(); if (url) break; }
          await new Promise((r) => setTimeout(r, 500));
        }
        if (!url) {
          return { content: [{ type: "text", text: "Échec : URL d'autorisation non générée. Réessayez /connecter." }] };
        }
        pendingAuth = { ts: Date.now() };
        return {
          content: [
            {
              type: "text",
              text: `Pour connecter votre compte Google :\n1. Ouvrez ce lien et autorisez :\n${url}\n2. Recollez ici l'URL de redirection obtenue après autorisation.`,
            },
          ],
        };
      },
    });
```

Dans le hook `before_agent_reply`, AVANT le match `/(confirm|annuler)/`, ajouter
l'interception du retour OAuth (le message n'est pas une slash-command) :
```ts
      // Retour OAuth : si une autorisation est en attente, le message courant est
      // l'URL de redirection à injecter dans le pont gog.
      if (pendingAuth && !/^\s*\//.test(text) && text.trim()) {
        if (Date.now() - pendingAuth.ts > AUTH_TTL_MS) {
          pendingAuth = null;
          return { handled: true, reply: { text: "Autorisation expirée — relancez /connecter." } };
        }
        pendingAuth = null;
        writeFileSync(`${GOG_CFG}/auth_response`, text.trim(), "utf8");
        let done = "";
        for (let i = 0; i < 60; i++) {
          if (existsSync(`${GOG_CFG}/auth_done`)) { done = readFileSync(`${GOG_CFG}/auth_done`, "utf8").trim(); break; }
          await new Promise((r) => setTimeout(r, 500));
        }
        const ok = done === "ok";
        return { handled: true, reply: { text: ok ? "Compte Google connecté." : `Échec de la connexion (${done || "timeout"}). Réessayez /connecter.` } };
      }
```

- [ ] **Step 5 : Skill `/connecter` + config**

`~/.openclaw-slm/workspace/skills/connecter/SKILL.md` :
```markdown
---
name: connecter
description: "Connecter (autoriser) votre compte Google. Dispatch déterministe vers gog_connect_start."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_connect_start
command-arg-mode: raw
---

`/connecter` démarre l'autorisation Google : un lien vous est envoyé, vous autorisez puis recollez l'URL de redirection.
```

Config (`openclaw.json`, backup d'abord `cp … .bak-connecter`) : ajouter
`gog_connect_start` à l'allow global et au deny des agents `wiki`, `scout`, `gog`.

- [ ] **Step 6 : Déployer**

```bash
cd ~/secretarius-plugin-spike/derisk-deleg && npm run build \
  && openclaw --profile slm plugins install . --force \
  && systemctl --user restart openclaw-gateway-slm && sleep 3 \
  && systemctl --user is-active openclaw-gateway-slm
```

- [ ] **Step 7 : Vérification E2E** (régénère réellement le token — fait avec l'utilisateur)

Telegram `/new` → `/connecter` → ouvrir le lien, autoriser → recoller l'URL de redirection
Expected : « Compte Google connecté. » Puis vérifier le scope calendar débloqué :
```bash
export XDG_CONFIG_HOME=~/.openclaw-slm/workspace/.gog-config GOG_KEYRING_BACKEND=file
export GOG_KEYRING_PASSWORD="$(cat $XDG_CONFIG_HOME/keyring-password)"
gog calendar events --max 1 2>&1 | head
```
Expected : plus de 403 (événements ou liste vide, mais pas « insufficient scopes »).

- [ ] **Step 8 : Commit**

```bash
cd ~/Secretarius && git add openclaw-config/gog-auth-bridge.sh \
  && git commit -m "feat(connecter): ajustement bridge OAuth après caractérisation" --allow-empty
cd ~/secretarius-plugin-spike/derisk-deleg && git add src/index.ts \
  && git commit -m "feat(plugin): /connecter — gog_connect_start + interception retour OAuth"
```

---

# Phase 2 — Commandes wiki (sans rebuild)

### Task 2.1 : Op `tags` dans `wiki.py`

**Files:**
- Modify: `~/Secretarius/Wiki_LM/tools/wiki.py`

- [ ] **Step 1 : Ajouter l'import et la fonction op**

Après les imports existants (`from query import WikiQuery`, ligne ~31), ajouter :
```python
from kb_tags import collect_tags
```
Avant `def main(`, ajouter :
```python
def op_tags() -> dict:
    tags = collect_tags(_wiki_root() / "wiki")
    return {"tags": sorted(tags.keys())}
```

- [ ] **Step 2 : Brancher l'op dans `main`**

Dans `main`, après la branche `query` (`if op == "query": return op_query(arg)`), ajouter :
```python
    if op == "tags":
        return op_tags()
```

- [ ] **Step 3 : Vérifier l'usage local (hôte)**

Run: `cd ~/Secretarius/Wiki_LM/tools && WIKI_PATH=~/Documents/Arbath/Wiki_LM python3 wiki.py tags`
Expected : JSON `{"tags": [...]}` (liste réelle des tags). Si `WIKI_PATH` diffère, utiliser le chemin réel du wiki.

- [ ] **Step 4 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/wiki.py && git commit -m "feat(wiki): op CLI tags"
```

### Task 2.2 : Ops `kb_update` + `_kb_update_worker` dans `wiki.py`

**Files:**
- Modify: `~/Secretarius/Wiki_LM/tools/wiki.py`

- [ ] **Step 1 : Imports**

Après `from kb_tags import collect_tags` :
```python
from kb_update import update_kb, _DEFAULT_EMBED_DIR, _DEFAULT_KB_DIR
```

- [ ] **Step 2 : Fonction worker (calquée sur `op_ingest_worker`)**

Ajouter avant `def main(` (s'inspirer du verrou/état de `op_ingest_worker` existant — réutiliser le même répertoire d'état) :
```python
def _kb_update_state() -> Path:
    return _wiki_root() / ".kb_update_state.json"


def op_kb_update() -> dict:
    wiki_dir = _wiki_root() / "wiki"
    clusterings_dir = wiki_dir / "clusterings"
    if not clusterings_dir.exists():
        return {"status": "error", "reason": "répertoire clusterings/ introuvable"}
    candidates = sorted(
        (c for c in clusterings_dir.iterdir() if c.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"status": "error", "reason": "aucun clustering disponible"}
    clustering_name = candidates[0].name
    stats = update_kb(
        wiki_root=wiki_dir,
        clustering_name=clustering_name,
        embed_dir=_DEFAULT_EMBED_DIR,
        kb_dir=_DEFAULT_KB_DIR,
    )
    return {"status": "ok", "clustering": clustering_name, **stats}


def op_kb_update_worker() -> dict:
    state = _kb_update_state()
    try:
        result = op_kb_update()
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    state.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return {"status": "worker_done"}
```

- [ ] **Step 3 : Brancher dans `main`**

Après la branche `tags` :
```python
    if op == "kb_update":
        return op_kb_update()
    if op == "_kb_update_worker":
        return op_kb_update_worker()
```

- [ ] **Step 4 : Vérifier l'import dans le conteneur** (sanity, sans lancer le run lourd)

Run: `cd ~/Secretarius/Wiki_LM/tools && python3 -c "import wiki; print('import ok')"`
Expected : `import ok` (si `frontmatter`/`numpy` manquent sur l'hôte, c'est sans incidence : ils sont dans l'image — la vérification réelle est l'E2E Task 2.4).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/wiki.py && git commit -m "feat(wiki): ops CLI kb_update + worker async"
```

### Task 2.3 : Outils plugin wiki + AGENTS.md wiki + config + skills

**Files:**
- Modify: `~/secretarius-plugin-spike/derisk-deleg/src/index.ts`
- Modify: `~/.openclaw-slm/workspace-wiki/AGENTS.md`
- Modify: `~/.openclaw-slm/openclaw.json`
- Create: `~/.openclaw-slm/workspace/skills/{tags,kbupdate}/SKILL.md`

- [ ] **Step 1 : Ajouter les deux outils plugin**

Dans `index.ts`, après `wiki_query` :
```ts
    api.registerTool({
      name: "wiki_tags",
      description: "Liste les tags du wiki (délègue 'op: tags' à l'agent wiki).",
      parameters: Type.Object({ command: Type.Optional(Type.String({ description: "Inutilisé." })) }),
      async execute(_id: string, _params: { command?: string }) {
        const out = await delegateWiki(api, "tags", "");
        return { content: [{ type: "text", text: out.slice(0, 1800) }] };
      },
    });

    api.registerTool({
      name: "wiki_kb_update",
      description: "Met à jour le KB depuis le dernier clustering (délègue 'op: kb_update' à l'agent wiki ; async).",
      parameters: Type.Object({ command: Type.Optional(Type.String({ description: "Inutilisé." })) }),
      async execute(_id: string, _params: { command?: string }) {
        const out = await delegateWiki(api, "kb_update", "");
        return { content: [{ type: "text", text: out.slice(0, 1800) }] };
      },
    });
```

- [ ] **Step 2 : Ops dans `workspace-wiki/AGENTS.md`**

Ajouter au tableau op → sortie :
```
| `tags` | — | `{"tags": [...]}` |
| `kb_update` | — | **cas async** : un seul `exec background: true` sur `python3 /wiki-tools/wiki.py _kb_update_worker`, puis répondre « Mise à jour du KB lancée en arrière-plan. » |
```
Et étendre la « Tolérance de format » : « mettre à jour le KB » / « kb update » → `kb_update` (procédure async, sans argument).

- [ ] **Step 3 : Config**

Backup (`cp … .bak-wiki-tools`). Ajouter `wiki_tags`, `wiki_kb_update` à l'allow global et au deny des agents `wiki`, `scout`, `gog`.

- [ ] **Step 4 : Skills**

`~/.openclaw-slm/workspace/skills/tags/SKILL.md` :
```markdown
---
name: tags
description: "Lister les tags de la base de connaissances. Dispatch déterministe vers wiki_tags."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_tags
command-arg-mode: raw
---

`/tags` liste les tags du wiki, de façon déterministe via l'agent wiki.
```

`~/.openclaw-slm/workspace/skills/kbupdate/SKILL.md` :
```markdown
---
name: kbupdate
description: "Mettre à jour la base de connaissances depuis le dernier clustering. Dispatch déterministe vers wiki_kb_update."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_kb_update
command-arg-mode: raw
---

`/kbupdate` lance la mise à jour du KB en arrière-plan, de façon déterministe via l'agent wiki.
```

- [ ] **Step 5 : Déployer**

```bash
cd ~/secretarius-plugin-spike/derisk-deleg && npm run build \
  && openclaw --profile slm plugins install . --force \
  && systemctl --user restart openclaw-gateway-slm && sleep 3 \
  && systemctl --user is-active openclaw-gateway-slm
```

- [ ] **Step 6 : Commit plugin**

```bash
git add src/index.ts && git commit -m "feat(plugin): outils wiki_tags + wiki_kb_update"
```

### Task 2.4 : Vérifications E2E — /tags, /kbupdate

- [ ] **Step 1 : /tags** — Telegram `/new` → `/tags`
Expected : liste réelle des tags du KB.

- [ ] **Step 2 : /kbupdate** — `/new` → `/kbupdate`
Expected : « Mise à jour du KB lancée en arrière-plan. » Puis, après quelques minutes, vérifier `.kb_update_state.json` côté wiki (status ok + clustering) :
```bash
cat ~/Documents/Arbath/Wiki_LM/.kb_update_state.json
```
Expected : `{"status": "ok", "clustering": "...", ...}`.

---

## Self-review (à la fin)

- **Couverture spec** : étape 0 (Task 0.1-0.3) ; /connecter (1.7) ; /chercher /lire /drive (1.2,1.5,1.6) ; /repondre (1.1,1.3,1.5,1.6) ; /tags (2.1,2.3,2.4) ; /kbupdate (2.2,2.3,2.4). Config allow/deny + skills à chaque phase.
- **Cohérence des noms** : outils `gog_search/gog_get/gog_drive_search/gog_reply/gog_connect_start/wiki_tags/wiki_kb_update` ; ops agent `search/get/drive_search/reply/auth_start/tags/kb_update` ; état `pending` (union) + `pendingAuth`.
- **Risque connu** : le format exact du flux `gog auth add --manual` (Task 1.7 Step 1) — caractériser avant de figer le bridge. C'est la seule inconnue technique du lot.
