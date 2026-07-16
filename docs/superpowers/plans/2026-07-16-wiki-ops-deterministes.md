# Ops wiki déterministes — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** le plugin `derisk-deleg` exécute `wiki.py <op>` directement dans le sandbox wiki (sans tour de LLM) pour les 6 ops, supprimant la latitude d'invention de l'agent orchestrateur.

**Architecture :** une fonction `runWikiOp(api, op, arg)` remplace `delegateWiki` : elle lance `python3 /wiki-tools/wiki.py <op> "<arg>"` dans le sandbox `secretarius-wiki` via le SDK (`resolveSandboxContext` → `runShellCommand`), parse le JSON et le formate. Euria reste le moteur génératif *dans* `wiki.py`.

**Tech stack :** TypeScript + vitest, plugin OpenClaw (SDK `openclaw/plugin-sdk/sandbox` + `/core`), build `tsc`, déploiement `openclaw plugins build`.

**Spec :** `docs/superpowers/specs/2026-07-16-wiki-ops-deterministes-design.md`

## Global Constraints

- Répertoire de travail plugin : `~/Secretarius/derisk-deleg` (source dépôt). Déploiement vers `~/.openclaw/extensions/derisk-deleg`.
- Les 6 ops : `capture`, `query`, `status`, `ingest`, `tags`, `kb_update`. gog **inchangé**.
- Isolation : `wiki.py` reste dans le sandbox `secretarius-wiki` (montages : `/wiki-tools`, `/Wiki_LM`, `/run/euria-key`, `/zim` ; réseau bridge). Ne PAS déplacer sur l'hôte sauf repli documenté.
- Si `wiki.py` renvoie `{error}` ou `{status:"error"}` → surfacer **verbatim** (`error`/`reason`), jamais d'invention.
- Commits en français. Ne pas pousser. `systemctl restart` = confirmation utilisateur.
- Tests : `npm test` (`vitest run`) dans `derisk-deleg`. Imports ESM avec extension `.js` (cf. `dispatch.test.ts`).
- Formes JSON de `wiki.py` (contrat, ne pas changer) : `query`→`{synthesis,references}｜{error}` ; `capture`→`{files:[...]}` ; `ingest`→`{status:"launched"｜"nothing_to_do"｜"already_running", queued:N}` ; `status`→`{running:bool, last_run, pending:N, blocked_files:[...]}` ; `tags`→`{tags:[...]}` ; `kb_update`→`{status:...}｜{status:"error", reason:...}`.

---

### Task 1 : `formatWikiResult` — formatage pur JSON→message (TDD)

**Files:**
- Create: `derisk-deleg/src/wiki-ops.ts`
- Test: `derisk-deleg/src/wiki-ops.test.ts`

**Interfaces:**
- Produces: `export function formatWikiResult(op: string, json: any): string` — convertit le JSON de `wiki.py` en message utilisateur. Consommé par `runWikiOp` (Task 3).

- [ ] **Step 1 : écrire les tests qui échouent**

`derisk-deleg/src/wiki-ops.test.ts` :

```typescript
import { describe, expect, it } from "vitest";
import { formatWikiResult } from "./wiki-ops.js";

describe("formatWikiResult", () => {
  it("query : renvoie la synthèse verbatim", () => {
    expect(formatWikiResult("query", { synthesis: "# GPU TEE\n…", references: ["c-x"] }))
      .toBe("# GPU TEE\n…");
  });
  it("query : erreur surfacée verbatim", () => {
    expect(formatWikiResult("query", { error: "index vide" })).toBe("index vide");
  });
  it("capture : liste les fichiers", () => {
    expect(formatWikiResult("capture", { files: ["a.url", "b.url"] }))
      .toBe("Capturé : a.url, b.url (en file d'attente pour ingestion).");
  });
  it("ingest : mappe le status", () => {
    expect(formatWikiResult("ingest", { status: "launched", queued: 3 }))
      .toBe("Ingestion lancée en arrière-plan.");
    expect(formatWikiResult("ingest", { status: "nothing_to_do", queued: 0 }))
      .toBe("Rien à ingérer.");
    expect(formatWikiResult("ingest", { status: "already_running", queued: 2 }))
      .toBe("Ingestion déjà en cours.");
  });
  it("status : rend l'état sobrement", () => {
    expect(formatWikiResult("status", { running: true, last_run: null, pending: 4, blocked_files: [] }))
      .toBe("Ingestion en cours. En attente : 4. Bloqués : 0.");
    expect(formatWikiResult("status", { running: false, last_run: { ingested: 2, errors: 0 }, pending: 0, blocked_files: ["x.url"] }))
      .toBe("Ingestion à l'arrêt (dernier run : 2 ingéré(s), 0 erreur(s)). En attente : 0. Bloqués : 1.");
  });
  it("tags : joint la liste", () => {
    expect(formatWikiResult("tags", { tags: ["gpu", "tee"] })).toBe("Tags : gpu, tee.");
  });
  it("kb_update : succès et erreur", () => {
    expect(formatWikiResult("kb_update", { status: "done" })).toBe("Base de connaissances mise à jour.");
    expect(formatWikiResult("kb_update", { status: "error", reason: "clusterings/ introuvable" }))
      .toBe("clusterings/ introuvable");
  });
  it("erreur générique inconnue → message par défaut", () => {
    expect(formatWikiResult("query", {})).toBe("Réponse wiki vide ou inattendue.");
  });
});
```

- [ ] **Step 2 : vérifier l'échec**

Run : `cd ~/Secretarius/derisk-deleg && npm test`
Attendu : FAIL — `formatWikiResult` introuvable (`Cannot find module './wiki-ops.js'`).

- [ ] **Step 3 : implémentation minimale**

`derisk-deleg/src/wiki-ops.ts` :

```typescript
// Formatage déterministe du JSON de wiki.py en message utilisateur.
// Aucune invention : sur erreur, on surface le texte de wiki.py verbatim.
export function formatWikiResult(op: string, json: any): string {
  if (json && typeof json.error === "string") return json.error;
  if (json && json.status === "error") return json.reason ?? json.error ?? "Erreur wiki.";

  switch (op) {
    case "query":
      return typeof json?.synthesis === "string" && json.synthesis.trim()
        ? json.synthesis
        : "Réponse wiki vide ou inattendue.";
    case "capture": {
      const files = Array.isArray(json?.files) ? json.files : [];
      return files.length
        ? `Capturé : ${files.join(", ")} (en file d'attente pour ingestion).`
        : "Rien à capturer.";
    }
    case "ingest":
      if (json?.status === "launched") return "Ingestion lancée en arrière-plan.";
      if (json?.status === "nothing_to_do") return "Rien à ingérer.";
      if (json?.status === "already_running") return "Ingestion déjà en cours.";
      return "État d'ingestion inconnu.";
    case "status": {
      const state = json?.running ? "Ingestion en cours" : "Ingestion à l'arrêt";
      const lr = json?.last_run;
      const lrTxt = json?.running || !lr
        ? ""
        : ` (dernier run : ${lr.ingested ?? "?"} ingéré(s), ${lr.errors ?? "?"} erreur(s))`;
      const blocked = Array.isArray(json?.blocked_files) ? json.blocked_files.length : 0;
      return `${state}${lrTxt}. En attente : ${json?.pending ?? 0}. Bloqués : ${blocked}.`;
    }
    case "tags": {
      const tags = Array.isArray(json?.tags) ? json.tags : [];
      return tags.length ? `Tags : ${tags.join(", ")}.` : "Aucun tag.";
    }
    case "kb_update":
      return "Base de connaissances mise à jour.";
    default:
      return "Réponse wiki vide ou inattendue.";
  }
}
```

- [ ] **Step 4 : vérifier le succès**

Run : `cd ~/Secretarius/derisk-deleg && npm test`
Attendu : tous les tests `formatWikiResult` PASS ; les tests existants (`dispatch`, `parse`) restent verts.

- [ ] **Step 5 : commit**

```bash
cd ~/Secretarius && git add derisk-deleg/src/wiki-ops.ts derisk-deleg/src/wiki-ops.test.ts
git commit -m "feat(derisk-deleg): formatWikiResult — formatage déterministe JSON wiki.py -> message"
```

---

### Task 2 : `execWikiSandbox` — primitive d'exécution en sandbox (SPIKE, vérifié E2E)

**Files:**
- Modify: `derisk-deleg/src/wiki-ops.ts` (ajout de `execWikiSandbox`)

**Interfaces:**
- Produces: `export async function execWikiSandbox(api: any, argv: string[]): Promise<{ code: number; stdout: string; stderr: string }>` — exécute `argv` (ex. `["python3","/wiki-tools/wiki.py","status"]`) dans le sandbox wiki. Consommé par `runWikiOp` (Task 3).

Cette tâche est un **spike** : le câblage exact du SDK sandbox depuis un plugin est l'inconnue de la spec. Elle est vérifiée par exécution réelle, pas en unitaire (l'exec sandbox est intrinsèquement de l'intégration).

- [ ] **Step 1 : explorer l'accès config + sessionKey**

Ajouter temporairement, dans un tool existant OU via un log au chargement du plugin, l'inspection de `api` :
```typescript
// exploration jetable — retirer après
console.error("[spike] api keys:", Object.keys(api ?? {}));
console.error("[spike] api.runtime keys:", Object.keys(api?.runtime ?? {}));
```
Build + deploy + restart gateway (confirmation utilisateur), déclencher un `/wikistatus`, lire `journalctl --user -u openclaw-gateway`. But : localiser l'`OpenClawConfig` (probablement `api.runtime.config` ou via `resolveSandboxContext` qui l'accepte optionnelle) et confirmer la `sessionKey` du sandbox wiki (utiliser `buildAgentSessionKey` de `openclaw/plugin-sdk/core` ; l'id d'agent est `wiki`).

- [ ] **Step 2 : implémenter la voie principale (SDK sandbox)**

Dans `wiki-ops.ts` :
```typescript
import { resolveSandboxContext } from "openclaw/plugin-sdk/sandbox";

export async function execWikiSandbox(
  api: any, argv: string[],
): Promise<{ code: number; stdout: string; stderr: string }> {
  const config = api?.runtime?.config ?? undefined; // confirmé au Step 1
  const sessionKey = "agent:wiki:subagent:ops"; // clé stable -> conteneur réutilisé
  const ctx = await resolveSandboxContext({ config, sessionKey });
  if (!ctx || !ctx.enabled) {
    return { code: 1, stdout: "", stderr: "sandbox wiki indisponible" };
  }
  const res = await ctx.backend!.runShellCommand({
    command: argv,
    timeoutMs: 120000,
  });
  return {
    code: res.exitCode ?? 0,
    stdout: res.stdout?.toString("utf8") ?? "",
    stderr: res.stderr?.toString("utf8") ?? "",
  };
}
```
Ajuster les noms de champs (`ctx.backend`, `runShellCommand`, `exitCode`, `stdout`/`stderr` en `Buffer`) selon `plugin-sdk/sandbox.d.ts` (types : `SandboxBackendHandle`, `SandboxBackendCommandResult` avec `stdout: Buffer`, `stderr: Buffer`). Si `ctx.backend` est absent, l'obtenir via `getSandboxBackendManager`/`SandboxBackendFactory` (cf. `sandbox.d.ts`).

- [ ] **Step 3 : build + déploiement + vérification réelle**

```bash
cd ~/Secretarius/derisk-deleg && npm run plugin:build
cp -r dist/* ~/.openclaw/extensions/derisk-deleg/dist/   # ou la procédure d'install du plugin
```
Restart gateway (confirmation). Déclencher `/wikistatus` sur Telegram, OU tester la primitive isolément via un log. Vérifier dans `journalctl --user -u openclaw-gateway` : `wiki.py status` **exécuté**, JSON `{running,…}` capturé, PAS de tour de LLM wiki.
Attendu : `execWikiSandbox(api, ["python3","/wiki-tools/wiki.py","status"])` renvoie `{code:0, stdout:'{"running":…}', stderr:""}`.

- [ ] **Step 4 : repli si la voie SDK est impraticable depuis un plugin**

Si `resolveSandboxContext`/`runShellCommand` ne sont pas utilisables depuis le plugin (erreur d'accès config/backend), replier sur `runPluginCommandWithTimeout` de `openclaw/plugin-sdk/sandbox` avec `docker exec` sur un conteneur wiki réutilisé (résolu par `docker ps --filter name=agent-wiki` ou un conteneur nommé stable créé une fois). Documenter le choix retenu en commentaire. Isolation préservée (même image/montages).

- [ ] **Step 5 : retirer l'exploration jetable + commit**

Retirer les `console.error` du Step 1.
```bash
cd ~/Secretarius && git add derisk-deleg/src/wiki-ops.ts
git commit -m "feat(derisk-deleg): execWikiSandbox — exécute wiki.py dans le sandbox wiki (SDK sandbox)"
```

---

### Task 3 : `runWikiOp` — composition (TDD, exec mocké)

**Files:**
- Modify: `derisk-deleg/src/wiki-ops.ts` (ajout de `runWikiOp`)
- Test: `derisk-deleg/src/wiki-ops.test.ts` (ajout)

**Interfaces:**
- Consumes: `formatWikiResult` (Task 1), `execWikiSandbox` (Task 2).
- Produces: `export async function runWikiOp(api: any, op: string, arg: string): Promise<string>`. Consommé par `index.ts` (Task 4). Pour la testabilité, `execWikiSandbox` est injectable via un 4e paramètre optionnel `exec`.

- [ ] **Step 1 : écrire les tests qui échouent**

Ajouter à `wiki-ops.test.ts` :
```typescript
import { runWikiOp } from "./wiki-ops.js";

describe("runWikiOp", () => {
  const okExec = (stdout: string) => async () => ({ code: 0, stdout, stderr: "" });

  it("parse le JSON et formate", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "references": []}'));
    expect(out).toBe("# GPU TEE");
  });
  it("passe op et arg à l'exec", async () => {
    let seen: string[] = [];
    const exec = async (_api: any, argv: string[]) => { seen = argv; return { code: 0, stdout: '{"files":["a.url"]}', stderr: "" }; };
    await runWikiOp(null, "capture", "#x https://e.com", exec);
    expect(seen).toEqual(["python3", "/wiki-tools/wiki.py", "capture", "#x https://e.com"]);
  });
  it("exit non nul → message d'erreur déterministe", async () => {
    const out = await runWikiOp(null, "status", "", async () => ({ code: 1, stdout: "", stderr: "boom" }));
    expect(out).toBe("Erreur wiki : boom");
  });
  it("stdout non-JSON → message d'erreur déterministe", async () => {
    const out = await runWikiOp(null, "status", "", async () => ({ code: 0, stdout: "pas du json", stderr: "" }));
    expect(out).toContain("Erreur wiki");
  });
});
```

- [ ] **Step 2 : vérifier l'échec**

Run : `cd ~/Secretarius/derisk-deleg && npm test`
Attendu : FAIL — `runWikiOp` non exporté.

- [ ] **Step 3 : implémentation**

Ajouter à `wiki-ops.ts` :
```typescript
type Exec = (api: any, argv: string[]) => Promise<{ code: number; stdout: string; stderr: string }>;

export async function runWikiOp(
  api: any, op: string, arg: string, exec: Exec = execWikiSandbox,
): Promise<string> {
  const argv = ["python3", "/wiki-tools/wiki.py", op];
  if (arg) argv.push(arg);
  const { code, stdout, stderr } = await exec(api, argv);
  if (code !== 0) return `Erreur wiki : ${(stderr || stdout || "échec").slice(0, 500)}`;
  let json: any;
  try {
    json = JSON.parse(stdout.trim());
  } catch {
    return `Erreur wiki : sortie inattendue (${stdout.slice(0, 200)})`;
  }
  return formatWikiResult(op, json);
}
```

- [ ] **Step 4 : vérifier le succès**

Run : `cd ~/Secretarius/derisk-deleg && npm test`
Attendu : tous les tests `runWikiOp` + `formatWikiResult` PASS ; existants verts.

- [ ] **Step 5 : commit**

```bash
cd ~/Secretarius && git add derisk-deleg/src/wiki-ops.ts derisk-deleg/src/wiki-ops.test.ts
git commit -m "feat(derisk-deleg): runWikiOp — exec wiki.py + parse + format, gestion d'erreur déterministe"
```

---

### Task 4 : câbler `index.ts`, retirer `delegateWiki`, build + déploiement + E2E

**Files:**
- Modify: `derisk-deleg/src/index.ts` (6 tools + hook `before_agent_reply` → `runWikiOp` ; suppression de `delegateWiki`)

**Interfaces:**
- Consumes: `runWikiOp` (Task 3).

- [ ] **Step 1 : importer runWikiOp**

En tête de `index.ts` :
```typescript
import { runWikiOp } from "./wiki-ops.js";
```

- [ ] **Step 2 : recâbler les 6 outils**

Dans chaque `execute` des tools `wiki_capture`/`wiki_query`/`wiki_status`/`wiki_ingest`/`wiki_tags`/`wiki_kb_update`, remplacer `const out = await delegateWiki(api, "<op>", arg);` par `const out = await runWikiOp(api, "<op>", arg);`. (Ex. `wiki_capture` conserve sa logique d'attachement `media/inbound` puis appelle `runWikiOp(api, "capture", arg)`.)

- [ ] **Step 3 : recâbler le hook**

Dans `before_agent_reply`, la branche `if (action.kind === "wiki")` :
```typescript
      if (action.kind === "wiki") {
        const out = await runWikiOp(api, action.op, routed.args);
        return { handled: true, reply: { text: out.slice(0, 1800) } };
      }
```

- [ ] **Step 4 : retirer `delegateWiki` (orphelin)**

Supprimer la fonction `delegateWiki` (lignes ~87-90) devenue inutilisée. Conserver `runAndRead`, `delegateGog`, `delegateScout`, `uniq` (toujours utilisés par gog/scout).

- [ ] **Step 5 : build + validation**

```bash
cd ~/Secretarius/derisk-deleg && npm test && npm run plugin:validate
```
Attendu : tests verts, `plugin:validate` OK (pas d'erreur TS, pas de `delegateWiki` référencé).

- [ ] **Step 6 : déployer + restart gateway (confirmation utilisateur)**

```bash
npm run plugin:build && cp -r dist/* ~/.openclaw/extensions/derisk-deleg/dist/
systemctl --user restart openclaw-gateway
```

- [ ] **Step 7 : vérification E2E réelle**

Sur Telegram (ou via le chemin routeur) :
- `/q tee gpu` → **la vraie synthèse** (surcoût 41,6×, NVIDIA CC…), pas d'invention.
- `/c #test https://example.com` → « Capturé : … » et un `.url` apparaît dans `raw/`.
- `/ingest` → « Ingestion lancée en arrière-plan. »
- `/wikistatus` → l'état réel.
Vérifier `journalctl --user -u openclaw-gateway` : chaque op **exécute `wiki.py`**, **aucun tour de LLM wiki**. Vérifier `docker ps` : **un seul** conteneur wiki réutilisé (plus de prolifération par op).

- [ ] **Step 8 : commit**

```bash
cd ~/Secretarius && git add derisk-deleg/src/index.ts
git commit -m "feat(derisk-deleg): ops wiki déterministes — runWikiOp remplace delegateWiki (6 ops), orchestrateur LLM retiré"
```

---

## Notes post-implémentation (hors périmètre de ce plan)

- Alléger l'`AGENTS.md` de l'agent wiki (orchestration devenue caduque) — cosmétique.
- Purger les conteneurs wiki orphelins existants (`docker rm`).
- La bascule ingestion sur phi-4 local reste un chantier distinct (`[[project_ingestion_phi4_passages]]`).
