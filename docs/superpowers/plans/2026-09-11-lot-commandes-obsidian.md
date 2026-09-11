# Exécution en lot de commandes wiki depuis une note Obsidian — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettre d'exécuter en un geste une liste de commandes `/` wiki préparée dans une note Obsidian (blocs ```` ```wiki ````), avec le résultat de chacune inséré dans la note.

**Architecture:** Une nouvelle route `POST /run` sur `Wiki_LM/tools/server.py` fait le pont entre une commande texte (`/q`, `/c`, …) et les fonctions `op_*` déjà existantes dans `Wiki_LM/tools/wiki.py`, via une table de correspondance qui exclut délibérément `/supprimer`. Côté plugin `obsidian-wikilm-capture`, un module TypeScript pur (`run-commands.ts`) extrait les blocs ```` ```wiki ```` d'une note et formate les résultats ; `main.ts` l'utilise pour orchestrer l'appel séquentiel à `/run` et réécrire la note en une seule fois.

**Tech Stack:** Python 3.12 / Flask (serveur), TypeScript / esbuild / vitest (plugin Obsidian), pytest (tests serveur).

**Spec:** `docs/superpowers/specs/2026-09-11-lot-commandes-obsidian-design.md`

## Global Constraints

- Commandes en lot limitées à ce sous-ensemble wiki : `/c`, `/q`, `/ingest`, `/wikistatus`, `/r`, `/tags`, `/kbupdate`, `/relire`, `/verifie`.
- `/supprimer` est **explicitement refusée** par la route serveur — absente de la table de dispatch, jamais exécutée, même demandée explicitement. Un test dédié le vérifie (aucun appel à `op_delete`/`op_delete_preview`).
- Une commande en échec (HTTP non-200, `error` dans le JSON, ou exception réseau) n'interrompt pas le lot — le message d'erreur est inséré comme un résultat normal, le bloc suivant s'exécute quand même.
- Relance sur une note déjà exécutée : le résultat déjà présent sous un bloc (repéré par les marqueurs `<!-- wikilm-run:start -->` / `<!-- wikilm-run:end -->`) est **remplacé**, jamais dupliqué. Le texte que l'utilisateur aurait écrit lui-même entre deux blocs n'est jamais touché.
- Pas de synthèse/formatage des résultats côté serveur : `/run` renvoie le JSON de l'opération `wiki.py` tel quel. Le formatage est entièrement dans le plugin.
- Hors périmètre pour ce premier jet : commandes gog/scout, support de `/supprimer` en lot, formatage riche (liens cliquables, Markdown avancé), ré-exécution partielle d'un lot.

---

## Note de conception (rebuild en une passe)

La spec décrit l'insertion comme séquentielle ("insère le résultat … avant de
passer au bloc suivant"). L'implémentation ci-dessous obtient le même résultat
observable — chaque résultat apparaît bien juste après son bloc dans la note
finale — via un algorithme plus simple et plus sûr : tous les blocs sont
extraits une seule fois depuis le texte original (offsets stables), exécutés
un par un dans l'ordre (séquentiellement, résultats accumulés dans un
tableau), puis la note est reconstruite en **une seule passe finale** et
écrite en **un seul** `vault.modify()`. Cela évite tout calcul de décalage
d'offsets pendant l'exécution (qui serait nécessaire si on réécrivait la note
après chaque commande). Le comportement visible pour l'utilisateur est
identique à une insertion incrémentale.

---

### Task 1: Route serveur `POST /run`

**Files:**
- Modify: `Wiki_LM/tools/server.py`
- Test: `Wiki_LM/tests/test_server.py`

**Interfaces:**
- Consumes (déjà existant, `Wiki_LM/tools/wiki.py`) : `op_capture(text: str) -> dict`,
  `op_query(question: str) -> dict`, `op_ingest() -> dict`, `op_status() -> dict`,
  `op_search(question: str) -> dict`, `op_tags() -> dict`, `op_kb_update() -> dict`,
  `op_review() -> dict`, `op_verify(slug: str) -> dict`.
- Produces: route Flask `POST /run`, body `{"command": str, "arg": str}` →
  200 + JSON de l'opération, ou 400 (`{"error": str}`) si commande inconnue/non
  autorisée, ou 500 (`{"error": str}`) si l'opération lève une exception.

Le fichier `Wiki_LM/tests/conftest.py` insère déjà `tools/` dans `sys.path` ;
les tests s'exécutent avec `python3 -m pytest tests/test_server.py -q` depuis
`Wiki_LM/`.

- [ ] **Step 1: Écrire les tests (échec attendu)**

Ouvrir `Wiki_LM/tests/test_server.py` et ajouter, à la suite de la classe
`TestHandleQuery` existante :

```python
class TestHandleRun:
    _COMMAND_TABLE = [
        ("/c", "op_capture", "texte de capture"),
        ("/q", "op_query", "une question ?"),
        ("/ingest", "op_ingest", ""),
        ("/wikistatus", "op_status", ""),
        ("/r", "op_search", "une recherche"),
        ("/tags", "op_tags", ""),
        ("/kbupdate", "op_kb_update", ""),
        ("/relire", "op_review", ""),
        ("/verifie", "op_verify", "src-test"),
    ]

    @pytest.mark.parametrize("command,op_name,arg", _COMMAND_TABLE)
    def test_dispatches_each_command_to_its_op(self, client, monkeypatch, command, op_name, arg):
        import server
        calls = []

        def fake(*args):
            calls.append(args)
            return {"status": "ok"}

        monkeypatch.setattr(server, op_name, fake)
        response = client.post("/run", json={"command": command, "arg": arg})

        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
        assert calls == ([(arg,)] if arg else [()])

    def test_unknown_command_returns_400(self, client):
        response = client.post("/run", json={"command": "/inconnue", "arg": ""})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_supprimer_is_refused_and_never_dispatched(self, client, monkeypatch):
        import server
        called = []
        monkeypatch.setattr(server, "op_delete", lambda slug: called.append(slug) or {"status": "ok"})
        monkeypatch.setattr(server, "op_delete_preview", lambda slug: called.append(slug) or {"status": "ok"})

        response = client.post("/run", json={"command": "/supprimer", "arg": "src-test"})

        assert response.status_code == 400
        assert "error" in response.get_json()
        assert called == []

    def test_op_exception_returns_500(self, client, monkeypatch):
        import server

        def boom(arg):
            raise RuntimeError("panne simulée")

        monkeypatch.setattr(server, "op_capture", boom)
        response = client.post("/run", json={"command": "/c", "arg": "texte"})

        assert response.status_code == 500
        assert "panne simulée" in response.get_json()["error"]
```

Note : `op_delete`/`op_delete_preview` ne sont pas encore importés dans
`server.py` avant l'étape 3 — `monkeypatch.setattr` sur un attribut absent du
module échouerait. Ce n'est pas un problème : ces deux fonctions ne sont
**jamais** importées dans `server.py` par ce plan (voir Step 3) ; le test
`test_supprimer_is_refused_and_never_dispatched` doit donc plutôt vérifier
l'absence d'effet de bord sans passer par `monkeypatch.setattr` sur ces noms.
Remplacer ce test par la version suivante (ne dépend d'aucun import) :

```python
    def test_supprimer_is_refused_and_never_dispatched(self, client):
        response = client.post("/run", json={"command": "/supprimer", "arg": "src-test"})
        assert response.status_code == 400
        assert "error" in response.get_json()
```

(Le fait que `/supprimer` ne soit dispatchée nulle part est garanti
structurellement par l'absence de cette clé dans `_RUN_OPS` — voir Step 3 —
pas par un espion runtime.)

- [ ] **Step 2: Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM && python3 -m pytest tests/test_server.py -q`
Expected: FAIL — `404 NOT FOUND` sur `/run` (la route n'existe pas encore).

- [ ] **Step 3: Implémenter la route `/run`**

Dans `Wiki_LM/tools/server.py`, ajouter l'import après `from capture import
capture_comment, _normalize_tags, raw_dir` :

```python
from wiki import (
    op_capture,
    op_ingest,
    op_kb_update,
    op_query,
    op_review,
    op_search,
    op_status,
    op_tags,
    op_verify,
)
```

Mettre à jour le docstring en tête de fichier (après le bloc `POST /capture`
existant, avant `GET /health`) :

```python
    POST /run
    Body  : {"command": "/q", "arg": "..."}
    Reply : JSON de l'opération wiki.py correspondante ; {"error": "..."} si
            commande inconnue (400) ou en échec (500). /supprimer est
            toujours refusée.
```

Ajouter la table de dispatch et la route, juste après `handle_capture` :

```python
_RUN_OPS = {
    "/c": lambda arg: op_capture(arg),
    "/q": lambda arg: op_query(arg),
    "/ingest": lambda arg: op_ingest(),
    "/wikistatus": lambda arg: op_status(),
    "/r": lambda arg: op_search(arg),
    "/tags": lambda arg: op_tags(),
    "/kbupdate": lambda arg: op_kb_update(),
    "/relire": lambda arg: op_review(),
    "/verifie": lambda arg: op_verify(arg),
}


@app.post("/run")
def handle_run():
    """Exécute une commande wiki pour le lot Obsidian (blocs ```wiki). /supprimer
    est absente de _RUN_OPS : jamais dispatchée, même demandée explicitement."""
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "")).strip()
    arg = str(data.get("arg", ""))

    op = _RUN_OPS.get(command)
    if op is None:
        return jsonify({"error": f"Commande inconnue ou non autorisée en lot : {command!r}"}), 400

    try:
        result = op(arg)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)
```

- [ ] **Step 4: Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM && python3 -m pytest tests/test_server.py -q`
Expected: PASS (13 tests : 4 existants + 9 paramétrés + 3 nouveaux — vérifier
le compte exact affiché correspond à 4 + 9 + 3 = 16 au total pour ce fichier).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/server.py Wiki_LM/tests/test_server.py
git commit -m "$(cat <<'EOF'
feat(wiki-server): ajoute POST /run pour l'exécution en lot de commandes wiki

Nouvelle route qui dispatche vers les fonctions op_* de wiki.py à partir
d'une table stable (commande → opération). /supprimer est explicitement
absente de la table : jamais dispatchée en lot, même demandée.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 2: Module pur `run-commands.ts` (extraction des blocs + formatage)

**Files:**
- Create: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts`
- Test: `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts`

**Interfaces:**
- Consumes: rien (module pur, aucune dépendance à l'API Obsidian — même esprit
  que `capture-text.ts`).
- Produces (consommés par Task 3 dans `main.ts`) :
  - `SUPPORTED_COMMANDS: readonly string[]` — les 9 commandes autorisées en lot.
  - `interface WikiBlockMatch { command: string; arg: string; content: string; start: number; end: number }`
  - `extractWikiBlocks(noteText: string): WikiBlockMatch[]`
  - `formatWikiResult(command: string, data: Record<string, unknown>, ok: boolean): string`
  - `applyWikiResults(noteText: string, blocks: WikiBlockMatch[], resultTexts: string[]): string`

- [ ] **Step 1: Écrire les tests (échec attendu)**

Créer `Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts` :

```typescript
import { describe, expect, it } from "vitest";
import {
  applyWikiResults,
  extractWikiBlocks,
  formatWikiResult,
  SUPPORTED_COMMANDS,
} from "./run-commands";

describe("extractWikiBlocks", () => {
  it("extracts command and argument from a single block", () => {
    const note = "```wiki\n/q\nQuestion ?\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].command).toBe("/q");
    expect(blocks[0].arg).toBe("Question ?");
  });

  it("preserves a multi-line argument joined by \\n, not flattened", () => {
    const note = "```wiki\n/c\nligne 1\nligne 2\nligne 3\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks[0].arg).toBe("ligne 1\nligne 2\nligne 3");
  });

  it("returns an empty argument for a block with no second line", () => {
    const note = "```wiki\n/wikistatus\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks[0].command).toBe("/wikistatus");
    expect(blocks[0].arg).toBe("");
  });

  it("extracts multiple blocks in document order", () => {
    const note = [
      "```wiki",
      "/c",
      "#zoologue https://en.wikipedia.org/wiki/Dmitry_Belyayev_(zoologist)",
      "```",
      "",
      "```wiki",
      "/q",
      "Qu'est-ce que la domestication du renard argenté ?",
      "```",
      "",
    ].join("\n");
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].command).toBe("/c");
    expect(blocks[0].arg).toBe(
      "#zoologue https://en.wikipedia.org/wiki/Dmitry_Belyayev_(zoologist)"
    );
    expect(blocks[1].command).toBe("/q");
    expect(blocks[1].arg).toBe("Qu'est-ce que la domestication du renard argenté ?");
  });

  it("ignores prose text outside ```wiki blocks", () => {
    const note = "Des notes avant.\n\n```wiki\n/tags\n```\n\nDes notes après.\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].command).toBe("/tags");
  });

  it("includes a pre-existing result marker in the block span, for rerun replacement", () => {
    const note =
      "```wiki\n/tags\n```\n\n<!-- wikilm-run:start -->\nancien résultat\n<!-- wikilm-run:end -->\n\nSuite du texte.";
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(1);
    const tail = note.slice(blocks[0].end);
    expect(tail.startsWith("\n\nSuite du texte.")).toBe(true);
  });
});

describe("formatWikiResult", () => {
  it("formats an error uniformly regardless of command", () => {
    expect(formatWikiResult("/q", { error: "KB vide" }, true)).toBe("**Erreur :** KB vide");
    expect(formatWikiResult("/q", {}, false)).toBe("**Erreur :** Erreur inconnue");
  });

  it("formats /c with files captured", () => {
    expect(formatWikiResult("/c", { files: ["src-1.md", "src-2.md"] }, true)).toBe(
      "Capturé : src-1.md, src-2.md"
    );
    expect(formatWikiResult("/c", { files: [] }, true)).toBe("Rien à capturer.");
  });

  it("formats /q with the full synthesis", () => {
    expect(formatWikiResult("/q", { synthesis: "Réponse complète." }, true)).toBe(
      "Réponse complète."
    );
  });

  it("formats /ingest per status", () => {
    expect(formatWikiResult("/ingest", { status: "launched", queued: 3 }, true)).toBe(
      "Ingestion lancée (3 en attente)."
    );
    expect(formatWikiResult("/ingest", { status: "already_running", queued: 1 }, true)).toBe(
      "Ingestion déjà en cours."
    );
    expect(formatWikiResult("/ingest", { status: "nothing_to_do", queued: 0 }, true)).toBe(
      "Rien à ingérer."
    );
  });

  it("formats /wikistatus", () => {
    const text = formatWikiResult(
      "/wikistatus",
      { running: false, pending: 2, blocked_files: ["a.url.error"] },
      true
    );
    expect(text).toBe("En cours : non. En attente : 2. Bloqués : 1.");
  });

  it("formats /r as a numbered list", () => {
    const text = formatWikiResult(
      "/r",
      { results: [{ title: "Titre A", excerpt: "Extrait A" }, { title: "Titre B", excerpt: "Extrait B" }] },
      true
    );
    expect(text).toBe("1. Titre A — Extrait A\n2. Titre B — Extrait B");
    expect(formatWikiResult("/r", { results: [] }, true)).toBe("Aucun résultat.");
  });

  it("formats /tags as a comma-separated list", () => {
    expect(formatWikiResult("/tags", { tags: ["ia", "wiki"] }, true)).toBe("ia, wiki");
    expect(formatWikiResult("/tags", { tags: [] }, true)).toBe("Aucun tag.");
  });

  it("formats /kbupdate", () => {
    expect(formatWikiResult("/kbupdate", { status: "ok", clustering: "2026-09-11" }, true)).toBe(
      "Base de connaissances mise à jour (clustering 2026-09-11)."
    );
  });

  it("formats /relire empty or with content", () => {
    expect(formatWikiResult("/relire", { status: "empty" }, true)).toBe("Rien à relire.");
    expect(formatWikiResult("/relire", { status: "ok", slug: "src-x", content: "# Contenu" }, true)).toBe(
      "# Contenu"
    );
  });

  it("formats /verifie", () => {
    expect(formatWikiResult("/verifie", { status: "ok", slug: "src-x" }, true)).toBe(
      "Page src-x marquée vérifiée."
    );
  });

  it("formats an unsupported command as a client-side refusal", () => {
    expect(formatWikiResult("/supprimer", {}, true)).toBe(
      "Commande non prise en charge en exécution par lot : /supprimer"
    );
  });
});

describe("SUPPORTED_COMMANDS", () => {
  it("excludes /supprimer", () => {
    expect(SUPPORTED_COMMANDS).not.toContain("/supprimer");
  });

  it("contains exactly the nine wiki commands from the spec", () => {
    expect([...SUPPORTED_COMMANDS].sort()).toEqual(
      ["/c", "/ingest", "/kbupdate", "/q", "/r", "/relire", "/tags", "/verifie", "/wikistatus"].sort()
    );
  });
});

describe("applyWikiResults", () => {
  it("inserts the result right after the block, preserving surrounding prose", () => {
    const note = "Avant.\n\n```wiki\n/tags\n```\n\nAprès.";
    const blocks = extractWikiBlocks(note);
    const output = applyWikiResults(note, blocks, ["ia, wiki"]);
    expect(output).toBe(
      "Avant.\n\n```wiki\n/tags\n```\n\n<!-- wikilm-run:start -->\nia, wiki\n<!-- wikilm-run:end -->\n\nAprès."
    );
  });

  it("replaces a prior result instead of duplicating it on rerun", () => {
    const note = "```wiki\n/tags\n```\n";
    const firstPass = applyWikiResults(note, extractWikiBlocks(note), ["ancien résultat"]);
    const secondBlocks = extractWikiBlocks(firstPass);
    expect(secondBlocks).toHaveLength(1);

    const secondPass = applyWikiResults(firstPass, secondBlocks, ["nouveau résultat"]);
    const marks = secondPass.match(/wikilm-run:start/g) ?? [];
    expect(marks).toHaveLength(1);
    expect(secondPass).toContain("nouveau résultat");
    expect(secondPass).not.toContain("ancien résultat");
  });
});
```

- [ ] **Step 2: Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npx vitest run src/run-commands.test.ts`
Expected: FAIL — `Cannot find module './run-commands'`.

- [ ] **Step 3: Implémenter `run-commands.ts`**

Créer `Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts` :

```typescript
export const SUPPORTED_COMMANDS = [
  "/c",
  "/q",
  "/ingest",
  "/wikistatus",
  "/r",
  "/tags",
  "/kbupdate",
  "/relire",
  "/verifie",
] as const;

export interface WikiBlockMatch {
  command: string;
  arg: string;
  content: string;
  start: number;
  end: number;
}

const BLOCK_RE =
  /```wiki\r?\n([\s\S]*?)\r?\n```(?:\r?\n+<!-- wikilm-run:start -->[\s\S]*?<!-- wikilm-run:end -->)?/g;

export function extractWikiBlocks(noteText: string): WikiBlockMatch[] {
  const blocks: WikiBlockMatch[] = [];
  const re = new RegExp(BLOCK_RE.source, "g");
  let match: RegExpExecArray | null;
  while ((match = re.exec(noteText)) !== null) {
    const content = match[1];
    const lines = content.split("\n");
    const command = lines[0].trim();
    const arg = lines.slice(1).join("\n").trim();
    blocks.push({
      command,
      arg,
      content,
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return blocks;
}

export function applyWikiResults(
  noteText: string,
  blocks: WikiBlockMatch[],
  resultTexts: string[]
): string {
  let output = "";
  let cursor = 0;
  blocks.forEach((block, i) => {
    output += noteText.slice(cursor, block.start);
    output += "```wiki\n" + block.content + "\n```\n\n";
    output += "<!-- wikilm-run:start -->\n" + resultTexts[i] + "\n<!-- wikilm-run:end -->";
    cursor = block.end;
  });
  output += noteText.slice(cursor);
  return output;
}

export function formatWikiResult(
  command: string,
  data: Record<string, unknown>,
  ok: boolean
): string {
  if (!ok || typeof data.error === "string") {
    const message = typeof data.error === "string" ? data.error : "Erreur inconnue";
    return `**Erreur :** ${message}`;
  }
  switch (command) {
    case "/c": {
      const files = (data.files as string[] | undefined) ?? [];
      return files.length > 0 ? `Capturé : ${files.join(", ")}` : "Rien à capturer.";
    }
    case "/q":
      return String(data.synthesis ?? "");
    case "/ingest": {
      const status = data.status as string | undefined;
      if (status === "launched") return `Ingestion lancée (${data.queued} en attente).`;
      if (status === "already_running") return "Ingestion déjà en cours.";
      return "Rien à ingérer.";
    }
    case "/wikistatus": {
      const running = data.running ? "oui" : "non";
      const blocked = ((data.blocked_files as string[] | undefined) ?? []).length;
      return `En cours : ${running}. En attente : ${data.pending}. Bloqués : ${blocked}.`;
    }
    case "/r": {
      const results = (data.results as { title: string; excerpt: string }[] | undefined) ?? [];
      if (results.length === 0) return "Aucun résultat.";
      return results.map((r, i) => `${i + 1}. ${r.title} — ${r.excerpt}`).join("\n");
    }
    case "/tags": {
      const tags = (data.tags as string[] | undefined) ?? [];
      return tags.length > 0 ? tags.join(", ") : "Aucun tag.";
    }
    case "/kbupdate":
      return `Base de connaissances mise à jour (clustering ${data.clustering}).`;
    case "/relire":
      return data.status === "empty" ? "Rien à relire." : String(data.content ?? "");
    case "/verifie":
      return `Page ${data.slug} marquée vérifiée.`;
    default:
      return `Commande non prise en charge en exécution par lot : ${command}`;
  }
}
```

Note : `BLOCK_RE` est défini avec le flag global au niveau module puis
recréé (`new RegExp(BLOCK_RE.source, "g")`) à chaque appel de
`extractWikiBlocks` — un `RegExp` global partagé garderait son `lastIndex`
entre deux appels et casserait l'extraction sur la deuxième note traitée
dans la même session du plugin.

- [ ] **Step 4: Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npx vitest run src/run-commands.test.ts`
Expected: PASS (tous les tests).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/obsidian-wikilm-capture/src/run-commands.ts Wiki_LM/obsidian-wikilm-capture/src/run-commands.test.ts
git commit -m "$(cat <<'EOF'
feat(wikilm-capture): ajoute l'extraction et le formatage des blocs wiki

Module pur : extractWikiBlocks (parsing des blocs \`\`\`wiki, marqueurs de
relance inclus dans le span pour éviter la duplication), formatWikiResult
(un format texte par commande), applyWikiResults (reconstruction en une
passe). Aucune dépendance à l'API Obsidian, testable en isolation comme
capture-text.ts.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 3: Commande plugin « Exécuter les commandes wiki de la note »

**Files:**
- Modify: `Wiki_LM/obsidian-wikilm-capture/src/main.ts`

**Interfaces:**
- Consumes (Task 2) : `SUPPORTED_COMMANDS`, `WikiBlockMatch`, `extractWikiBlocks`,
  `applyWikiResults`, `formatWikiResult`.
- Produces: rien consommé par une tâche ultérieure — c'est la dernière tâche
  du plan.

- [ ] **Step 1: Ajouter l'import et la commande/l'icône dans `onload()`**

Dans `Wiki_LM/obsidian-wikilm-capture/src/main.ts`, modifier l'import en tête
de fichier :

```typescript
import { buildCaptureText } from "./capture-text";
import {
  applyWikiResults,
  extractWikiBlocks,
  formatWikiResult,
  SUPPORTED_COMMANDS,
  WikiBlockMatch,
} from "./run-commands";
```

Dans `onload()`, juste après l'enregistrement existant de `capture-current-note` :

```typescript
    this.addRibbonIcon("play", "Exécuter les commandes wiki de la note", () =>
      this.runWikiCommands()
    );
    this.addCommand({
      id: "run-wiki-commands",
      name: "Exécuter les commandes wiki de la note",
      callback: () => this.runWikiCommands(),
    });
```

- [ ] **Step 2: Ajouter les méthodes `runWikiCommands` et `runOneCommand`**

Dans la classe `WikilmCapturePlugin`, après `captureCurrentNote` :

```typescript
  async runWikiCommands(): Promise<void> {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("Aucune note ouverte");
      return;
    }

    const raw = await this.app.vault.read(file);
    const blocks = extractWikiBlocks(raw);
    if (blocks.length === 0) {
      new Notice("Aucune commande wiki trouvée dans la note");
      return;
    }

    const resultTexts: string[] = [];
    for (const block of blocks) {
      resultTexts.push(await this.runOneCommand(block));
    }

    const newContent = applyWikiResults(raw, blocks, resultTexts);
    await this.app.vault.modify(file, newContent);
    new Notice(`Lot exécuté : ${blocks.length} commande(s)`);
  }

  async runOneCommand(block: WikiBlockMatch): Promise<string> {
    if (!(SUPPORTED_COMMANDS as readonly string[]).includes(block.command)) {
      return formatWikiResult(block.command, {}, true);
    }
    try {
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/run`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ command: block.command, arg: block.arg }),
        throw: false,
      });
      const data = (response.json as Record<string, unknown>) ?? {};
      return formatWikiResult(block.command, data, response.status === 200);
    } catch (err) {
      return formatWikiResult(block.command, { error: String(err) }, false);
    }
  }
```

- [ ] **Step 3: Vérifier la compilation TypeScript**

Run: `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && node esbuild.config.mjs production`
Expected: build réussi, `main.js` régénéré, aucune erreur TypeScript (esbuild
transpile sans vérification de types complète — si une erreur de type doit
être détectée en amont, utiliser `npx tsc --noEmit` en complément).

Run aussi : `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npx tsc --noEmit`
Expected: aucune erreur.

- [ ] **Step 4: Lancer la suite de tests complète du plugin**

Run: `cd ~/Secretarius/Wiki_LM/obsidian-wikilm-capture && npm test`
Expected: PASS (`capture-text.test.ts` + `run-commands.test.ts`).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/obsidian-wikilm-capture/src/main.ts Wiki_LM/obsidian-wikilm-capture/main.js
git commit -m "$(cat <<'EOF'
feat(wikilm-capture): nouvelle commande d'exécution en lot des blocs wiki

Nouvelle entrée dédiée (palette + icône barre latérale), distincte de la
capture de note existante. Extrait les blocs \`\`\`wiki de la note active,
appelle POST /run séquentiellement pour chacun, et réécrit la note en une
seule fois avec les résultats insérés sous chaque bloc (une commande en
échec n'interrompt pas les suivantes).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

- [ ] **Step 6: Vérification manuelle bout en bout (à faire par l'utilisateur)**

Cette étape nécessite un accès à l'application Obsidian graphique (tablette),
donc hors de portée de l'agent — comme la checklist manuelle laissée en
attente pour `wikilm-capture` lors de sa première livraison. Rappel de la
procédure :

1. Recopier le plugin buildé (`main.js`, `manifest.json`, `styles.css` s'il
   existe) dans le dossier `.obsidian/plugins/wikilm-capture/` du coffre
   synchronisé (ou relancer `ob sync` si le build est versionné et
   synchronisé automatiquement).
2. Dans Obsidian sur la tablette, créer une note de test avec deux blocs
   ```` ```wiki ```` (par exemple `/tags` et `/wikistatus`, sans effet de
   bord), lancer la commande « Exécuter les commandes wiki de la note » via
   la palette de commandes.
3. Vérifier que les deux résultats apparaissent sous chaque bloc, encadrés
   par les marqueurs `<!-- wikilm-run:start -->`/`<!-- wikilm-run:end -->`.
4. Relancer la même commande sur la même note, vérifier que les résultats
   sont remplacés et non dupliqués.
5. Ajouter un bloc avec une commande volontairement absente de la table
   (`/supprimer` ou une commande inventée) et vérifier qu'un message
   d'erreur clair apparaît sans qu'aucune requête n'ait été envoyée au
   serveur (observable via les logs du service `wiki-lm-server`, qui ne
   doivent montrer aucune requête `/run` pour ce bloc).

---

## Self-Review

**Couverture de la spec :**
- Format des blocs ```` ```wiki ```` (commande + argument multi-lignes,
  bloc sans argument, texte hors bloc ignoré) → Task 2, `extractWikiBlocks`
  + tests.
- Table commande → opération (9 entrées) → Task 1 `_RUN_OPS` et Task 2
  `SUPPORTED_COMMANDS`, gardées synchronisées (mêmes 9 valeurs, vérifiées
  chacune de son côté par ses propres tests).
- `/supprimer` explicitement refusée côté serveur, jamais dispatchée → Task 1
  Step 3 (absence de la clé) + test dédié.
- Route `POST /run`, renvoie le JSON tel quel → Task 1.
- Nouvelle commande plugin dédiée (palette + icône) → Task 3 Step 1.
- Résultat inséré sous chaque bloc, dans la note → Task 3 Step 2 +
  `applyWikiResults` (Task 2).
- Continue en cas d'échec → Task 3 `runOneCommand` (try/catch + `throw:
  false`, jamais d'exception qui interromprait la boucle `for`).
- Relance = remplacement, pas duplication, texte utilisateur préservé →
  Task 2 `extractWikiBlocks` (span inclut le marqueur existant) + tests
  dédiés (`includes a pre-existing result marker…`, `replaces a prior
  result…`).
- Formatage par type de commande (9 cas + erreur) → Task 2
  `formatWikiResult` + tests, un par commande.
- Hors périmètre (gog/scout, `/supprimer` en lot, formatage riche, relance
  partielle) : aucune tâche ne les implémente — conforme.
- Tests serveur (dispatch de chaque commande + refus `/supprimer`) → Task 1.
- Tests plugin (extraction multi-blocs, argument multi-lignes, bloc sans
  argument, texte hors bloc, formatage par type) → Task 2.

**Balayage des placeholders :** aucun "TBD"/"TODO" — toutes les étapes
contiennent du code complet et exécutable, tous les tests ont des assertions
concrètes.

**Cohérence des types/signatures :** `WikiBlockMatch` (Task 2) utilisé tel
quel dans `runOneCommand` (Task 3) ; `SUPPORTED_COMMANDS`, `extractWikiBlocks`,
`applyWikiResults`, `formatWikiResult` ont la même signature à leur
définition (Task 2) et à leur usage (Task 3). Les 9 commandes de `_RUN_OPS`
(Task 1, Python) et de `SUPPORTED_COMMANDS` (Task 2, TypeScript) sont
identiques terme à terme.

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-11-lot-commandes-obsidian.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
