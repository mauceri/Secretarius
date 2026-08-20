# Plugin Obsidian de capture de la note courante — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un plugin Obsidian qui capture la note actuellement ouverte vers la file `raw/` de Wiki_LM (équivalent de `/c` pour une note déjà rédigée dans le coffre), plus le nouvel endpoint serveur qui le reçoit.

**Architecture:** Le plugin (TypeScript, bundlé avec esbuild) extrait un texte de capture depuis la note active (section `# Résumé`/`# Summary` verbatim, sinon incipit 200 caractères) via une fonction pure et testée (`buildCaptureText`), puis l'envoie par `requestUrl` à un nouvel endpoint `POST /capture` sur `wiki-lm-server` (Flask, déjà actif en permanence sur sanroque, port 5051). Le serveur réutilise `capture_comment()` de `capture.py` sans le modifier.

**Tech Stack:** Python 3 / Flask / pytest côté serveur (`~/Secretarius/Wiki_LM/.venv/bin/pytest`) ; TypeScript / esbuild / vitest côté plugin (nouveau projet `obsidian-wikilm-capture/` à la racine du dépôt, npm/node déjà disponibles).

## Global Constraints

- Aucune dépendance sur la directive `@simple` (sous-projet 1, déjà livré) : les captures texte sont déjà toujours verbatim dans `ingest.py`, indépendamment de tout marqueur `simple:`.
- Contrat serveur : `POST /capture`, body `{"text": "...", "tags": ["..."]}`, réponse `{"status": "ok", "filename": "..."}` ; `text` vide → 400. Le texte reçu est déjà entièrement construit côté plugin (préfixe de traçabilité inclus) — le serveur n'a besoin d'aucun champ `title`/`path` séparé.
- Règle de contenu (implémentée dans `buildCaptureText`) : si le premier titre Markdown non vide après le frontmatter est de **niveau 1 exactement** et que son texte (insensible à la casse) est « Résumé » ou « Summary », copier tout le contenu de cette section verbatim (jusqu'au titre suivant, quel que soit son niveau, ou la fin de note). Sinon, prendre les 200 premiers caractères du corps, coupés au dernier espace avant la limite, suffixés par « … ». Toujours préfixer par `Note d'origine : <titre> (<chemin>)\n\n`.
- Pas de test automatisé pour l'intégration Obsidian elle-même (API non mockable simplement dans ce dépôt) — mais la logique pure de construction du texte (`buildCaptureText`) DOIT être testée avec vitest, car elle ne dépend d'aucune API Obsidian.
- `requestUrl` (API Obsidian) obligatoire pour l'appel HTTP côté plugin — jamais `fetch()` (bloqué par le CSP d'Electron, contrainte déjà documentée dans `docs/components/obsidian.md`).
- Commandes de test : `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/<fichier> -v` (Python) ; `cd ~/Secretarius/obsidian-wikilm-capture && npm test` (TypeScript, une fois le projet scaffoldé en Task 3).

---

### Task 1: `capture.py` — extraction du helper `raw_dir()`

**Files:**
- Modify: `Wiki_LM/tools/capture.py:37-38` (nouvelle fonction) et `Wiki_LM/tools/capture.py:260-263` (`main()`)
- Test: `Wiki_LM/tests/test_capture.py`

**Interfaces:**
- Produces: `raw_dir() -> Path` — résout `WIKI_RAW_PATH`/`RAW_DEFAULT`, crée le dossier si absent, le retourne. Consommée par la Task 2.

- [ ] **Step 1: Write the failing test**

Ajouter dans `Wiki_LM/tests/test_capture.py`, en étendant l'import existant (ligne 7) :

```python
from capture import (
    capture_urls, capture_comment, capture_file, _normalize_url,
    _parse_simple_directive, raw_dir,
)
```

Ajouter une nouvelle classe, par exemple juste avant `class TestNormalizeUrl:` :

```python
class TestRawDir:
    def test_uses_env_var_and_creates_dir(self, tmp_path, monkeypatch):
        target = tmp_path / "raw_env"
        monkeypatch.setenv("WIKI_RAW_PATH", str(target))
        result = raw_dir()
        assert result == target
        assert target.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_capture.py::TestRawDir -v`
Expected: FAIL — `ImportError: cannot import name 'raw_dir'`

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/capture.py`, ajouter cette fonction juste après la ligne 37 (`_TAG_NORMALIZE_THRESHOLD = 0.85`) :

```python
def raw_dir() -> Path:
    """Résout et crée le dossier raw/ (WIKI_RAW_PATH ou défaut)."""
    d = Path(os.environ.get("WIKI_RAW_PATH", str(RAW_DEFAULT))).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d
```

Puis remplacer dans `main()` (lignes 260-263) :

```python
def main() -> None:
    import os
    raw = Path(os.environ.get("WIKI_RAW_PATH", str(RAW_DEFAULT))).expanduser()
    raw.mkdir(parents=True, exist_ok=True)
```

par :

```python
def main() -> None:
    raw = raw_dir()
```

(Le `import os` local disparaît : `os` est déjà importé au niveau module, ligne 25 — cet import local était redondant.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_capture.py -v`
Expected: PASS (29 tests passants)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/capture.py Wiki_LM/tests/test_capture.py
git commit -m "refactor: extraire raw_dir() de capture.py pour réutilisation par le serveur"
```

---

### Task 2: `server.py` — endpoint `POST /capture`

**Files:**
- Modify: `Wiki_LM/tools/server.py:1-13` (docstring), `Wiki_LM/tools/server.py:26-28` (imports), `Wiki_LM/tools/server.py:68-70` (nouvelle route)
- Test: `Wiki_LM/tests/test_server.py` (nouveau)

**Interfaces:**
- Consumes: `raw_dir()` (Task 1), `capture_comment`, `_normalize_tags` (existants dans `capture.py`, inchangés).
- Produces: route Flask `POST /capture` sur `app` — consommée par le plugin (Task 4), via HTTP, pas d'import direct.

- [ ] **Step 1: Write the failing tests**

Créer `Wiki_LM/tests/test_server.py` :

```python
"""Tests de l'endpoint /capture de server.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def raw_path(tmp_path, monkeypatch):
    target = tmp_path / "raw"
    monkeypatch.setenv("WIKI_RAW_PATH", str(target))
    return target


class TestHandleCapture:
    def test_missing_text_returns_400(self, client, raw_path):
        response = client.post("/capture", json={"tags": ["ia"]})
        assert response.status_code == 400

    def test_writes_file_with_text_and_tags(self, client, raw_path):
        response = client.post("/capture", json={
            "text": "Note d'origine : Ma note (dossier/ma-note.md)\n\nContenu de test.",
            "tags": ["documentation"],
        })
        assert response.status_code == 200
        data = response.get_json()
        created = raw_path / data["filename"]
        assert created.exists()
        content = created.read_text(encoding="utf-8")
        assert "Contenu de test." in content
        assert "documentation" in content

    def test_returns_created_filename(self, client, raw_path):
        response = client.post("/capture", json={"text": "Contenu minimal."})
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["filename"].endswith(".md")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_server.py -v`
Expected: FAIL — `404 NOT FOUND` (la route `/capture` n'existe pas encore)

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/server.py`, modifier le docstring du module (lignes 1-13), remplacer :

```python
"""
Serveur Flask local pour interroger Wiki_LM depuis Obsidian.

Usage :
    python server.py [--port 5051] [--mode hybrid]

Endpoint :
    POST /query
    Body  : {"question": "...", "top_k": 5, "save": false, "mode": "hybrid"}
    Reply : {"text": "...", "references": [...], "saved_slug": ""}

    GET /health
    Reply : {"status": "ok", "pages": <n>}
"""
```

par :

```python
"""
Serveur Flask local pour interroger Wiki_LM depuis Obsidian.

Usage :
    python server.py [--port 5051] [--mode hybrid]

Endpoint :
    POST /query
    Body  : {"question": "...", "top_k": 5, "save": false, "mode": "hybrid"}
    Reply : {"text": "...", "references": [...], "saved_slug": ""}

    POST /capture
    Body  : {"text": "...", "tags": ["..."]}
    Reply : {"status": "ok", "filename": "..."}

    GET /health
    Reply : {"status": "ok", "pages": <n>}
"""
```

Modifier les imports (lignes 26-29), remplacer :

```python
from llm import LLM
from query import WikiQuery
from cluster import run_clustering
```

par :

```python
from llm import LLM
from query import WikiQuery
from cluster import run_clustering
from capture import capture_comment, _normalize_tags, raw_dir
```

Ajouter la nouvelle route juste après `handle_query` (après la ligne `    })` qui clôt la fonction, avant `@app.get("/health")`) :

```python
@app.post("/capture")
def handle_capture():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Paramètre 'text' manquant"}), 400
    tags_raw = [str(t) for t in (data.get("tags") or [])]
    tags = _normalize_tags(tags_raw) if tags_raw else []
    path = capture_comment(text, raw_dir(), tags=tags or None)
    return jsonify({"status": "ok", "filename": path.name})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_server.py -v`
Expected: PASS (3 tests)

Puis la suite complète :

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest -q`
Expected: PASS, 0 failure

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/server.py Wiki_LM/tests/test_server.py
git commit -m "feat: endpoint POST /capture sur wiki-lm-server"
```

---

### Task 3: Scaffold du plugin Obsidian + logique pure `buildCaptureText`

**Files:**
- Create: `obsidian-wikilm-capture/package.json`
- Create: `obsidian-wikilm-capture/tsconfig.json`
- Create: `obsidian-wikilm-capture/src/capture-text.ts`
- Test: `obsidian-wikilm-capture/src/capture-text.test.ts`

**Interfaces:**
- Produces: `buildCaptureText({ body, title, path }: NoteInput) -> string`. Consommée par la Task 4.

- [ ] **Step 1: Create the project scaffold**

Créer `obsidian-wikilm-capture/package.json` :

```json
{
  "name": "obsidian-wikilm-capture",
  "version": "0.1.0",
  "private": true,
  "description": "Capture la note Obsidian courante dans la file raw/ de Wiki_LM.",
  "main": "main.js",
  "scripts": {
    "build": "node esbuild.config.mjs production",
    "dev": "node esbuild.config.mjs",
    "test": "vitest run"
  },
  "devDependencies": {
    "@types/node": "^24.0.0",
    "esbuild": "^0.24.0",
    "obsidian": "latest",
    "typescript": "^5.9.0",
    "vitest": "^3.2.0"
  }
}
```

Créer `obsidian-wikilm-capture/tsconfig.json` :

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "inlineSourceMap": true,
    "inlineSources": true,
    "module": "ESNext",
    "target": "ES6",
    "allowJs": true,
    "noImplicitAny": true,
    "moduleResolution": "node",
    "importHelpers": true,
    "isolatedModules": true,
    "strict": true,
    "lib": ["DOM", "ES5", "ES6", "ES7"]
  },
  "include": ["src/**/*.ts"]
}
```

Installer les dépendances :

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm install`
Expected: installation réussie, `node_modules/` créé

- [ ] **Step 2: Write the failing tests**

Créer `obsidian-wikilm-capture/src/capture-text.test.ts` :

```typescript
import { describe, expect, it } from "vitest";
import { buildCaptureText } from "./capture-text";

describe("buildCaptureText", () => {
  it("always prefixes with the origin line", () => {
    const result = buildCaptureText({
      body: "Courte note.",
      title: "Ma note",
      path: "dossier/ma-note.md",
    });
    expect(result.startsWith("Note d'origine : Ma note (dossier/ma-note.md)\n\n")).toBe(true);
  });

  it("returns the body as-is when shorter than the incipit length", () => {
    const result = buildCaptureText({ body: "Courte note.", title: "T", path: "p.md" });
    expect(result.endsWith("Courte note.")).toBe(true);
    expect(result).not.toContain("…");
  });

  it("truncates long bodies at the last space before 200 characters", () => {
    const body = "mot ".repeat(60).trim();
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    const content = result.split("\n\n")[1];
    expect(content.endsWith("…")).toBe(true);
    expect(content.length).toBeLessThanOrEqual(201);
    expect(content).not.toMatch(/ …$/);
  });

  it("extracts a level-1 Résumé section verbatim, ignoring length limit", () => {
    const longSummary = "Phrase de résumé assez longue pour dépasser deux cents caractères si on la répète. ".repeat(5).trim();
    const body = `# Résumé\n\n${longSummary}\n\n## Autre section\n\nIgnoré.`;
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    const content = result.slice(result.indexOf("\n\n") + 2);
    expect(content.trim()).toBe(longSummary);
    expect(content).not.toContain("Ignoré");
  });

  it("matches '# Summary' case-insensitively", () => {
    const body = "# summary\nHello world.";
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    expect(result.endsWith("Hello world.")).toBe(true);
  });

  it("does not treat a level-2 heading as a summary section", () => {
    const body = "## Résumé\nTexte court.";
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    expect(result.endsWith("## Résumé\nTexte court.")).toBe(true);
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm test`
Expected: FAIL — `Cannot find module './capture-text'`

- [ ] **Step 4: Write minimal implementation**

Créer `obsidian-wikilm-capture/src/capture-text.ts` :

```typescript
export interface NoteInput {
  body: string;
  title: string;
  path: string;
}

const INCIPIT_LENGTH = 200;
const SUMMARY_HEADINGS = ["résumé", "summary"];

export function buildCaptureText({ body, title, path }: NoteInput): string {
  const content = extractContent(body);
  return `Note d'origine : ${title} (${path})\n\n${content}`;
}

function extractContent(body: string): string {
  const summary = extractSummarySection(body);
  return summary !== null ? summary : truncateIncipit(body);
}

function extractSummarySection(body: string): string | null {
  const lines = body.split("\n");
  let i = 0;
  while (i < lines.length && lines[i].trim() === "") i++;
  if (i >= lines.length) return null;

  const headingMatch = lines[i].match(/^#\s+(.+)$/);
  if (!headingMatch) return null;
  if (!SUMMARY_HEADINGS.includes(headingMatch[1].trim().toLowerCase())) return null;

  const sectionLines: string[] = [];
  for (let j = i + 1; j < lines.length; j++) {
    if (/^#{1,6}\s+/.test(lines[j])) break;
    sectionLines.push(lines[j]);
  }
  return sectionLines.join("\n").trim();
}

function truncateIncipit(body: string): string {
  const trimmed = body.trim();
  if (trimmed.length <= INCIPIT_LENGTH) return trimmed;
  const slice = trimmed.slice(0, INCIPIT_LENGTH);
  const lastSpace = slice.lastIndexOf(" ");
  const cut = lastSpace > 0 ? slice.slice(0, lastSpace) : slice;
  return `${cut}…`;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm test`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius
git add obsidian-wikilm-capture/package.json obsidian-wikilm-capture/package-lock.json \
        obsidian-wikilm-capture/tsconfig.json obsidian-wikilm-capture/src/capture-text.ts \
        obsidian-wikilm-capture/src/capture-text.test.ts
git commit -m "feat: scaffold plugin Obsidian + buildCaptureText (résumé/incipit)"
```

(Ne pas ajouter `node_modules/` — créer `obsidian-wikilm-capture/.gitignore` avec le contenu `node_modules/\nmain.js\n*.js.map\n` avant le `git add` s'il n'existe pas déjà, pour éviter de le commiter par erreur.)

---

### Task 4: Plugin Obsidian — glue et déclenchement

**Files:**
- Create: `obsidian-wikilm-capture/manifest.json`
- Create: `obsidian-wikilm-capture/esbuild.config.mjs`
- Create: `obsidian-wikilm-capture/src/main.ts`
- Modify: `docs/components/obsidian.md` (nouvelle section d'installation)

**Interfaces:**
- Consumes: `buildCaptureText` (Task 3), endpoint `POST /capture` (Task 2, via HTTP).

- [ ] **Step 1: Create the manifest**

Créer `obsidian-wikilm-capture/manifest.json` :

```json
{
  "id": "wikilm-capture",
  "name": "Wiki_LM Capture",
  "version": "0.1.0",
  "minAppVersion": "1.4.0",
  "description": "Capture la note courante dans la file raw/ de Wiki_LM.",
  "author": "Christian Mauceri",
  "isDesktopOnly": false
}
```

- [ ] **Step 2: Create the esbuild config**

Créer `obsidian-wikilm-capture/esbuild.config.mjs` :

```javascript
import esbuild from "esbuild";
import process from "process";

const production = process.argv[2] === "production";

const context = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: ["obsidian", "electron"],
  format: "cjs",
  target: "es2018",
  logLevel: "info",
  sourcemap: production ? false : "inline",
  treeShaking: true,
  outfile: "main.js",
  minify: production,
});

if (production) {
  await context.rebuild();
  process.exit(0);
} else {
  await context.watch();
}
```

- [ ] **Step 3: Write the plugin entry point**

Créer `obsidian-wikilm-capture/src/main.ts` :

```typescript
import {
  App,
  CachedMetadata,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  requestUrl,
} from "obsidian";
import { buildCaptureText } from "./capture-text";

interface WikilmCaptureSettings {
  serverUrl: string;
}

const DEFAULT_SETTINGS: WikilmCaptureSettings = {
  serverUrl: "http://sanroque:5051",
};

function stripFrontmatter(raw: string, cache: CachedMetadata | null): string {
  const pos = cache?.frontmatterPosition;
  if (!pos) return raw;
  return raw.slice(pos.end.offset).replace(/^\s*\n/, "");
}

function extractTags(cache: CachedMetadata | null): string[] {
  const tags = cache?.frontmatter?.tags;
  if (!tags) return [];
  return Array.isArray(tags) ? tags.map(String) : [String(tags)];
}

export default class WikilmCapturePlugin extends Plugin {
  settings: WikilmCaptureSettings = DEFAULT_SETTINGS;

  async onload() {
    await this.loadSettings();
    this.addSettingTab(new WikilmCaptureSettingTab(this.app, this));
    this.addRibbonIcon("upload", "Capturer dans Wiki_LM", () => this.captureCurrentNote());
    this.addCommand({
      id: "capture-current-note",
      name: "Capturer la note courante dans Wiki_LM",
      callback: () => this.captureCurrentNote(),
    });
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  async captureCurrentNote(): Promise<void> {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("Aucune note ouverte");
      return;
    }

    const cache = this.app.metadataCache.getFileCache(file);
    const raw = await this.app.vault.read(file);
    const body = stripFrontmatter(raw, cache);
    const tags = extractTags(cache);
    const text = buildCaptureText({ body, title: file.basename, path: file.path });

    if (cache?.frontmatter?.wiki_capture) {
      new Notice(`Déjà capturée le ${cache.frontmatter.wiki_capture} — nouvelle capture en cours…`);
    }

    try {
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/capture`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ text, tags }),
      });
      const data = response.json as { filename: string };
      await this.app.fileManager.processFrontMatter(file, (fm) => {
        fm.wiki_capture = new Date().toISOString();
      });
      new Notice(`Capturée : ${data.filename}`);
    } catch (err) {
      new Notice(`Erreur de capture : ${err}`);
    }
  }
}

class WikilmCaptureSettingTab extends PluginSettingTab {
  plugin: WikilmCapturePlugin;

  constructor(app: App, plugin: WikilmCapturePlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("URL du serveur Wiki_LM")
      .setDesc("Adresse du serveur wiki-lm-server (ex. http://sanroque:5051)")
      .addText((text) =>
        text
          .setPlaceholder("http://sanroque:5051")
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value.trim();
            await this.plugin.saveSettings();
          })
      );
  }
}
```

- [ ] **Step 4: Verify the build compiles**

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm run build`
Expected: se termine sans erreur, produit `obsidian-wikilm-capture/main.js`

Run à nouveau la suite vitest pour confirmer qu'aucune régression n'a été introduite :

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm test`
Expected: PASS (6 tests, inchangé depuis la Task 3)

- [ ] **Step 5: Document installation**

Dans `docs/components/obsidian.md`, ajouter une nouvelle section après la section « Template de requête Wiki_LM (Templater) » (avant « ## Archivage du coffre ») :

```markdown
## Plugin de capture Wiki_LM

Capturer la note actuellement ouverte dans la file `raw/` de Wiki_LM (équivalent
de `/c`, sans quitter Obsidian), depuis desktop ou mobile. Source :
`obsidian-wikilm-capture/` (projet TypeScript séparé, à la racine du dépôt).

### Prérequis

- Service `wiki-lm-server` actif sur sanroque (le plugin appelle le nouvel
  endpoint `POST /capture`, voir `docs/components/wiki-lm.md`).
- L'appareil Obsidian atteint `sanroque:5051` (réseau local ou Tailscale).

### Installation

1. `cd obsidian-wikilm-capture && npm install && npm run build` — produit
   `main.js` à côté de `manifest.json`.
2. Copier `manifest.json` et `main.js` dans
   `<coffre>/.obsidian/plugins/wikilm-capture/`.
3. Dans Obsidian : Paramètres → Modules communautaires → activer
   « Wiki_LM Capture ».
4. Dans les réglages du plugin, renseigner l'URL du serveur (ex.
   `http://sanroque:5051`).

### Utilisation

1. Ouvrir la note à capturer.
2. Cliquer l'icône dans la barre latérale, ou lancer la commande
   « Capturer la note courante dans Wiki_LM » (Ctrl/Cmd-P).
3. Si la note commence par un titre `# Résumé` ou `# Summary`, cette section
   est capturée intégralement ; sinon, les 200 premiers caractères de la note
   sont utilisés comme aperçu.
4. Une notification confirme la capture ; la note est marquée
   `wiki_capture: <date>` dans son frontmatter.
```

- [ ] **Step 6: Commit**

```bash
cd ~/Secretarius
git add obsidian-wikilm-capture/manifest.json obsidian-wikilm-capture/esbuild.config.mjs \
        obsidian-wikilm-capture/src/main.ts docs/components/obsidian.md
git commit -m "feat: plugin Obsidian de capture de la note courante (ribbon + commande)"
```

---

### Task 5: Vérification finale

**Files:** aucun fichier modifié — vérification uniquement.

- [ ] **Step 1: Run both test suites**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest -q`
Expected: PASS, 0 failure (baseline 313 + 4 nouveaux tests serveur + 1 nouveau test `raw_dir` = 318)

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm test`
Expected: PASS, 6 tests

Run: `cd ~/Secretarius/obsidian-wikilm-capture && npm run build`
Expected: se termine sans erreur

- [ ] **Step 2: Manual end-to-end checklist (à exécuter par l'utilisateur, pas par l'agent)**

L'agent n'a pas d'accès à une instance Obsidian graphique — cette étape ne peut
pas être auto-vérifiée et doit être confirmée par l'utilisateur après
installation réelle (voir `docs/components/obsidian.md`, section « Plugin de
capture Wiki_LM ») :

1. Ouvrir une note sans section `# Résumé` → capturer → vérifier dans
   `Wiki_LM/raw/` un fichier `.md` contenant les 200 premiers caractères de la
   note, précédés de la ligne `Note d'origine : ...`.
2. Ouvrir une note commençant par `# Résumé` → capturer → vérifier que le
   fichier `raw/` contient l'intégralité de cette section, pas juste 200
   caractères.
3. Recapturer la même note → vérifier la notification d'avertissement
   (non bloquante) et qu'une seconde capture est bien créée.
4. Vérifier que le frontmatter de la note source contient `wiki_capture:`
   après une capture réussie.
5. `/ingest` (ou `ingest_raw_dir()`) sur les fichiers créés → vérifier que les
   pages produites sont verbatim (pas de section `## Résumé` générée par LLM).

- [ ] **Step 3: Report**

Aucun commit pour cette tâche (vérification seule). Si une étape manuelle
échoue, revenir à la tâche concernée avant de considérer le plan terminé.
