# Historique des requêtes + régime par canal (/q) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every `/q` answer writes a timestamped history note in `Wiki_LM/historique/`, and each channel gets the right regime — Telegram gets a short LLM-generated summary plus an `obsidian://` link (never a wall of raw Markdown), WebChat and Obsidian keep the full synthesis (WebChat renders Markdown fine; Obsidian opens the history note in a new tab instead of inserting it into the note you're reading).

**Architecture:** `WikiQuery.query()` in `query.py` is the single write point — it now always writes the history note and generates the short brief, and both callers (`wiki.py`'s CLI/sandbox facade used by Telegram/WebChat, and `server.py`'s Flask facade used by Obsidian) inherit this for free. `derisk-deleg` reads `ctx.messageProvider` (a hook parameter it currently ignores) to pick `"full"` for WebChat and `"brief"` everywhere else, and threads that choice through `runWikiOp`/`formatWikiResult`. The two Obsidian templates stop building and inserting a text block; they just open the history note the server already wrote.

**Tech Stack:** Python 3 (pytest, dataclasses), TypeScript (vitest), Flask, Obsidian Templater (JS in `<%* %>` blocks).

**Spec:** `docs/superpowers/specs/2026-09-08-historique-requetes-double-regime-design.md`

## Global Constraints

- History notes are written **unconditionally** on every `query()` call — never behind a flag (spec "Décisions actées").
- `historique/` is a sibling of `wiki/` and `raw/` under `WIKI_PATH`, **outside** `WikiSearch`'s indexed tree (`wiki_root/wiki`) and outside the ingestion queue (`raw/`) — never write history notes inside `wiki/`.
- The existing `--save`/`saved_slug`/`_save_synth` mechanism in `query.py` is untouched — it is a separate, opt-in feature (writes an LLM-elaborated page **inside** `wiki/`, indexed and permanent). Do not merge it with the new mechanism.
- The brief-generation LLM call must **never** raise out of `query()` — on failure, fall back to `synthesis[:300]`.
- Regime defaults to `"brief"` everywhere the channel isn't identifiable (including the LLM-invoked `wiki_query`/etc. tool path, which has no channel context available in the SDK) — `"full"` is the explicit exception, only for `ctx.messageProvider === "webchat"`.
- `/r` (search) and the existing `--save` CLI flag are out of scope — do not modify their behavior.

---

### Task 1: `query.py` — history note + brief, single write point

**Files:**
- Modify: `Wiki_LM/tools/query.py:18-30` (imports), `:37-47` (`QueryResult`), `:151-158` (`query()` tail), add two new private methods near `:220` (after `_append_log`)
- Test: `Wiki_LM/tests/test_query.py` (new file)

**Interfaces:**
- Consumes: `capture.slugify(text: str, max_words: int = 6) -> str`, `capture.timestamp() -> str` (both already exist in `Wiki_LM/tools/capture.py:95` and `:102`); `LLM.complete(self, prompt: str = "", *, messages=None, system: str = "", max_tokens: int = 2048) -> str` (`Wiki_LM/tools/llm.py:180`).
- Produces: `QueryResult.history_slug: str`, `QueryResult.brief: str` (new dataclass fields, both default `""`) — consumed by Task 2 and Task 3.

- [ ] **Step 1: Write the failing tests**

Create `Wiki_LM/tests/test_query.py`:

```python
"""Tests de WikiQuery.query() : historique horodaté + résumé bref."""

from __future__ import annotations

from search import SearchResult
from query import WikiQuery


class _StubLLM:
    def __init__(self, brief: str = "Résumé bref.", fail_brief: bool = False) -> None:
        self.calls: list[dict] = []
        self._brief = brief
        self._fail_brief = fail_brief

    def complete(self, prompt: str = "", *, messages=None, system: str = "", max_tokens: int = 2048) -> str:
        self.calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        if len(self.calls) == 1:
            return "Synthèse test avec [[c-test]]."
        if self._fail_brief:
            raise RuntimeError("LLM indisponible")
        return self._brief


def _make_query(tmp_path, llm=None) -> WikiQuery:
    (tmp_path / "wiki").mkdir()
    page = tmp_path / "wiki" / "c-test.md"
    page.write_text("Contenu de test.", encoding="utf-8")
    wq = WikiQuery(tmp_path, llm=llm or _StubLLM(), mode="bm25")
    result = SearchResult(slug="c-test", path=page, title="Test", category="concept",
                          score=1.0, excerpt="Contenu de test.")
    wq._search.search = lambda q, top_k=5: [result]
    return wq


def test_query_writes_history_note(tmp_path):
    wq = _make_query(tmp_path)
    result = wq.query("Question test ?")

    assert result.history_slug
    history_files = list((tmp_path / "historique").glob("*.md"))
    assert len(history_files) == 1
    assert history_files[0].stem == result.history_slug
    content = history_files[0].read_text(encoding="utf-8")
    assert "Question test ?" in content
    assert "Synthèse test avec [[c-test]]." in content


def test_query_generates_brief(tmp_path):
    wq = _make_query(tmp_path, llm=_StubLLM(brief="Un court résumé."))
    result = wq.query("Question test ?")

    assert result.brief == "Un court résumé."


def test_query_brief_falls_back_to_truncation_on_llm_failure(tmp_path):
    wq = _make_query(tmp_path, llm=_StubLLM(fail_brief=True))
    result = wq.query("Question test ?")

    assert result.brief == result.text[:300]


def test_query_save_flag_unaffected(tmp_path):
    wq = _make_query(tmp_path)
    result = wq.query("Question test ?", save=True)

    assert result.saved_slug
    assert (tmp_path / "wiki" / f"{result.saved_slug}.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `~/Secretarius/Wiki_LM`): `.venv/bin/pytest tests/test_query.py -v`
Expected: FAIL — `AttributeError: 'QueryResult' object has no attribute 'history_slug'` (or similar) on every test.

- [ ] **Step 3: Add the two new fields to `QueryResult`**

In `Wiki_LM/tools/query.py`, replace:

```python
@dataclass
class QueryResult:
    question: str
    text: str                          # synthèse en Markdown
    references: list[str] = field(default_factory=list)   # slugs utilisés
    saved_slug: str = ""               # slug de la page synth- si --save
```

with:

```python
@dataclass
class QueryResult:
    question: str
    text: str                          # synthèse en Markdown
    references: list[str] = field(default_factory=list)   # slugs utilisés
    saved_slug: str = ""               # slug de la page synth- si --save
    history_slug: str = ""             # slug de l'enregistrement horodaté (historique/)
    brief: str = ""                    # résumé court (canaux à espace limité, ex. Telegram)
```

- [ ] **Step 4: Add the `capture` import**

In `Wiki_LM/tools/query.py`, replace:

```python
from llm import LLM
from search import WikiSearch, WikiSemanticSearch, hybrid_search
```

with:

```python
from capture import slugify, timestamp
from llm import LLM
from search import WikiSearch, WikiSemanticSearch, hybrid_search
```

- [ ] **Step 5: Add the brief prompt constants**

In `Wiki_LM/tools/query.py`, after `_PROMPT_SYNTH_PAGE` (ends around line 97), add:

```python
_SYSTEM_BRIEF = """\
Tu résumes une réponse de wiki personnel en 1 à 2 phrases très courtes, \
pour un message de chat. Pas de Markdown, pas de citations [[slug]], \
juste le sens en langage naturel."""

_PROMPT_BRIEF = """\
Question : {question}

Réponse complète :
---
{synthesis}
---

Résume cette réponse en 1 à 2 phrases courtes."""
```

- [ ] **Step 6: Add `_write_history` and `_generate_brief`, wire them into `query()`**

In `Wiki_LM/tools/query.py`, replace:

```python
        result = QueryResult(question=question, text=synthesis, references=references)

        # 5. Optionnel : sauvegarder comme page synth-
        if save:
            result.saved_slug = self._save_synth(question, synthesis, references)
            self._append_log("query", question)

        return result
```

with:

```python
        result = QueryResult(question=question, text=synthesis, references=references)
        result.history_slug = self._write_history(question, str(result))
        result.brief = self._generate_brief(question, synthesis)

        # 5. Optionnel : sauvegarder comme page synth-
        if save:
            result.saved_slug = self._save_synth(question, synthesis, references)
            self._append_log("query", question)

        return result
```

Then, in `Wiki_LM/tools/query.py`, after `_append_log` (the last helper method, right before the `# CLI` section comment), add:

```python
    def _write_history(self, question: str, content: str) -> str:
        """Enregistre la requête et sa réponse complète, horodaté, hors de
        l'arbre indexé (jamais dans wiki/, jamais vu par la recherche ou
        l'ingestion). Retourne le slug (sans extension)."""
        history_dir = self.wiki_root / "historique"
        history_dir.mkdir(parents=True, exist_ok=True)
        slug = f"{timestamp()}-{slugify(question)}"
        (history_dir / f"{slug}.md").write_text(content, encoding="utf-8")
        return slug

    def _generate_brief(self, question: str, synthesis: str) -> str:
        """Résumé court pour les canaux à espace limité (ex. Telegram). Ne
        doit jamais faire échouer query() : repli sur une troncature simple
        si l'appel LLM échoue."""
        try:
            prompt = _PROMPT_BRIEF.format(question=question, synthesis=synthesis)
            return self.llm.complete(prompt, system=_SYSTEM_BRIEF, max_tokens=120).strip()
        except Exception:
            return synthesis[:300]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_query.py -v`
Expected: PASS (4 tests).

- [ ] **Step 8: Run the full Wiki_LM suite to check for regressions**

Run: `.venv/bin/pytest -q`
Expected: PASS, same count as before plus 4 new (no existing test references `QueryResult` positionally with all fields, so the two new defaulted fields should not break anything — confirm none do).

- [ ] **Step 9: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/query.py Wiki_LM/tests/test_query.py
git commit -m "$(cat <<'EOF'
feat(wiki_lm): historique horodaté + résumé bref sur toute requête /q

WikiQuery.query() écrit désormais systématiquement un enregistrement dans
historique/ (hors index) et génère un résumé court via un second appel LLM
léger (repli sur troncature si l'appel échoue). Point d'écriture unique :
wiki.py et server.py en hériteront sans dupliquer la logique.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 2: `wiki.py::op_query` — expose `brief` and `obsidian_uri`

**Files:**
- Modify: `Wiki_LM/tools/wiki.py:7-13` (imports), `:92-99` (`op_query`)
- Test: `Wiki_LM/tests/test_wiki_cli.py:48-64` (existing `test_query_returns_synthesis`, modify)

**Interfaces:**
- Consumes: `QueryResult.history_slug`, `QueryResult.brief` (Task 1).
- Produces: `op_query(question: str) -> dict` now returns `{"synthesis": str, "references": list[str], "brief": str, "obsidian_uri": str}` on success (unchanged `{"error": str}` on failure) — consumed by Task 4 (`formatWikiResult`'s `"query"` case reads `json.brief`/`json.obsidian_uri`). New helper `_build_obsidian_uri(history_slug: str) -> str`.

- [ ] **Step 1: Write the failing test**

In `Wiki_LM/tests/test_wiki_cli.py`, replace the existing `test_query_returns_synthesis`:

```python
def test_query_returns_synthesis(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = "Synthèse."
        references = ["src-a"]

    class _Q:
        def __init__(self, *a, **k):
            pass

        def query(self, q, top_k=5):
            return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    out = wiki.op_query("question ?")
    assert out == {"synthesis": "Synthèse.", "references": ["src-a"]}
```

with:

```python
def test_query_returns_synthesis(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        text = "Synthèse."
        references = ["src-a"]
        history_slug = "20260908-120000-question"
        brief = "Résumé."

    class _Q:
        def __init__(self, *a, **k):
            pass

        def query(self, q, top_k=5):
            return _R()

    monkeypatch.setattr(wiki, "WikiQuery", _Q)
    out = wiki.op_query("question ?")
    assert out["synthesis"] == "Synthèse."
    assert out["references"] == ["src-a"]
    assert out["brief"] == "Résumé."
    assert out["obsidian_uri"].startswith("obsidian://open?vault=")
    assert "&file=Wiki_LM%2Fhistorique%2F20260908-120000-question" in out["obsidian_uri"]
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `~/Secretarius/Wiki_LM`): `.venv/bin/pytest tests/test_wiki_cli.py::test_query_returns_synthesis -v`
Expected: FAIL — `KeyError: 'brief'` (or the `assert out == {...}` form failing since the dict doesn't have those keys yet).

- [ ] **Step 3: Add the `quote` import**

In `Wiki_LM/tools/wiki.py`, replace:

```python
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
```

with:

```python
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
```

- [ ] **Step 4: Add `_build_obsidian_uri` and update `op_query`**

In `Wiki_LM/tools/wiki.py`, replace:

```python
def op_query(question: str) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question)
        if not result.text:
            return {"error": "KB vide — lancer ingest d'abord"}
        return {"synthesis": result.text, "references": result.references}
    except Exception as exc:
        return {"error": str(exc)}
```

with:

```python
def _build_obsidian_uri(history_slug: str) -> str:
    vault_root = _wiki_root().parent
    rel_path = f"Wiki_LM/historique/{history_slug}"
    return f"obsidian://open?vault={quote(vault_root.name, safe='')}&file={quote(rel_path, safe='')}"


def op_query(question: str) -> dict:
    try:
        result = WikiQuery(_wiki_root()).query(question)
        if not result.text:
            return {"error": "KB vide — lancer ingest d'abord"}
        return {
            "synthesis": result.text,
            "references": result.references,
            "brief": result.brief,
            "obsidian_uri": _build_obsidian_uri(result.history_slug),
        }
    except Exception as exc:
        return {"error": str(exc)}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_wiki_cli.py::test_query_returns_synthesis -v`
Expected: PASS.

- [ ] **Step 6: Run the full Wiki_LM suite**

Run: `.venv/bin/pytest -q`
Expected: PASS, no regressions (in particular `test_query_empty_kb`, which doesn't touch the new fields, must still pass unchanged).

- [ ] **Step 7: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/wiki.py Wiki_LM/tests/test_wiki_cli.py
git commit -m "$(cat <<'EOF'
feat(wiki_lm): op_query expose brief + lien obsidian:// vers l'historique

wiki.py (façade CLI/sandbox utilisée par Telegram et WebChat) relaie
désormais le résumé court et un lien obsidian://open vers la note
d'historique déjà écrite par WikiQuery.query().

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 3: `server.py::handle_query` — expose `history_slug` and `brief`

**Files:**
- Modify: `Wiki_LM/tools/server.py:57-74` (`handle_query`)
- Test: `Wiki_LM/tests/test_server.py` (add a new `TestHandleQuery` class)

**Interfaces:**
- Consumes: `QueryResult.history_slug`, `QueryResult.brief` (Task 1), via `_wq.query(...)`.
- Produces: `POST /query` response JSON gains `history_slug` and `brief` alongside the existing `text`/`references`/`saved_slug` — consumed by Task 6 (Obsidian templates read `data.history_slug`).

- [ ] **Step 1: Write the failing test**

In `Wiki_LM/tests/test_server.py`, update the module docstring (line 1) from:

```python
"""Tests de l'endpoint /capture de server.py."""
```

to:

```python
"""Tests des endpoints /capture et /query de server.py."""
```

Then append at the end of the file:

```python
class TestHandleQuery:
    def test_returns_history_slug_and_brief(self, client, monkeypatch):
        class _Result:
            text = "Synthèse."
            references = ["c-test"]
            saved_slug = ""
            history_slug = "20260908-120000-question-test"
            brief = "Résumé bref."

        class _Q:
            mode = "hybrid"

            def query(self, question, top_k=5, save=False):
                return _Result()

        import server
        monkeypatch.setattr(server, "_wq", _Q())

        response = client.post("/query", json={"question": "Question ?"})

        assert response.status_code == 200
        data = response.get_json()
        assert data["text"] == "Synthèse."
        assert data["references"] == ["c-test"]
        assert data["history_slug"] == "20260908-120000-question-test"
        assert data["brief"] == "Résumé bref."
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `~/Secretarius/Wiki_LM`): `.venv/bin/pytest tests/test_server.py::TestHandleQuery -v`
Expected: FAIL — `KeyError: 'history_slug'`.

- [ ] **Step 3: Update `handle_query`**

In `Wiki_LM/tools/server.py`, replace:

```python
    result = _wq.query(question, top_k=top_k, save=save)
    return jsonify({
        "text": result.text,
        "references": result.references,
        "saved_slug": result.saved_slug,
    })
```

with:

```python
    result = _wq.query(question, top_k=top_k, save=save)
    return jsonify({
        "text": result.text,
        "references": result.references,
        "saved_slug": result.saved_slug,
        "history_slug": result.history_slug,
        "brief": result.brief,
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_server.py -v`
Expected: PASS (all `TestHandleCapture` tests plus the new `TestHandleQuery` test).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/server.py Wiki_LM/tests/test_server.py
git commit -m "$(cat <<'EOF'
feat(wiki_lm): /query (server.py) expose history_slug + brief

Nécessaire pour que les templates Obsidian ouvrent la note d'historique
au lieu de reconstruire son contenu côté client.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 4: `derisk-deleg/src/wiki-ops.ts` — `regime` parameter

**Files:**
- Modify: `derisk-deleg/src/wiki-ops.ts:66-97` (`runWikiOp`, `formatWikiResult`)
- Test: `derisk-deleg/src/wiki-ops.test.ts` (modify 2 existing tests, add 7 new)

**Interfaces:**
- Consumes: `wiki.py::op_query`'s JSON shape from Task 2 (`json.brief`, `json.obsidian_uri`, `json.synthesis`).
- Produces: `export type WikiOpRegime = "brief" | "full"`; `formatWikiResult(op: string, json: any, regime: WikiOpRegime = "brief"): string`; `runWikiOp(api: any, op: string, arg: string, exec: Exec = execWikiSandbox, regime: WikiOpRegime = "brief"): Promise<string>` — consumed by Task 5 (`index.ts`'s wiki dispatch branch passes an explicit `regime`).

- [ ] **Step 1: Update the two existing "query" tests that assumed full-synthesis-always**

In `derisk-deleg/src/wiki-ops.test.ts`, replace:

```typescript
  it("query : renvoie la synthèse verbatim", () => {
    expect(formatWikiResult("query", { synthesis: "# GPU TEE\n…", references: ["c-x"] }))
      .toBe("# GPU TEE\n…");
  });
```

with:

```typescript
  it("query (regime full) : renvoie la synthèse verbatim", () => {
    expect(formatWikiResult("query", { synthesis: "# GPU TEE\n…", references: ["c-x"] }, "full"))
      .toBe("# GPU TEE\n…");
  });
```

And replace:

```typescript
  it("erreur vide → ne renvoie pas un message vide (retombe sur l'op)", () => {
    expect(formatWikiResult("query", { error: "", synthesis: "# X" })).toBe("# X");
  });
```

with:

```typescript
  it("erreur vide → ne renvoie pas un message vide (retombe sur l'op)", () => {
    expect(formatWikiResult("query", { error: "", synthesis: "# X" }, "full")).toBe("# X");
  });
```

(The `query : erreur surfacée verbatim` test and the `erreur générique inconnue` test are untouched — the first short-circuits before the `op` switch regardless of regime, the second returns the same default message in both regimes since neither `synthesis` nor `brief`/`obsidian_uri` are present.)

- [ ] **Step 2: Add new tests for the brief regime and the full-vs-brief distinction**

In `derisk-deleg/src/wiki-ops.test.ts`, immediately after the `erreur générique inconnue → message par défaut` test (still inside `describe("formatWikiResult", ...)`), add:

```typescript
  it("query (regime brief, défaut) : résumé + lien", () => {
    expect(formatWikiResult("query", {
      synthesis: "# X", brief: "Résumé.", obsidian_uri: "obsidian://open?vault=V&file=F",
    })).toBe("Résumé.\n\nobsidian://open?vault=V&file=F");
  });
  it("query (regime brief) : brief vide → lien seul", () => {
    expect(formatWikiResult("query", { brief: "", obsidian_uri: "obsidian://open?vault=V&file=F" }))
      .toBe("obsidian://open?vault=V&file=F");
  });
  it("query (regime brief) : lien absent → brief seul", () => {
    expect(formatWikiResult("query", { brief: "Résumé.", obsidian_uri: "" })).toBe("Résumé.");
  });
  it("query (regime brief) : les deux absents → message par défaut", () => {
    expect(formatWikiResult("query", { synthesis: "# X" })).toBe("Réponse wiki vide ou inattendue.");
  });
  it("query (regime full) : ignore brief/obsidian_uri même présents", () => {
    expect(formatWikiResult("query",
      { synthesis: "# X", brief: "Résumé.", obsidian_uri: "obsidian://open?vault=V&file=F" }, "full"))
      .toBe("# X");
  });
```

Then, inside `describe("runWikiOp", ...)`, after the existing `it("ignore les lignes de diagnostic avant le JSON...")` test, add:

```typescript
  it("passe le régime à formatWikiResult (full → synthèse verbatim)", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "brief": "Résumé.", "obsidian_uri": "obsidian://x"}'),
      "full");
    expect(out).toBe("# GPU TEE");
  });
  it("régime par défaut (brief) : résumé + lien, pas la synthèse complète", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "brief": "Résumé.", "obsidian_uri": "obsidian://x"}'));
    expect(out).toBe("Résumé.\n\nobsidian://x");
  });
```

- [ ] **Step 3: Run tests to verify they fail**

Run (from `~/Secretarius/derisk-deleg`): `npm test -- wiki-ops.test.ts`
Expected: FAIL — the two modified tests fail because `formatWikiResult` doesn't accept a third argument yet and defaults to full-synthesis behavior for everything; the new brief-regime tests fail because `formatWikiResult`/`runWikiOp` don't read `json.brief`/`json.obsidian_uri` yet.

- [ ] **Step 4: Add the `regime` parameter**

In `derisk-deleg/src/wiki-ops.ts`, replace:

```typescript
type Exec = (api: any, argv: string[]) => Promise<{ code: number; stdout: string; stderr: string }>;

// Compose execWikiSandbox : construit argv, exécute, parse JSON, formate ou renvoie erreur.
export async function runWikiOp(
  api: any, op: string, arg: string, exec: Exec = execWikiSandbox,
): Promise<string> {
  const argv = ["python3", "/wiki-tools/wiki.py", op];
  if (arg) argv.push(arg);
  const { code, stdout, stderr } = await exec(api, argv);
  if (code !== 0) return `Erreur wiki : ${(stderr || stdout || "échec").slice(0, 500)}`;
  // wiki.py imprime parfois des lignes de diagnostic avant le JSON (ex. query :
  // « [query] Embeddings absents… ») ; le résultat JSON est toujours la DERNIÈRE
  // ligne non vide (un seul print(json.dumps(...)) final, sur une ligne).
  const lines = stdout.split("\n").map((l) => l.trim()).filter(Boolean);
  const lastLine = lines.length ? lines[lines.length - 1] : "";
  let json: any;
  try {
    json = JSON.parse(lastLine);
  } catch {
    return `Erreur wiki : sortie inattendue (${stdout.slice(0, 200)})`;
  }
  return formatWikiResult(op, json);
}

// Formatage déterministe du JSON de wiki.py en message utilisateur.
// Aucune invention : sur erreur, on surface le texte de wiki.py verbatim.
export function formatWikiResult(op: string, json: any): string {
  if (json && typeof json.error === "string" && json.error.trim()) return json.error;
  if (json && json.status === "error") return json.reason ?? json.error ?? "Erreur wiki.";

  switch (op) {
    case "query":
      return typeof json?.synthesis === "string" && json.synthesis.trim()
        ? json.synthesis
        : "Réponse wiki vide ou inattendue.";
```

with:

```typescript
type Exec = (api: any, argv: string[]) => Promise<{ code: number; stdout: string; stderr: string }>;

// "full" = synthèse complète (Obsidian, WebChat — rendent le Markdown
// correctement). "brief" = résumé court + lien vers l'historique (Telegram,
// et tout canal non identifiable). Voir la spec pour le détail par canal.
export type WikiOpRegime = "brief" | "full";

// Compose execWikiSandbox : construit argv, exécute, parse JSON, formate ou renvoie erreur.
export async function runWikiOp(
  api: any, op: string, arg: string, exec: Exec = execWikiSandbox,
  regime: WikiOpRegime = "brief",
): Promise<string> {
  const argv = ["python3", "/wiki-tools/wiki.py", op];
  if (arg) argv.push(arg);
  const { code, stdout, stderr } = await exec(api, argv);
  if (code !== 0) return `Erreur wiki : ${(stderr || stdout || "échec").slice(0, 500)}`;
  // wiki.py imprime parfois des lignes de diagnostic avant le JSON (ex. query :
  // « [query] Embeddings absents… ») ; le résultat JSON est toujours la DERNIÈRE
  // ligne non vide (un seul print(json.dumps(...)) final, sur une ligne).
  const lines = stdout.split("\n").map((l) => l.trim()).filter(Boolean);
  const lastLine = lines.length ? lines[lines.length - 1] : "";
  let json: any;
  try {
    json = JSON.parse(lastLine);
  } catch {
    return `Erreur wiki : sortie inattendue (${stdout.slice(0, 200)})`;
  }
  return formatWikiResult(op, json, regime);
}

// Formatage déterministe du JSON de wiki.py en message utilisateur.
// Aucune invention : sur erreur, on surface le texte de wiki.py verbatim.
export function formatWikiResult(op: string, json: any, regime: WikiOpRegime = "brief"): string {
  if (json && typeof json.error === "string" && json.error.trim()) return json.error;
  if (json && json.status === "error") return json.reason ?? json.error ?? "Erreur wiki.";

  switch (op) {
    case "query": {
      if (regime === "full") {
        return typeof json?.synthesis === "string" && json.synthesis.trim()
          ? json.synthesis
          : "Réponse wiki vide ou inattendue.";
      }
      const brief = typeof json?.brief === "string" ? json.brief.trim() : "";
      const uri = typeof json?.obsidian_uri === "string" ? json.obsidian_uri : "";
      if (!brief && !uri) return "Réponse wiki vide ou inattendue.";
      return [brief, uri].filter(Boolean).join("\n\n");
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test -- wiki-ops.test.ts`
Expected: PASS (all tests in the file, including the 7 new ones).

- [ ] **Step 6: Rebuild the plugin dist**

Run: `npm run build`
Expected: succeeds, `dist/wiki-ops.js` regenerated.

- [ ] **Step 7: Commit**

```bash
cd ~/Secretarius
git add derisk-deleg/src/wiki-ops.ts derisk-deleg/src/wiki-ops.test.ts derisk-deleg/dist/
git commit -m "$(cat <<'EOF'
feat(derisk-deleg): formatWikiResult/runWikiOp gagnent un régime brief/full

"full" = synthèse complète (Obsidian, WebChat). "brief" (défaut) = résumé
+ lien vers l'historique (Telegram, canal non identifiable). Remplace le
comportement précédent qui envoyait la synthèse complète à tous les canaux
sans distinction — c'était la source du dump verbeux sur Telegram.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 5: `derisk-deleg/src/index.ts` — pick the regime from `ctx.messageProvider`

**Files:**
- Modify: `derisk-deleg/src/index.ts:418` (hook signature), `:505-522` (wiki dispatch branch)
- Test: `derisk-deleg/src/index.test.ts:69-72` (global `afterEach`, add one line), append new `describe` block at end of file (after line 680)

**Interfaces:**
- Consumes: `runWikiOp(api, op, arg, exec, regime)` (Task 4).
- Produces: no new exported interface — this task wires an existing hook parameter (`ctx.messageProvider`, already part of the SDK's `PluginHookAgentContext` type) into the regime choice at the one call site that handles real `/q` traffic (typed or natural-language-routed, on any channel).

- [ ] **Step 1: Write the failing tests**

In `derisk-deleg/src/index.test.ts`, append at the end of the file (after the final `});` that closes the `before_tool_call` describe block):

```typescript
describe("before_agent_reply — régime de réponse par canal (query)", () => {
  function stubRouterQuery() {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/q", args: "question test" }),
      })),
    );
  }

  it("WebChat (ctx.messageProvider === 'webchat') → runWikiOp reçoit regime 'full'", async () => {
    stubRouterQuery();
    const runWikiOpSpy = vi.fn(async () => "réponse simulée");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "question test" }, { messageProvider: "webchat" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "query", "question test", undefined, "full");
  });

  it("Telegram (ctx.messageProvider === 'telegram') → runWikiOp reçoit regime 'brief'", async () => {
    stubRouterQuery();
    const runWikiOpSpy = vi.fn(async () => "réponse simulée");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "question test" }, { messageProvider: "telegram" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "query", "question test", undefined, "brief");
  });

  it("ctx absent (canal inconnu) → runWikiOp reçoit regime 'brief'", async () => {
    stubRouterQuery();
    const runWikiOpSpy = vi.fn(async () => "réponse simulée");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "question test" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "query", "question test", undefined, "brief");
  });
});
```

Then, in the same file, update the global `afterEach` (currently):

```typescript
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
```

to:

```typescript
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  vi.doUnmock("./wiki-ops.js");
});
```

(This undoes the per-test `vi.doMock("./wiki-ops.js", ...)` above so it never leaks into other tests in the file — safe to call even when nothing was mocked.)

- [ ] **Step 2: Run tests to verify they fail**

Run (from `~/Secretarius/derisk-deleg`): `npm test -- index.test.ts -t "régime de réponse par canal"`
Expected: FAIL — `runWikiOpSpy` is called with only `(api, "query", "question test")` (no `exec`/`regime` args), so `toHaveBeenCalledWith(..., undefined, "full")` / `"brief"` fails.

- [ ] **Step 3: Capture `ctx` in the hook and compute the regime**

In `derisk-deleg/src/index.ts`, replace:

```typescript
    api.on("before_agent_reply", async (event: any) => {
```

with:

```typescript
    api.on("before_agent_reply", async (event: any, ctx: any) => {
```

Then replace:

```typescript
      if (action.kind === "wiki") {
        // Écriture inférée en langage naturel : jamais exécutée directement,
        // même logique de mise en attente que gog_send/gog_reply.
        if (!WIKI_READ_OPS.has(action.op)) {
          pending = { kind: "router-write", op: action.op, args: routed.args, ts: Date.now() };
          return {
            handled: true,
            reply: {
              text: `Action wiki en attente : ${action.op}${routed.args ? " | " + routed.args : ""}\n\nTapez /confirm pour l'exécuter (valable 10 min), ou /annuler pour abandonner.`,
            },
          };
        }
        const out = await runWikiOp(api, action.op, routed.args);
        return { handled: true, reply: { text: out.slice(0, 4000) } };
      }
```

with:

```typescript
      if (action.kind === "wiki") {
        // Écriture inférée en langage naturel : jamais exécutée directement,
        // même logique de mise en attente que gog_send/gog_reply.
        if (!WIKI_READ_OPS.has(action.op)) {
          pending = { kind: "router-write", op: action.op, args: routed.args, ts: Date.now() };
          return {
            handled: true,
            reply: {
              text: `Action wiki en attente : ${action.op}${routed.args ? " | " + routed.args : ""}\n\nTapez /confirm pour l'exécuter (valable 10 min), ou /annuler pour abandonner.`,
            },
          };
        }
        // WebChat rend le Markdown correctement (contrairement à Telegram) :
        // synthèse complète conservée pour ce canal. Bref+lien partout
        // ailleurs, y compris si le canal n'est pas identifiable (défaut sûr).
        const regime: "brief" | "full" = ctx?.messageProvider === "webchat" ? "full" : "brief";
        const out = await runWikiOp(api, action.op, routed.args, undefined, regime);
        return { handled: true, reply: { text: out.slice(0, 4000) } };
      }
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `npm test -- index.test.ts -t "régime de réponse par canal"`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full derisk-deleg suite**

Run: `npm test`
Expected: PASS, no regressions (in particular the pre-existing "une lecture (/q → query) s'exécute immédiatement" test in the `confirmation des écritures wiki routées` describe block, which calls the hook with only one argument — `ctx` will be `undefined`, `ctx?.messageProvider` evaluates to `undefined`, regime falls back to `"brief"`, behavior unchanged for that test's loose assertion).

- [ ] **Step 6: Rebuild the plugin dist**

Run: `npm run build`
Expected: succeeds, `dist/index.js` regenerated.

- [ ] **Step 7: Commit**

```bash
cd ~/Secretarius
git add derisk-deleg/src/index.ts derisk-deleg/src/index.test.ts derisk-deleg/dist/
git commit -m "$(cat <<'EOF'
feat(derisk-deleg): choisit le régime /q selon ctx.messageProvider

before_agent_reply capture désormais son second paramètre (ctx), jusqu'ici
ignoré, pour lire ctx.messageProvider : "webchat" → régime complet,
tout le reste (Telegram, canal non identifiable) → bref+lien (défaut sûr).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 6: Obsidian templates — open the history note instead of inserting text

**Files:**
- Modify: `Wiki_LM/obsidian_template_wikilm.md` (full rewrite of the tail), `Wiki_LM/obsidian_template_wikilm_android.md` (full rewrite of the tail)

**Interfaces:**
- Consumes: `POST /query` response field `data.history_slug` (Task 3).
- Produces: no code interface — this is the terminal consumer.

No automated test exists for these files (Obsidian Templater JS, not runnable outside the Obsidian app in this repo) — verify manually per Step 3.

- [ ] **Step 1: Update `Wiki_LM/obsidian_template_wikilm.md`**

Replace the entire file content with:

```
<%*
// Template Templater — Interroger Wiki_LM
// Placer dans le dossier Templates configuré dans Templater > Template folder location
// Appeler via : Templater > Open Insert Template modal (PAS "Create new note
// from template" — ce template n'insère plus rien dans la note en cours ;
// utiliser "Create new note" laisserait une note vide derrière lui).
//
// N'insère plus la réponse dans la note en cours : le serveur écrit lui-même
// un enregistrement horodaté dans Wiki_LM/historique/, ce template se
// contente de l'ouvrir dans un nouvel onglet.

const WIKI_SERVER = "http://127.0.0.1:5051";

const question = await tp.system.prompt("Question pour Wiki_LM");
if (!question) { return; }

let data;
try {
    const resp = await fetch(`${WIKI_SERVER}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, top_k: 5 })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
} catch (e) {
    new Notice(`Wiki_LM : erreur — ${e.message}`, 8000);
    return;
}

const historyPath = `Wiki_LM/historique/${data.history_slug}.md`;
const file = app.vault.getAbstractFileByPath(historyPath);
if (file) {
    await app.workspace.getLeaf(true).openFile(file);
} else {
    new Notice(`Wiki_LM : note d'historique introuvable (${historyPath})`, 8000);
}
%>
```

- [ ] **Step 2: Update `Wiki_LM/obsidian_template_wikilm_android.md`**

Replace the entire file content with:

```
<%*
// Template Templater — Interroger Wiki_LM depuis Obsidian desktop/Android (via Tailscale)
// Nécessite : server.py lancé sur sanroque
// requestUrl contourne le CSP d'Electron, contrairement à fetch()
//
// N'insère plus la réponse dans la note en cours : le serveur écrit lui-même
// un enregistrement horodaté dans Wiki_LM/historique/, ce template se
// contente de l'ouvrir dans un nouvel onglet. Appeler via Templater > Open
// Insert Template modal (pas "Create new note from template").

const WIKI_SERVER = "http://sanroque:5051";

const mode = await tp.system.suggester(
    ["Hybride (BM25 + sémantique)", "Sémantique", "BM25"],
    ["hybrid", "semantic", "bm25"],
    false,
    "Mode de recherche"
) || "hybrid";

const question = await tp.system.prompt("Question pour Wiki_LM");
if (!question) { return; }

// requestUrl (API Obsidian) contourne le CSP d'Electron, contrairement à fetch().
const { requestUrl } = tp.obsidian ?? require("obsidian");

let data;
try {
    const resp = await requestUrl({
        url: `${WIKI_SERVER}/query`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ question: question, top_k: 5, mode: mode }),
        throw: false,
    });
    if (resp.status !== 200) throw new Error(`HTTP ${resp.status}`);
    data = resp.json;
} catch (e) {
    new Notice(`Wiki_LM : erreur — ${e.message}`, 8000);
    return;
}

const historyPath = `Wiki_LM/historique/${data.history_slug}.md`;
const file = app.vault.getAbstractFileByPath(historyPath);
if (file) {
    await app.workspace.getLeaf(true).openFile(file);
} else {
    new Notice(`Wiki_LM : note d'historique introuvable (${historyPath})`, 8000);
}
%>
```

- [ ] **Step 3: Manual verification (requires the Obsidian app — not automatable)**

1. Copy the updated `Wiki_LM/obsidian_template_wikilm_android.md` into the vault's configured Templates folder (per `docs/components/obsidian.md`, e.g. `Templates/Wiki_LM Query.md`), overwriting the previous copy.
2. Confirm `wiki-lm-server.service` is running: `systemctl --user status wiki-lm-server`.
3. In Obsidian, open any note, run Ctrl/Cmd-P → "Templater: Open Insert Template modal" → select the Wiki_LM template → answer the mode/question prompts.
4. Expected: a new tab opens showing the full answer (a note under `Wiki_LM/historique/`); the note you started from is completely unchanged (nothing inserted).
5. If a `Wiki_LM : note d'historique introuvable` notice appears instead, check the vault has finished syncing the newly-created `historique/` file (Obsidian Sync latency) before retrying.

- [ ] **Step 4: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/obsidian_template_wikilm.md Wiki_LM/obsidian_template_wikilm_android.md
git commit -m "$(cat <<'EOF'
feat(wiki_lm): les templates Obsidian ouvrent l'historique au lieu d'insérer

/query écrit désormais lui-même la note complète dans historique/ ; les
templates n'ont plus qu'à l'ouvrir dans un nouvel onglet, au lieu de
reconstruire un bloc de texte et l'insérer dans la note en cours de lecture.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/components/wiki-lm.md` (query.py description, `/query` endpoint signature)
- Modify: `docs/components/obsidian.md` (template behavior description, usage step 4)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `docs/components/wiki-lm.md` — describe the history mechanism**

Replace:

```
**`query.py`** — Interrogation en langage naturel (BM25 + LLM)

```bash
python tools/query.py "Comment fonctionne le Memex ?" --top 5
python tools/query.py "Karpathy et les wikis" --top 5 --save
```
```

with:

```
**`query.py`** — Interrogation en langage naturel (BM25 + LLM)

```bash
python tools/query.py "Comment fonctionne le Memex ?" --top 5
python tools/query.py "Karpathy et les wikis" --top 5 --save
```

Chaque requête écrit automatiquement un enregistrement horodaté dans
`historique/` (hors de `wiki/`, non indexé, jamais vu par la recherche ni
l'ingestion) : `<horodatage>-<slug question>.md`, avec la synthèse complète.
Distinct du flag `--save`, qui écrit en plus une page polie et **indexée**
dans `wiki/` (`synth-<slug>.md`).
```

- [ ] **Step 2: Update `docs/components/wiki-lm.md` — endpoint signature**

Replace:

```
Endpoints : `POST /query` `{question, top_k, mode}` → `{text, references, saved_slug}` ;
```

with:

```
Endpoints : `POST /query` `{question, top_k, mode}` → `{text, references, saved_slug, history_slug, brief}` ;
```

- [ ] **Step 3: Update `docs/components/obsidian.md` — template behavior description**

Replace:

```
Interroger le wiki en langage naturel **directement depuis Obsidian** (desktop ou
Android) : la synthèse et les liens `[[source]]` sont insérés dans la note courante.
Le template appelle le serveur `wiki-lm-server` (port 5051, voir
`docs/components/wiki-lm.md`). Fichier source : `Wiki_LM/obsidian_template_wikilm_android.md`.
```

with:

```
Interroger le wiki en langage naturel **directement depuis Obsidian** (desktop ou
Android) : la réponse s'ouvre dans un **nouvel onglet** (note d'historique
horodatée), jamais insérée dans la note en cours. Le template appelle le
serveur `wiki-lm-server` (port 5051, voir `docs/components/wiki-lm.md`), qui
écrit lui-même cette note dans `Wiki_LM/historique/`. Fichier source :
`Wiki_LM/obsidian_template_wikilm_android.md`.
```

- [ ] **Step 4: Update `docs/components/obsidian.md` — usage step 4**

Replace:

```
4. Choisir le **mode** (Hybride recommandé / Sémantique / BM25).
5. Saisir la **question** → la synthèse + les sources s'insèrent au curseur.
```

with:

```
4. Choisir le **mode** (Hybride recommandé / Sémantique / BM25).
5. Saisir la **question** → la réponse s'ouvre dans un nouvel onglet (rien
   n'est inséré dans la note en cours).
```

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add docs/components/wiki-lm.md docs/components/obsidian.md
git commit -m "$(cat <<'EOF'
docs(wiki_lm): documente historique/, le régime par canal et l'ouverture en onglet

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Kuc14MHhgPjgWCgWxFF6Ps
EOF
)"
```

---

## Deployment note (not part of this plan's tasks — flag for the user before doing it)

This plan only changes the repo. Getting it live on sanroque additionally requires, in this order: (1) `openclaw plugins build` + reinstalling the `derisk-deleg` plugin so the gateway picks up the new `dist/`, (2) restarting `slm-llama_cpp`/`tiron-router`/`openclaw-gateway` as relevant, (3) restarting `wiki-lm-server.service` so `server.py` picks up the new `/query` response shape, (4) copying the updated Templater file into the live vault's `Templates/` folder (Task 6, Step 3). Per the project's existing rule, `systemctl restart`/`docker compose` on the live host need explicit confirmation before running — do not do this automatically at the end of Task 7.
