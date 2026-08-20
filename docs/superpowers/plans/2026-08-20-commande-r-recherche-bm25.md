# Commande `/r` — recherche brute BM25 sans synthèse LLM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une commande Telegram `/r <mots-clés>` qui retourne les titres et extraits des pages wiki les plus pertinentes (BM25), sans aucun appel LLM — plus rapide que `/q`.

**Architecture:** Extension mécanique du patron déjà en place pour les 5 commandes déterministes existantes : `Wiki_LM/tools/wiki.py` (nouvel op `search`, réutilise `WikiSearch` existant) → outil `derisk-deleg` (`wiki_search`, exécution directe sandboxée, jamais de sous-agent LLM) → nouveau skill `/r`.

**Tech Stack:** Python 3 / pytest côté `wiki.py` (`~/Secretarius/Wiki_LM/.venv/bin/pytest`) ; TypeScript / vitest côté `derisk-deleg` (`~/Secretarius/derisk-deleg`, `npm test`).

## Global Constraints

- Zéro appel LLM à aucune étape de `/r` — uniquement `WikiSearch.search()` (BM25 pur, déjà existant dans `search.py`, ne pas le modifier).
- `op_search` retourne `{"results": [{"title": ..., "excerpt": ...}, ...]}` — `slug`/`score`/`path` du `SearchResult` non exposés.
- `top_k` par défaut : 5 (même défaut que `WikiSearch.search()`).
- Format du message formaté : liste numérotée `N. <titre>\n   <extrait>`, résultats séparés par une ligne vide ; `"Aucun résultat."` si la liste est vide.
- `/r` sans argument → message d'usage direct côté outil TypeScript (`"Usage: /r <mots-clés>"`), sans appeler `wiki.py` (même patron que `source_read` pour `/source`).
- Commandes de test : `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_wiki_cli.py -v` (baseline 18 tests) ; `cd ~/Secretarius/derisk-deleg && npm test` (baseline 22 tests, dont 14 dans `wiki-ops.test.ts`).

---

### Task 1: `wiki.py` — nouvel op `search`

**Files:**
- Modify: `Wiki_LM/tools/wiki.py:32` (import), `Wiki_LM/tools/wiki.py:97-99` (nouvelle fonction), `Wiki_LM/tools/wiki.py:273-274` (dispatcher)
- Test: `Wiki_LM/tests/test_wiki_cli.py`

**Interfaces:**
- Produces: `op_search(question: str) -> dict` — `{"results": [{"title": str, "excerpt": str}, ...]}`. Consommée par le dispatcher `main()` de ce même fichier ; aucune autre tâche n'en dépend directement (le côté TypeScript consomme la sortie JSON via l'exécution sandboxée, pas un import Python).

- [ ] **Step 1: Write the failing tests**

Ajouter dans `Wiki_LM/tests/test_wiki_cli.py`, juste après `test_query_empty_kb` (dont la dernière ligne est `assert "error" in wiki.op_query("q")`) :

```python
def test_search_returns_results(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _R:
        def __init__(self, title, excerpt):
            self.title = title
            self.excerpt = excerpt

    class _S:
        def __init__(self, *a, **k):
            pass

        def search(self, q, top_k=5):
            return [_R("Titre A", "extrait A"), _R("Titre B", "extrait B")]

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.op_search("mots-clés")
    assert out == {"results": [
        {"title": "Titre A", "excerpt": "extrait A"},
        {"title": "Titre B", "excerpt": "extrait B"},
    ]}


def test_search_no_results(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _S:
        def __init__(self, *a, **k):
            pass

        def search(self, q, top_k=5):
            return []

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.op_search("mots-clés")
    assert out == {"results": []}


def test_main_search_dispatch(monkeypatch, tmp_path):
    wiki = _wiki(monkeypatch, tmp_path)

    class _S:
        def __init__(self, *a, **k):
            pass

        def search(self, q, top_k=5):
            return []

    monkeypatch.setattr(wiki, "WikiSearch", _S)
    out = wiki.main(["search", "mots-clés"])
    assert "results" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_wiki_cli.py::test_search_returns_results -v`
Expected: FAIL — `AttributeError: module 'wiki' has no attribute 'WikiSearch'` (ou équivalent : `op_search` non défini)

- [ ] **Step 3: Write minimal implementation**

Dans `Wiki_LM/tools/wiki.py`, modifier la ligne 32 :

```python
from query import WikiQuery
```

par :

```python
from query import WikiQuery
from search import WikiSearch
```

Ajouter cette fonction juste après `op_query` (après la ligne 97 `return {"error": str(exc)}`, avant les deux lignes vides précédant `def _state_path`) :

```python
def op_search(question: str) -> dict:
    results = WikiSearch(_wiki_root()).search(question, top_k=5)
    return {"results": [{"title": r.title, "excerpt": r.excerpt} for r in results]}
```

Dans `main()`, modifier les lignes 273-274 :

```python
    if op == "query":
        return op_query(arg)
```

par :

```python
    if op == "query":
        return op_query(arg)
    if op == "search":
        return op_search(arg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest tests/test_wiki_cli.py -v`
Expected: PASS (21 tests)

Puis la suite complète :

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest -q`
Expected: PASS, 0 failure

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/tools/wiki.py Wiki_LM/tests/test_wiki_cli.py
git commit -m "feat: op search dans wiki.py (recherche BM25 brute, sans LLM)"
```

---

### Task 2: `derisk-deleg` — formatage du résultat `search`

**Files:**
- Modify: `derisk-deleg/src/wiki-ops.ts:121-122` (nouveau cas dans `formatWikiResult`)
- Test: `derisk-deleg/src/wiki-ops.test.ts`

**Interfaces:**
- Consumes: la forme JSON produite par `op_search` (Task 1) : `{"results": [{"title": str, "excerpt": str}, ...]}`.
- Produces: `formatWikiResult("search", json) -> string`. Consommée par `runWikiOp` (existant, inchangé) et par le nouvel outil `wiki_search` (Task 3).

- [ ] **Step 1: Write the failing tests**

Ajouter dans `derisk-deleg/src/wiki-ops.test.ts`, juste après le test `"tags : joint la liste"` :

```typescript
  it("search : liste numérotée titre + extrait", () => {
    expect(formatWikiResult("search", { results: [
      { title: "Titre A", excerpt: "extrait A" },
      { title: "Titre B", excerpt: "extrait B" },
    ] })).toBe("1. Titre A\n   extrait A\n\n2. Titre B\n   extrait B");
  });

  it("search : liste vide", () => {
    expect(formatWikiResult("search", { results: [] })).toBe("Aucun résultat.");
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Secretarius/derisk-deleg && npx vitest run src/wiki-ops.test.ts -t "search"`
Expected: FAIL — sortie `"Réponse wiki vide ou inattendue."` (cas `default` du switch) au lieu du texte attendu

- [ ] **Step 3: Write minimal implementation**

Dans `derisk-deleg/src/wiki-ops.ts`, remplacer le bloc (lignes 118-121) :

```typescript
    case "tags": {
      const tags = Array.isArray(json?.tags) ? json.tags : [];
      return tags.length ? `Tags : ${tags.join(", ")}.` : "Aucun tag.";
    }
```

par :

```typescript
    case "tags": {
      const tags = Array.isArray(json?.tags) ? json.tags : [];
      return tags.length ? `Tags : ${tags.join(", ")}.` : "Aucun tag.";
    }
    case "search": {
      const results = Array.isArray(json?.results) ? json.results : [];
      if (!results.length) return "Aucun résultat.";
      return results
        .map((r: any, i: number) => `${i + 1}. ${r.title}\n   ${r.excerpt}`)
        .join("\n\n");
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Secretarius/derisk-deleg && npm test`
Expected: PASS (24 tests, dont 16 dans `wiki-ops.test.ts`)

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add derisk-deleg/src/wiki-ops.ts derisk-deleg/src/wiki-ops.test.ts
git commit -m "feat: formatWikiResult formate les résultats de recherche BM25"
```

---

### Task 3: `derisk-deleg` — outil `wiki_search` + skill `/r`

**Files:**
- Modify: `derisk-deleg/src/index.ts:191-193` (nouvel outil, entre `wiki_query` et `wiki_tags`)
- Create: `openclaw-config/workspace/skills/r/SKILL.md`

**Interfaces:**
- Consumes: `runWikiOp` (existant, inchangé), `formatWikiResult` (Task 2, via `runWikiOp`).

- [ ] **Step 1: Add the tool registration**

Dans `derisk-deleg/src/index.ts`, repérer ce bloc actuel :

```typescript
    api.registerTool({
      name: "wiki_query",
      description:
        "Ask the wiki knowledge base a question (delegates 'op: query' to the wiki agent).",
      parameters: Type.Object({
        command: Type.Optional(
          Type.String({ description: "Raw args: the natural-language question." }),
        ),
      }),
      async execute(_id: string, params: { command?: string }) {
        const arg = (params?.command ?? "").trim();
        const out = await runWikiOp(api, "query", arg);
        return { content: [{ type: "text", text: out.slice(0, 4000) }] };
      },
    });

    api.registerTool({
      name: "wiki_tags",
```

Insérer un nouveau `registerTool` entre les deux, pour obtenir :

```typescript
    api.registerTool({
      name: "wiki_query",
      description:
        "Ask the wiki knowledge base a question (delegates 'op: query' to the wiki agent).",
      parameters: Type.Object({
        command: Type.Optional(
          Type.String({ description: "Raw args: the natural-language question." }),
        ),
      }),
      async execute(_id: string, params: { command?: string }) {
        const arg = (params?.command ?? "").trim();
        const out = await runWikiOp(api, "query", arg);
        return { content: [{ type: "text", text: out.slice(0, 4000) }] };
      },
    });

    api.registerTool({
      name: "wiki_search",
      description:
        "Recherche brute BM25 dans le wiki, sans synthèse LLM (délègue 'op: search' à l'agent wiki).",
      parameters: Type.Object({
        command: Type.Optional(
          Type.String({ description: "Raw args: mots-clés de recherche." }),
        ),
      }),
      async execute(_id: string, params: { command?: string }) {
        const arg = (params?.command ?? "").trim();
        if (!arg) {
          return { content: [{ type: "text", text: "Usage: /r <mots-clés>" }] };
        }
        const out = await runWikiOp(api, "search", arg);
        return { content: [{ type: "text", text: out.slice(0, 4000) }] };
      },
    });

    api.registerTool({
      name: "wiki_tags",
```

- [ ] **Step 2: Verify the TypeScript build compiles**

Run: `cd ~/Secretarius/derisk-deleg && npm run build`
Expected: se termine sans erreur (`tsc -p tsconfig.json`)

Puis la suite vitest, pour confirmer qu'aucune régression n'a été introduite :

Run: `cd ~/Secretarius/derisk-deleg && npm test`
Expected: PASS (24 tests, inchangé depuis la Task 2 — aucun test automatisé n'existe pour l'enregistrement d'outils dans `index.ts`, non testable sans runtime openclaw ; la logique de formatage est déjà couverte par la Task 2)

- [ ] **Step 3: Create the skill file**

Créer `openclaw-config/workspace/skills/r/SKILL.md` :

```markdown
---
name: r
description: "Recherche brute BM25 dans le wiki, sans synthèse LLM. Dispatch déterministe vers l'outil wiki_search (délègue op: search à l'agent wiki)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_search
command-arg-mode: raw
---

`/r <mots-clés>` retourne les titres et extraits des pages les plus
pertinentes (BM25), sans appel LLM — plus rapide que `/q`, utile pour
localiser rapidement une page.
```

- [ ] **Step 4: Commit**

```bash
cd ~/Secretarius
git add derisk-deleg/src/index.ts openclaw-config/workspace/skills/r/SKILL.md
git commit -m "feat: outil wiki_search + skill /r (recherche BM25 sans LLM)"
```

---

### Task 4: Vérification finale

**Files:** aucun fichier modifié — vérification uniquement.

- [ ] **Step 1: Run both test suites**

Run: `cd ~/Secretarius/Wiki_LM && .venv/bin/pytest -q`
Expected: PASS, 0 failure (baseline 317 + 3 nouveaux = 320)

Run: `cd ~/Secretarius/derisk-deleg && npm test`
Expected: PASS, 24 tests (baseline 22 + 2 nouveaux)

Run: `cd ~/Secretarius/derisk-deleg && npm run build`
Expected: se termine sans erreur

- [ ] **Step 2: Manual verification against real wiki data (read-only, exécutable directement)**

Contrairement au plugin Obsidian, `op_search` est vérifiable directement en
CLI sans interface graphique ni Telegram — appel Python direct, aucune
écriture :

```bash
cd ~/Secretarius/Wiki_LM/tools
python3 wiki.py search "wiki_lm"
```

Expected : JSON `{"results": [...]}` avec des `title`/`excerpt` cohérents
(pages réelles du wiki mentionnant « wiki_lm »), retour quasi instantané
(pas d'appel réseau/LLM — comparer au temps de réponse de
`python3 wiki.py query "wiki_lm"`, nettement plus lent).

- [ ] **Step 3: Report**

Aucun commit pour cette tâche (vérification seule). Si l'étape 2 échoue ou
renvoie un JSON inattendu, revenir à la Task 1 avant de considérer le plan
terminé. Le test bout en bout via Telegram (`/r <mots-clés>`) reste à faire
par l'utilisateur — hors de portée de l'agent (pas d'accès à Telegram),
mais à ce stade la seule inconnue restante est le déploiement du plugin
`derisk-deleg` (build + redémarrage du service concerné), pas la logique
elle-même.
