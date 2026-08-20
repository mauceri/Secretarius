# Commande `/r` — recherche brute BM25 sans synthèse LLM — design

## Contexte

Idée notée au backlog (`project_ideas_backlog.md`, demandée 2026-06-24) : `/q`
passe par une synthèse LLM (Euria), ce qui est le vrai goulot d'attente
(BM25 lui-même est quasi instantané). `/r` doit retourner directement les
titres et extraits des pages les plus pertinentes, sans aucun appel LLM —
utile pour localiser rapidement une page depuis mobile.

Périmètre : une seule commande, extension mécanique du patron déjà en place
pour les 5 commandes déterministes existantes (`/c`, `/q`, `/tags`,
`/wikistatus`, `/ingest`, `/kbupdate`) — même architecture
`wiki.py <op>` → outil `derisk-deleg` → agent wiki en sandbox (jamais de
sous-agent LLM pour ces outils, exécution directe et déterministe).

## Conception

### `Wiki_LM/tools/wiki.py`

- Ajout de l'import `from search import WikiSearch` (module déjà existant,
  classe `WikiSearch.search(query, top_k) -> list[SearchResult]`, BM25 pur,
  aucun appel réseau ni LLM).
- Nouvelle fonction :

```python
def op_search(question: str) -> dict:
    results = WikiSearch(_wiki_root()).search(question, top_k=5)
    return {"results": [{"title": r.title, "excerpt": r.excerpt} for r in results]}
```

  `slug`, `score`, `path` du `SearchResult` ne sont pas exposés dans le JSON
  — inutiles pour le format Telegram retenu (titre + extrait).
- Câblage dans `main()` : `if op == "search": return op_search(arg)`.

### `derisk-deleg/src/wiki-ops.ts`

Nouveau cas dans `formatWikiResult(op, json)` :

```typescript
case "search": {
  const results = Array.isArray(json?.results) ? json.results : [];
  if (!results.length) return "Aucun résultat.";
  return results
    .map((r: any, i: number) => `${i + 1}. ${r.title}\n   ${r.excerpt}`)
    .join("\n\n");
}
```

### `derisk-deleg/src/index.ts`

Nouvel outil, sur le patron exact de `wiki_query`, avec un garde-fou sur
argument vide (même patron que `source_read` pour `/source`) :

```typescript
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
```

### Nouveau skill `openclaw-config/workspace/skills/r/SKILL.md`

Sur le patron exact de `q/SKILL.md` :

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

## Tests

- `Wiki_LM/tests/test_wiki_cli.py` : `op_search` (résultats formatés depuis
  un `WikiSearch` mocké, cas liste vide) ; dispatch `main(["search", "..."])`.
- `derisk-deleg/src/wiki-ops.test.ts` : `formatWikiResult("search", ...)`
  (résultats multiples, liste vide).

## Vérification

- Suite Python complète verte avant/après.
- Suite vitest `derisk-deleg` complète verte avant/après.
- Test manuel : `/r <mots-clés existants dans le wiki>` sur Telegram →
  réponse quasi instantanée (pas d'attente LLM), liste numérotée titre +
  extrait.

## Hors périmètre

- Aucune modification de `/q` ni du mode BM25 existant de `WikiQuery`.
- Aucun réentraînement du routeur d'intention — `/r` est une commande
  déterministe (`disable-model-invocation: true`), pas une intention à
  classer.
