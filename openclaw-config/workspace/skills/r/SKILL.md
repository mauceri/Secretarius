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
