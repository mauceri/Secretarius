---
name: q
description: "Interroger la base de connaissances wiki. Dispatch déterministe vers l'outil wiki_query (délègue op: query à l'agent wiki)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_query
command-arg-mode: raw
---

`/q <question>` interroge le wiki, de façon déterministe : l'outil `wiki_query`
délègue `op: query | <question>` à l'agent wiki et relaie la synthèse.
