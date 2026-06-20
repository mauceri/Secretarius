---
name: ingest
description: "Traiter la file de captures du wiki. Dispatch déterministe vers l'outil wiki_ingest (délègue op: ingest à l'agent wiki, traitement async)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_ingest
command-arg-mode: raw
---

`/ingest` lance le traitement de toute la file des captures en attente, de façon
déterministe : l'outil `wiki_ingest` délègue `op: ingest` à l'agent wiki (worker
asynchrone). N'accepte aucun argument et ne cible jamais une URL précise.
