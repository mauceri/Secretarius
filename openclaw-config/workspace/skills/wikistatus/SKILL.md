---
name: wikistatus
description: "État de l'ingestion du wiki. Dispatch déterministe vers l'outil wiki_status (délègue op: status à l'agent wiki)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_status
command-arg-mode: raw
---

`/wikistatus` rapporte l'état de l'ingestion du wiki, de façon déterministe :
l'outil `wiki_status` délègue `op: status` à l'agent wiki et relaie le JSON.
