---
name: tags
description: "Lister les tags de la base de connaissances. Dispatch déterministe vers wiki_tags."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_tags
command-arg-mode: raw
---

`/tags` liste les tags du wiki, de façon déterministe via l'agent wiki.
