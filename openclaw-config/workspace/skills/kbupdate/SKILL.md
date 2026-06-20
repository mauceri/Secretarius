---
name: kbupdate
description: "Mettre à jour la base de connaissances depuis le dernier clustering. Dispatch déterministe vers wiki_kb_update."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_kb_update
command-arg-mode: raw
---

`/kbupdate` lance la mise à jour du KB en arrière-plan, de façon déterministe via l'agent wiki.
