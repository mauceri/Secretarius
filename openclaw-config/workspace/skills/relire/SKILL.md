---
name: relire
description: "Afficher la prochaine page du wiki à relire. Dispatch déterministe vers wiki_review."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_review
command-arg-mode: raw
---

`/relire` affiche la prochaine page source résumée par le LLM et non encore
vérifiée (file FIFO), de façon déterministe via l'agent wiki. Lecture libre,
aucune confirmation requise.
