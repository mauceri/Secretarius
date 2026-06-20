---
name: c
description: "Capturer une URL/note dans le wiki. Dispatch déterministe vers l'outil wiki_capture (délègue op: capture à l'agent wiki)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: wiki_capture
command-arg-mode: raw
---

`/c [#tags] <url|texte>` capture la ressource dans le wiki, de façon déterministe
(aucune décision du modèle) : l'outil `wiki_capture` délègue `op: capture | <args>`
à l'agent wiki et relaie le résultat.
