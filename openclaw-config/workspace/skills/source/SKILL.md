---
name: source
description: "Lire/consulter une page web externe MAINTENANT via scout (anti-injection). Dispatch déterministe vers l'outil source_read (délègue url: <url> à l'agent scout)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: source_read
command-arg-mode: raw
---

`/source <url>` lit le contenu d'une page externe via l'agent scout (filtrage
anti-injection), de façon déterministe : l'outil `source_read` délègue
`url: <url>` à scout et relaie le contenu nettoyé (`<UNTRUSTED>`). Sans rapport
avec le wiki ; ne sauvegarde rien.
