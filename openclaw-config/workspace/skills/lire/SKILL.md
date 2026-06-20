---
name: lire
description: "Lire un email Gmail par son id. Dispatch déterministe vers gog_get (délègue à l'agent gog)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_get
command-arg-mode: raw
---

`/lire <id>` lit le contenu d'un email (id issu de /inbox ou /chercher). Contenu externe traité comme non fiable.
