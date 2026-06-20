---
name: chercher
description: "Rechercher des emails Gmail. Dispatch déterministe vers gog_search (délègue à l'agent gog)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_search
command-arg-mode: raw
---

`/chercher <requête>` recherche des emails Gmail, de façon déterministe via l'agent gog (lecture seule).
