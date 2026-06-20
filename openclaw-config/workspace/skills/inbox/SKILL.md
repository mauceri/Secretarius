---
name: inbox
description: "Lister les emails récents. Dispatch déterministe vers l'outil gog_inbox (délègue à l'agent gog qui exécute gog en sandbox)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_inbox
command-arg-mode: raw
---

`/inbox [requête Gmail]` liste les emails récents (défaut : in:inbox), de façon
déterministe : l'outil gog_inbox délègue à l'agent gog (lecture seule).
