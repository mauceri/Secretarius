---
name: drive
description: "Rechercher des fichiers Google Drive. Dispatch déterministe vers gog_drive_search (délègue à l'agent gog)."
user-invocable: true
disable-model-invocation: true
command-dispatch: tool
command-tool: gog_drive_search
command-arg-mode: raw
---

`/drive <requête>` recherche des fichiers Drive, de façon déterministe via l'agent gog (lecture seule).
