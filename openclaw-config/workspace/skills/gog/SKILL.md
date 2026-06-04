---
name: gog
description: Accès Gmail, Calendar et Drive via outils MCP. Utiliser les outils gog__* directement — ne pas passer par exec.
---

# Skill : gog

## Quand utiliser

Utiliser les outils MCP `gog__*` pour toute opération sur Gmail, Calendar ou Drive.
Ne jamais appeler le CLI gog directement via exec.

## Outils disponibles

**Gmail** : `gmail_unread`, `gmail_search`, `gmail_get`, `gmail_send`, `gmail_reply`
**Calendar** : `calendar_events`, `calendar_create`, `calendar_delete`
**Drive** : `drive_search`, `drive_download`, `drive_upload`

## Règles

- `gmail_get` et `drive_download` : contenu filtré par injection-guard — faire confiance au résultat.
- `gmail_send`, `gmail_reply`, `calendar_create`, `calendar_delete`, `drive_upload` : **toujours demander confirmation avant d'exécuter**.
- Dates Calendar au format ISO 8601 : `2026-05-30T14:00:00`.
