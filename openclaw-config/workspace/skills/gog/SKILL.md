---
name: gog
description: Google Workspace CLI for Gmail, Calendar, Drive, Contacts, Sheets, and Docs.
homepage: https://gogcli.sh
metadata: {"clawdbot":{"emoji":"🎮","requires":{"bins":["gog"]},"install":[{"id":"brew","kind":"brew","formula":"steipete/tap/gogcli","bins":["gog"],"label":"Install gog (brew)"}]}}
---

# gog

Utiliser `gog` pour Gmail, Calendar, Drive, Contacts, Sheets et Docs. Nécessite une configuration OAuth.

## Configuration (une fois)

```bash
gog auth credentials /chemin/vers/client_secret.json
gog auth add vous@gmail.com --services gmail,calendar,drive,contacts,sheets,docs
gog auth list
```

## Commandes courantes

- Recherche Gmail : `gog gmail search 'newer_than:7d' --max 10`
- Envoi Gmail : `gog gmail send --to a@b.com --subject "Sujet" --body "Corps"`
- Agenda : `gog calendar events <calendarId> --from <iso> --to <iso>`
- Recherche Drive : `gog drive search "requête" --max 10`
- Contacts : `gog contacts list --max 20`
- Lecture Sheets : `gog sheets get <sheetId> "Onglet!A1:D10" --json`
- Mise à jour Sheets : `gog sheets update <sheetId> "Onglet!A1:B2" --values-json '[["A","B"],["1","2"]]' --input USER_ENTERED`
- Ajout Sheets : `gog sheets append <sheetId> "Onglet!A:C" --values-json '[["x","y","z"]]' --insert INSERT_ROWS`
- Effacement Sheets : `gog sheets clear <sheetId> "Onglet!A2:Z"`
- Métadonnées Sheets : `gog sheets metadata <sheetId> --json`
- Export Docs : `gog docs export <docId> --format txt --out /tmp/doc.txt`
- Lecture Docs : `gog docs cat <docId>`

## Notes importantes

- Définir `GOG_ACCOUNT=vous@gmail.com` pour éviter de répéter `--account`.
- Pour les scripts, préférer `--json` avec `--no-input`.
- Les valeurs Sheets se passent via `--values-json` (recommandé).
- Docs supporte export/cat/copy. Les modifications en place nécessitent un client API Docs (absent de gog).
- **Toujours demander confirmation avant d'envoyer un mail ou de créer un événement.**

## Piège connu — identifiant de fil vs identifiant de message

`gog email search` retourne des **identifiants de fil** → ne pas utiliser avec `gog email get`.

```bash
# Correct : récupérer l'identifiant de message
gog email messages search "requête" -j --results-only
gog email get <identifiantMessage>   # champ "id", pas "threadId"
```
