---
tags: [documentation, secretarius]
date: 2026-05-14
---

# Composant : gog

## Rôle

Google Workspace CLI pour Gmail, Calendar, Drive, Contacts, Sheets et Docs.
Permet à l'agent OpenClaw d'interagir avec les services Google via OAuth2.

## Prérequis

- Compte Google avec accès aux APIs souhaitées
- `gog` installé : `brew install steipete/tap/gogcli`
- Fichier `client_secret.json` (Google Cloud Console)

## Installation

```bash
# macOS
brew install steipete/tap/gogcli

# Linux (télécharger le binaire depuis gogcli.sh)
curl -L https://gogcli.sh/install.sh | bash
```

## Configuration OAuth (une fois)

```bash
# 1. Associer les credentials
gog auth credentials /chemin/vers/client_secret.json

# 2. Authentifier le compte
gog auth add vous@gmail.com --services gmail,calendar,drive,contacts,sheets,docs

# 3. Vérifier
gog auth list
```

Le secret `GOG_KEYRING_PASSWORD` dans `gateway.systemd.env` protège le keyring
des tokens OAuth.

## Désinstallation

```bash
gog auth remove vous@gmail.com
brew uninstall gogcli   # macOS
```

## Usage courant

```bash
# Gmail
gog gmail search 'newer_than:7d' --max 10
gog gmail send --to dest@example.com --subject "Sujet" --body "Corps"

# Récupérer un message (par ID de message, pas de fil)
gog email messages search "requête" -j --results-only
gog email get <messageId>   # champ "id", PAS "threadId"

# Calendar
gog calendar events <calendarId> --from 2026-05-14 --to 2026-05-21

# Drive
gog drive search "rapport" --max 10

# Sheets
gog sheets get <sheetId> "Feuille1!A1:D10" --json
gog sheets update <sheetId> "Feuille1!A1" --values-json '[["valeur"]]' --input USER_ENTERED

# Docs
gog docs cat <docId>
gog docs export <docId> --format txt --out /tmp/doc.txt
```

## Piège connu

`gog email search` retourne des **identifiants de fil** (`threadId`).
Pour `gog email get`, utiliser le champ `id` (identifiant de message), pas `threadId`.

## Notes d'architecture

Toujours demander confirmation avant d'envoyer un mail ou de créer un événement.
Définir `GOG_ACCOUNT=vous@gmail.com` pour éviter `--account` à chaque commande.
Pour les scripts : préférer `--json` avec `--no-input`.
