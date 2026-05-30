# Design : gog MCP Server

**Date** : 2026-05-30
**Branche** : `v0.2.0-dev`
**Objectif** : Exposer Gmail, Calendar et Drive à Tiron via un serveur MCP wrappant le CLI `gog`, avec filtrage injection-guard sur le contenu entrant.

---

## Contexte

`gog` v0.9.0 est installé sur sanroque (`/home/linuxbrew/.linuxbrew/bin/gog`) et authentifié via OAuth. Le skill `gog/SKILL.md` existant est insuffisant : il demande à Tiron de mémoriser toute la syntaxe CLI et est incohérent. Un serveur MCP remplace cette approche — Tiron appelle des outils nommés, sans connaître gog.

Le pattern suit `wiki-lm` : serveur Python fastmcp, subprocess stdio, intégration dans `openclaw.json.template`.

---

## Architecture

```
Tiron (OpenClaw)
    → outil MCP (ex. gmail_get)
        → gog_mcp/mcp_server.py  (subprocess fastmcp)
            → injection_guard :8990  (pour contenu entrant uniquement)
            → gog CLI  (subprocess)
                → API Google
```

**Emplacement** : `gog_mcp/mcp_server.py` à la racine du repo.
**Venv** : réutilise `Wiki_LM/.venv` (fastmcp déjà installé).
**Démarrage** : subprocess stdio lancé par OpenClaw au démarrage du gateway.

---

## Outils (11)

### Gmail

| Outil | Paramètres | Filtrage |
|-------|-----------|---------|
| `gmail_unread` | `max: int = 10` | non (métadonnées) |
| `gmail_search` | `query: str`, `max: int = 10` | non (métadonnées) |
| `gmail_get` | `message_id: str` | **oui** (contenu) |
| `gmail_send` | `to: str`, `subject: str`, `body: str`, `cc: str = ""` | — (écriture) |
| `gmail_reply` | `message_id: str`, `body: str` | — (écriture) |

### Calendar

| Outil | Paramètres | Filtrage |
|-------|-----------|---------|
| `calendar_events` | `from_date: str`, `to_date: str`, `calendar_id: str = "primary"` | non (métadonnées) |
| `calendar_create` | `title: str`, `start: str`, `end: str`, `calendar_id: str = "primary"`, `description: str = ""` | — (écriture) |
| `calendar_delete` | `event_id: str`, `calendar_id: str = "primary"` | — (écriture) |

### Drive

| Outil | Paramètres | Filtrage |
|-------|-----------|---------|
| `drive_search` | `query: str`, `max: int = 10` | non (métadonnées) |
| `drive_download` | `file_id: str`, `filename: str` | **oui** (contenu) |
| `drive_upload` | `file_path: str`, `folder_id: str = ""` | — (écriture) |

`drive_download` télécharge dans `~/Downloads/gog/<filename>` (répertoire fixe, pas de path arbitraire).

---

## Filtrage injection-guard

Les outils retournant du **contenu** (`gmail_get`, `drive_download`) passent par `injection_guard` sur `http://localhost:8990/check` avant de retourner à Tiron — même interface que `wiki_ingest`.

- Si `injection_guard` est indisponible → retourner `{"ok": false, "error": "injection_guard indisponible — contenu non transmis"}`
- Si le contenu est bloqué → retourner `{"ok": false, "blocked": true, "reason": "..."}`
- Si le contenu est propre → retourner le contenu nettoyé (`clean_text`)

Les outils de **recherche** (métadonnées uniquement) et d'**écriture** ne passent pas par injection_guard.

---

## Gestion des erreurs

Tous les outils retournent `{"ok": true, ...}` ou `{"ok": false, "error": "..."}`.

| Cas | Comportement |
|-----|-------------|
| Timeout gog (30s) | `{"ok": false, "error": "timeout"}` |
| Code retour non-nul | `{"ok": false, "error": "<stderr>"}` |
| JSON invalide | `{"ok": false, "error": "parse_error"}` |
| `GOG_ACCOUNT` absent | Erreur au démarrage du serveur, logged |

---

## Outils d'écriture — confirmation obligatoire

Les outils `gmail_send`, `gmail_reply`, `calendar_create`, `calendar_delete`, `drive_upload` incluent dans leur description MCP la mention `⚠ Demander confirmation avant d'exécuter`. Tiron voit cette description et applique la règle de `AGENTS.md`.

---

## Intégration OpenClaw

### `openclaw.json.template` — section `mcp.servers`

```json
"gog": {
  "command": "${HOME}/Secretarius/Wiki_LM/.venv/bin/python3",
  "args": ["${HOME}/Secretarius/gog_mcp/mcp_server.py"]
}
```

### `tools.sandbox.tools.allow`

11 entrées à ajouter : `gog__gmail_unread`, `gog__gmail_search`, `gog__gmail_get`, `gog__gmail_send`, `gog__gmail_reply`, `gog__calendar_events`, `gog__calendar_create`, `gog__calendar_delete`, `gog__drive_search`, `gog__drive_download`, `gog__drive_upload`.

(+ variantes préfixées `gog__` selon la convention OpenClaw)

### `skills/gog/SKILL.md`

Remplacé par une version courte : contexte uniquement (quand utiliser les outils gog), sans syntaxe CLI.

---

## Structure du repo

```
gog_mcp/
└── mcp_server.py      # serveur fastmcp, 11 outils

openclaw-config/
├── openclaw.json.template   # ajout section mcp.servers.gog + tools.allow
└── workspace/skills/gog/SKILL.md   # simplifié
```

---

## Ce qui ne change pas

- `injection_guard.py` et son service systemd
- `wiki-lm` MCP server
- Auth gog (déjà configurée sur sanroque)
