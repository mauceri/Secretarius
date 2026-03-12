# Session Messenger Bridge

Ce dossier contient le pont Bun/TypeScript entre Session Messenger et le canal Python `session_messenger` de Secretarius.

## Prérequis

- Bun installe et disponible dans le `PATH`
- dependances Node/Bun installees via `bun install`
- API Python `session_messenger` demarree depuis `Prototype`

## Installation

```bash
cd /home/mauceric/Secretarius/Prototype/session_messenger_bridge
bun install
```

## Lancement

Dans un premier terminal :

```bash
cd /home/mauceric/Secretarius/Prototype
../.venv/bin/python main_multicanal.py
```

Dans un second terminal :

```bash
cd /home/mauceric/Secretarius/Prototype/session_messenger_bridge
bun run start
```

## Verification

Le fichier suivant doit etre cree au demarrage du bridge :

```bash
tail -f /home/mauceric/Secretarius/Prototype/logs/session_messenger.log
```

Ligne attendue :

```text
bot_session_id=<identifiant-session-du-bot>
```

Ensuite, un message envoye au bot doit produire :

- une ligne `CHAT USER`
- une ligne `CHAT ASSISTANT`

## Variables utiles

- `SESSION_MESSENGER_API_URL` : URL de base de l'API Python, par defaut `http://127.0.0.1:8002`
- `SESSION_MESSENGER_JOURNAL_PATH` : chemin du log de canal
- `SESSION_MESSENGER_SQLITE_PATH` : chemin de la base SQLite de deduplication
- `SESSION_BOT_MNEMONIC` : mnemotechnique Session du bot
- `SESSION_BOT_DISPLAY_NAME` : nom affiche du bot dans Session
