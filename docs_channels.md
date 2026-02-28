# Secretarius - Runtime multi-canaux

Ce document décrit l'architecture pour servir le meme agent sur plusieurs canaux.

## Principe

- Coeur unique: `secretarius/agent_runtime.py`
  - encapsule `qwen-agent` (`Assistant`)
  - configure les tools MCP `secretarius` via `run_secretarius_mcp.py`
- Routeur de canaux: `secretarius/channel_adapters.py`
  - interface unifiee `ChannelEvent`
  - point d'entree unique `handle_channel_event(...)`
- Adaptateur OpenAI/Open WebUI: `serverOpenAI.py`
  - traduit `POST /v1/chat/completions` en `ChannelEvent(channel="openwebui_openai", ...)`

## Variables d'environnement

- `SECRETARIUS_QWEN_MODEL` (defaut: `Qwen3-0.6B`)
- `SECRETARIUS_QWEN_MODEL_SERVER` (defaut: `http://127.0.0.1:8000/v1`)
- `SECRETARIUS_QWEN_API_KEY` (defaut: `EMPTY`)
- `SECRETARIUS_MCP_PYTHON` (defaut: `./.venv/bin/python`)
- `SECRETARIUS_MCP_ENTRYPOINT` (defaut: `./run_secretarius_mcp.py`)
- `SECRETARIUS_QWEN_CODE_INTERPRETER` (`true|false`, defaut: `false`)
- `SECRETARIUS_QWEN_THOUGHT_IN_CONTENT` (`true|false`, defaut: `true`)

## Open WebUI (connecteur inerte)

Lancer:

```bash
python serverOpenAI.py
```

Configurer Open WebUI en mode OpenAI-compatible (sans logique metier):
- Base URL: `http://<host>:8000/v1`
- API key: valeur quelconque (non validee ici)
- Model: `secretarius-agent` (ou le champ `model` envoye par Open WebUI)

Le serveur `serverOpenAI.py` reste un adaptateur de canal.
Il ne doit pas porter la logique metier de memoire.

### Service systemd --user (Open WebUI API)

Fichiers fournis:
- unit: `deploy/systemd-user/secretarius-openwebui-api.service`
- env exemple: `deploy/env/openwebui-api.env.example`

Installation:

```bash
mkdir -p ~/.config/secretarius ~/.config/systemd/user
cp /home/mauceric/Secretarius/deploy/env/openwebui-api.env.example ~/.config/secretarius/openwebui-api.env
cp /home/mauceric/Secretarius/deploy/systemd-user/secretarius-openwebui-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now secretarius-openwebui-api.service
```

Diagnostic:

```bash
systemctl --user status secretarius-openwebui-api.service
journalctl --user -u secretarius-openwebui-api.service -f
```

## Extension vers d'autres canaux

Pour Telegram / Messenger / WhatsApp / Email:

1. Ecrire un adaptateur d'entree (webhook/polling) pour ce canal.
2. Construire un `ChannelEvent`.
3. Appeler `handle_channel_event(event)`.
4. Renvoyer `output_text` via l'API du canal.

Le coeur agent et les tools MCP ne changent pas.

## API metier memoire (service separe)

Pour les usages applicatifs (add/search), utiliser `serverMemory.py`:

```bash
python serverMemory.py
```

Endpoints:
- `POST /memory/add`
- `POST /memory/search`

Ce service porte l'orchestration extraction -> embeddings -> Milvus.

## Telegram (concret, polling)

Un connecteur Telegram est disponible:
- `secretarius/telegram_adapter.py`
- entrypoint: `serverTelegram.py`

Variables:
- `TELEGRAM_BOT_TOKEN` (obligatoire)
- `TELEGRAM_ALLOWED_CHAT_IDS` (optionnel, CSV d'identifiants chat)
- `TELEGRAM_POLL_TIMEOUT_S` (defaut: `25`)
- `TELEGRAM_POLL_INTERVAL_S` (defaut: `0.5`)

Lancement:

```bash
python serverTelegram.py
```

Le connecteur transforme chaque message Telegram en `ChannelEvent(channel="telegram", ...)`
et renvoie la reponse de l'agent dans le meme chat.

### Service systemd --user (Telegram)

Fichiers fournis:
- unit: `deploy/systemd-user/secretarius-telegram.service`
- env exemple: `deploy/env/telegram.env.example`

Installation:

```bash
mkdir -p ~/.config/secretarius ~/.config/systemd/user
cp /home/mauceric/Secretarius/deploy/env/telegram.env.example ~/.config/secretarius/telegram.env
cp /home/mauceric/Secretarius/deploy/systemd-user/secretarius-telegram.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now secretarius-telegram.service
```

Diagnostic:

```bash
systemctl --user status secretarius-telegram.service
journalctl --user -u secretarius-telegram.service -f
```

## Email (concret, IMAP + SMTP)

Un connecteur Email est disponible:
- `secretarius/email_adapter.py`
- entrypoint: `serverEmail.py`

Variables IMAP:
- `EMAIL_IMAP_HOST` (obligatoire)
- `EMAIL_IMAP_PORT` (defaut: `993`)
- `EMAIL_IMAP_USER` (obligatoire)
- `EMAIL_IMAP_PASSWORD` (obligatoire)
- `EMAIL_IMAP_MAILBOX` (defaut: `INBOX`)

Variables SMTP:
- `EMAIL_SMTP_HOST` (obligatoire)
- `EMAIL_SMTP_PORT` (defaut: `587`)
- `EMAIL_SMTP_USER` (obligatoire)
- `EMAIL_SMTP_PASSWORD` (obligatoire)
- `EMAIL_SENDER` (obligatoire; defaut implicite: `EMAIL_SMTP_USER`)

Filtrage/rythme:
- `EMAIL_ALLOWED_SENDERS` (optionnel, CSV)
- `EMAIL_POLL_INTERVAL_S` (defaut: `10`)

Lancement:

```bash
python serverEmail.py
```

Le connecteur:
1. lit les messages `UNSEEN` via IMAP,
2. transforme le contenu texte en `ChannelEvent(channel="email", ...)`,
3. envoie la reponse de l'agent par SMTP (`Re:` + `In-Reply-To`/`References`),
4. marque le message comme lu.

### Service systemd --user (Email)

Fichiers fournis:
- unit: `deploy/systemd-user/secretarius-email.service`
- env exemple: `deploy/env/email.env.example`

Installation:

```bash
mkdir -p ~/.config/secretarius ~/.config/systemd/user
cp /home/mauceric/Secretarius/deploy/env/email.env.example ~/.config/secretarius/email.env
cp /home/mauceric/Secretarius/deploy/systemd-user/secretarius-email.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now secretarius-email.service
```

Diagnostic:

```bash
systemctl --user status secretarius-email.service
journalctl --user -u secretarius-email.service -f
```
