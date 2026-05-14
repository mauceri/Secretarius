---
tags: [documentation, secretarius]
date: 2026-05-14
---

# Composant : c (Telegram -> Secretarius)

## Rôle

Skill de capture rapide depuis Telegram. Sauvegarde des URLs, textes libres et
fichiers dans `raw/` pour ingestion ultérieure dans le wiki. Fonctionne aussi
comme point d'entrée principal pour interagir avec Secretarius via Telegram.

## Prérequis

- Bot Telegram créé via BotFather
- OpenClaw installé et configuré avec `TELEGRAM_BOT_TOKEN`
- Service `openclaw-gateway.service` actif

## Connexion de Telegram à Secretarius

### 1. Créer un bot BotFather

1. Ouvrir Telegram, chercher `@BotFather`
2. Envoyer `/newbot` et suivre les instructions
3. Récupérer le token (format : `1234567890:ABCdef...`)

### 2. Configurer les secrets

```bash
nano ~/.openclaw/gateway.systemd.env
```

Renseigner :
```
TELEGRAM_BOT_TOKEN=<token BotFather>
OPENCLAW_GATEWAY_TOKEN=<votre identifiant Telegram numérique>
GATEWAY_PASSWORD=<mot de passe choisi librement>
```

Pour trouver votre identifiant Telegram numérique :
envoyer un message à `@userinfobot` sur Telegram.

### 3. Démarrer le service

```bash
systemctl --user daemon-reload
systemctl --user enable --now openclaw-gateway.service
systemctl --user status openclaw-gateway.service
```

### 4. Appairer

Envoyer `/start` au bot Telegram, puis :
```bash
openclaw pairing approve telegram <CODE_AFFICHÉ>
```

## Usage

### Capturer une URL

```
/c https://example.com/article
```

Crée `raw/YYYYMMDD-HHMMSS-example.url`

### Capturer un texte

```
/c #memo Note importante sur les transformers
```

Crée `raw/YYYYMMDD-HHMMSS-note-importante.md`

### Capturer une URL avec commentaire

```
/c #attention https://arxiv.org/abs/1706.03762 Article fondateur
```

Crée un `.md` avec URL + commentaire + tags.

### Envoyer une URL nue (partage Android)

Envoyer directement `https://...` sans préfixe `/c` — détecté automatiquement.

### Capturer un fichier joint

Envoyer le fichier avec un message optionnel `#tags commentaire`.

## Notes d'architecture

Le skill `c` utilise `capture.py` de Wiki_LM. Les fichiers créés dans `raw/`
sont ensuite ingérés via `ingest.py --raw`. Le contenu Telegram passe par
`prompt-injection-guard` avant traitement par l'agent principal.
