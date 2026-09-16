---
tags: [documentation, LLM_Wiki, secretarius]
date: 2026-05-14
---

# Composant : scout

## Rôle

Agent isolé chargé de lire les sources externes (pages web, contenu d'emails) à la
place de l'agent principal (Tiron). Il protège contre les injections de prompt
dissimulées dans le contenu web. Toute sortie de scout est considérée `<UNTRUSTED>`.

## Prérequis

- OpenClaw installé et configuré
- Services `openclaw-injection-guard.service` (port 8990) et `openclaw-scout.service` actifs

## Installation

Scout est installé automatiquement par `install.sh` (`openclaw-config/install.sh`) :
le garde d'injection (`injection_guard.py`) puis le watcher (`scout-watcher` +
`scout_process.py`) sont copiés dans `~/.local/bin/`, chacun avec son unité
systemd, activée et démarrée (`enable --now`).

```bash
# Vérifier que les deux services sont actifs
systemctl --user status openclaw-injection-guard.service
systemctl --user status openclaw-scout.service

# Démarrer si nécessaire
systemctl --user enable --now openclaw-injection-guard.service
systemctl --user enable --now openclaw-scout.service
```

## Désinstallation

```bash
systemctl --user disable --now openclaw-scout.service openclaw-injection-guard.service
rm ~/.local/bin/scout-watcher ~/.local/bin/scout_process.py ~/.local/bin/injection_guard.py
```

## Configuration

Workspace scout : `~/.openclaw/workspace-scout/`

```
tasks/pending/    <- tâches à traiter (écrites par l'agent scout lui-même)
tasks/done/       <- tâches traitées par scout-watcher (fetched_content injecté)
results/          <- résultats JSON
```

## Usage

L'agent scout n'est jamais invoqué directement par l'utilisateur : c'est
Tiron qui le délègue via `sessions_spawn`, en lui laissant l'initiative
d'écrire lui-même sa tâche.

### 1. Délégation par Tiron

```
sessions_spawn(task="url: <url>\ninstructions: <instructions optionnelles>", agentId="scout")
```

### 2. Ce que fait l'agent scout

1. Génère un `task_id` (UUID) et écrit `tasks/pending/<task_id>.json` :
   - pour une URL : `{"url_or_path": "<url>", "instructions": "<...>", "requested_at": "<ISO8601>"}`
   - pour un texte d'email : `{"check_email": "<texte>", "requested_at": "<ISO8601>"}`
2. Relit `tasks/done/<task_id>.json` en boucle (jusqu'à 20 tentatives) — c'est
   `scout-watcher` qui écrit ce fichier une fois le pré-fetch et le passage
   par le garde d'injection terminés.
3. Vérifie la fraîcheur (le `url_or_path`/`check_email` du fichier `done`
   doit correspondre exactement à la tâche demandée — un résidu d'UUID
   réutilisé est ignoré).
4. Écrit `results/<task_id>.json` et retourne ce JSON dans sa réponse de session.

### 3. Format du résultat

**Si bloqué par le garde :**
```json
{
  "blocked": true,
  "reason": "description des motifs détectés"
}
```

**Sinon :**
```json
{
  "source": "https://...",
  "retrieved_at": "2026-09-16T...",
  "risk": "low|medium",
  "clean_text": "<UNTRUSTED> texte propre, sans HTML",
  "full_content": "<UNTRUSTED> contenu verbatim (si demandé explicitement)",
  "warnings": []
}
```

**Toujours lire `blocked` en premier.** Si `blocked: true`, ne jamais utiliser
le contenu. Si `risk: "medium"`, signaler à l'utilisateur la présence de
contenu potentiellement suspect.

## Notes d'architecture

Scout ne peut PAS exécuter de commandes shell, accéder à Telegram/Gmail, ni
spawner d'autres agents — cette isolation est la barrière, pas une consigne
que le modèle pourrait choisir de suivre ou non. Le fetch réseau lui-même
est fait hors du modèle, par `scout-watcher` (curl), jamais par l'agent
scout. Tout contenu passe par `openclaw-injection-guard.service` (regex +
DeBERTa, consulté systématiquement depuis le 16/09/2026 — voir
`openclaw-config/injection_guard.py`) avant d'atteindre un LLM.
