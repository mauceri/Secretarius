# CLAUDE.md — Projet Secretarius

## Artefacts précieux (hors dépôt)

- `~/lora_slm/` — scripts et artefacts du pipeline LoRA (adaptateurs, checkpoints,
  GGUF). Coûteux à reproduire : ne pas déplacer ni supprimer sans confirmation.
- `~/Modèles/` — modèles GGUF servis par les services llama.cpp.

## Règles importantes

- Confirmation requise avant : `systemctl start/stop/enable/restart`,
  `docker compose up/down`, `git push`.
- Le wiki (`WIKI_PATH`) est partagé entre sanroque et santiago via un unique coffre
  Obsidian synchronisé, et le verrou d'ingestion est **local à chaque machine** :
  ne jamais lancer deux ingestions en même temps depuis deux machines.
- `git *` est pré-approuvé sauf `push --force`.
