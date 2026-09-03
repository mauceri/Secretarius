# CLAUDE.md — Projet Secretarius

## Artefacts précieux (hors dépôt)

- `~/lora_slm/` — scripts et artefacts du pipeline LoRA (adaptateurs, checkpoints,
  GGUF). Coûteux à reproduire : ne pas déplacer ni supprimer sans confirmation.
- `~/Modèles/` — modèles GGUF servis par les services llama.cpp.

## Règles importantes

- **Vouvoiement** : l'utilisateur préfère être vouvoyé — s'adresser à lui à la
  deuxième personne du pluriel (« vous », « votre »), jamais en tutoiement.
- Confirmation requise avant : `systemctl start/stop/enable/restart`,
  `docker compose up/down`, `git push`.
- Le wiki (`WIKI_PATH`) est partagé entre sanroque et santiago via un unique coffre
  Obsidian synchronisé, et le verrou d'ingestion est **local à chaque machine** :
  ne jamais lancer deux ingestions en même temps depuis deux machines.
- `git *` est pré-approuvé sauf `push --force`.

## Sessions Claude Code — protection de l'historique (incident du 2026-09-03)

- Le 2026-09-03, le nettoyage automatique de Claude Code (30 jours par défaut)
  a supprimé les transcripts de mars à juillet 2026. Reconstitution :
  `~/Documents/Arbath/Interactions/Claude/2026-09-03-chronologie-reconstituee.md`.
- `cleanupPeriodDays` est fixé à 36500 dans `~/.claude/settings.json` : ne pas
  le retirer.
- Les transcripts et la mémoire sont rangés par répertoire de lancement. Les
  sessions historiques sont sous `~/.claude/projects/-home-mauceric/` (lancement
  depuis `~`). La mémoire de `~/Secretarius` est un lien symbolique vers celle
  de `~` : ne jamais la remplacer par un répertoire.
- Sauvegarde quotidienne (timer utilisateur `claude-code-backup.timer`) de
  `~/.claude/projects` et `history.jsonl` vers
  `~/Documents/Arbath/Interactions/Claude/claude_code_backup/`.
- En fin de session importante : écrire une fiche mémoire datée (état, commits,
  reste à faire) — c'est ce qui a permis la reconstitution.
