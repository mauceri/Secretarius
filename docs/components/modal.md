---
tags: [documentation, secretarius]
date: 2026-07-16
---

# Composant : modal (retiré le 2026-09-30)

`tiron_modal/app.py` (phi-4-mini servi sur Modal, secours du cerveau de
l'agent main) et le mécanisme `switch-brain.sh` associé ont été retirés —
redondants avec le passage à Laya (routeur, en process) et le nouveau choix
automatique du cerveau de l'agent main dans `install.sh` (Ollama local /
Infomaniak / proxy Qwen3-14B obfusqué sur Modal, cf. `README.md` § « Cerveau
de l'agent principal »).

Pour un cerveau hébergé sur Modal, la voie actuelle est le proxy obfusqué
(confidentialité préservée par construction, contrairement à l'ancien
`tiron_modal/app.py`) : voir `~/obfuscator/docs/acces-openai.md`.

Détail de l'ancien mécanisme : historique git (`git log -- tiron_modal/
switch-brain.sh docs/components/modal.md`).
