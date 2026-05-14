---
tags: [documentation, LLM_Wiki, secretarius]
date: 2026-05-14
---

# Composant : superpowers

## Rôle

Workflow de développement logiciel structuré pour les agents IA (Claude Code,
OpenClaw). Adapté de obra/superpowers (https://github.com/obra/superpowers).
Impose un pipeline spec-first + TDD + sous-agents pour tout développement.

## Prérequis

- Claude Code installé (`claude`) pour le plugin Claude Code
- OpenClaw pour le skill OpenClaw

## Installation

### Plugin Claude Code

```bash
# Vérifier si déjà installé
ls ~/.claude/plugins/superpowers/ 2>/dev/null && echo "OK" || echo "A installer"

# Installer manuellement
mkdir -p ~/.claude/plugins
git clone --depth=1 https://github.com/obra/superpowers /tmp/superpowers-src
cp -r /tmp/superpowers-src/. ~/.claude/plugins/superpowers/
rm -rf /tmp/superpowers-src
```

### Skill OpenClaw

Installé automatiquement par `install.sh` via `openclaw-config/workspace/skills/superpowers/`.

## Pipeline

```
Idée -> Brainstorm -> Plan -> Build (TDD + sous-agents) -> Review -> Finish
```

### Phase 1 : Brainstorming

Avant tout code : explorer le contexte, questions clarificatrices (une à la fois),
2-3 approches, design en sections, spec sauvée dans `docs/superpowers/specs/`.

**HARD GATE :** aucun code avant approbation du design.

### Phase 2 : Writing Plans

Plan détaillé tâche par tâche, chaque tâche = 2-5 min. TDD : test échoue -> implémente -> test passe -> commit.
Sauvé dans `docs/superpowers/plans/`.

### Phase 3 : Subagent-Driven Development

Un sous-agent par tâche (`sessions_spawn`), double review (spec + qualité) entre chaque.

### Phase 4 : Systematic Debugging

Root cause avant tout fix. Quatre phases : investigation -> patterns -> hypothèse+test -> fix+vérification.

### Phase 5 : Finishing Branch

Tests passent -> merge/PR/push au choix.

## Déclencheurs

- "Construisons X" -> Phase 1 (Brainstorming)
- "Ce bug ne passe pas" -> Phase 4 (Debugging)
- "Tous les tests passent" -> Phase 5 (Finish Branch)

## Notes d'architecture

Superpowers impose une discipline de développement plutôt qu'un outil. Le SKILL.md
d'OpenClaw et le plugin Claude Code partagent les mêmes principes mais s'adaptent
aux outils de chaque plateforme (`sessions_spawn` pour OpenClaw, Agent tool pour
Claude Code).
