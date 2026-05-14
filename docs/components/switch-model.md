---
tags: [documentation, secretarius]
date: 2026-05-14
---

# Composant : switch-model

## Rôle

Skill OpenClaw pour basculer le modèle LLM actif entre les différents backends
disponibles. Le changement redémarre le gateway OpenClaw (~5 secondes d'indisponibilité).

## Prérequis

- OpenClaw configuré avec plusieurs backends dans `openclaw.json`
- Modèles Ollama installés localement si backends ollama utilisés

## Installation

Installé automatiquement par `install.sh` via `openclaw-config/workspace/skills/`.

## Modèles disponibles

| Alias | Modèle | Type |
|-------|--------|------|
| `deepseek` | `deepseek/deepseek-chat` | API distante (défaut) |
| `ollm` | `ollm/near/DeepSeek-V3.1` | OLLM distant |
| `gemma4` | `ollama/gemma4:latest` | Local Ollama |
| `glm4` | `ollama/glm4:latest` | Local Ollama |
| `granite3b` | `ollama/granite4:3b` | Local Ollama (frugal) |
| `lorawiki` | `llamacpp/.../model-Q6_K.gguf` | LoRA local llama.cpp |

## Usage

Via OpenClaw (Telegram) :
```
switch-model deepseek      <- revenir au modèle par défaut
switch-model gemma4        <- basculer vers Gemma 4 local
switch-model granite3b     <- modèle léger pour tâches simples
```

## Comportement

- Si le modèle demandé est déjà actif -> message informatif, aucune action
- Si changement -> mise à jour `~/.openclaw/openclaw.json` + redémarrage gateway

## Connaître le modèle actif

```bash
cat ~/.openclaw/openclaw.json | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['agents']['defaults']['model']['primary'])"
```

## Notes d'architecture

La frugalité de Secretarius se concrétise ici : les modèles locaux (Ollama, llama.cpp)
permettent un fonctionnement hors-réseau. `granite3b` (3B paramètres) convient aux
tâches de capture et résumé simples ; `deepseek` est utilisé pour l'ingestion wiki
et les requêtes complexes.
