---
name: switch-model
description: Basculer le modèle IA actif (euria, deepseek, ollm, gemma4, glm4, granite3b, tiron-llm). Prévenir l'utilisateur avant d'exécuter — le gateway redémarre ~5s.
---

# Skill : switch-model

## Rôle

Basculer le modèle IA actif entre les différents modèles disponibles.
Le changement nécessite un redémarrage du gateway (~5 secondes d'indisponibilité).

## Modèles disponibles

| Alias commande | Modèle complet                                              | Notes                            |
|----------------|-------------------------------------------------------------|----------------------------------|
| `euria`        | euria/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8             | **Défaut** — Infomaniak (Suisse) |
| `deepseek`     | deepseek/deepseek-chat                                      | Fallback cloud                   |
| `ollm`         | ollm/near/DeepSeek-V3.1                                     | DeepSeek V3.1 OLLM               |
| `gemma4`       | ollama/gemma4:latest                                        | Gemma 4 8B local                 |
| `glm4`         | ollama/glm4:latest                                          | GLM 4 9B local                   |
| `granite3b`    | ollama/granite4:3b                                          | Granite 4 3B local               |
| `tiron-llm`    | openai/phi-4-mini-instruct                                  | phi-4-mini local (v0.2.0)        |

## Utilisation

```
switch-model <alias>
```

Exemples :
```
switch-model ollm       # Bascule vers DeepSeek V3.1 via OLLM
switch-model deepseek   # Revient à DeepSeek direct
switch-model gemma4     # Bascule vers Gemma 4 local
switch-model lorawiki   # Bascule vers le fine-tune LoRA Wikipedia
```

## Comportement

- Si le modèle demandé est déjà actif : aucune action, message informatif
- Si le modèle change : met à jour `openclaw.json` et redémarre `openclaw-gateway.service`
- Le redémarrage prend ~5 secondes ; prévenir l'utilisateur avant de lancer

## Pour connaître le modèle actif

```bash
cat ~/.openclaw/openclaw.json | python3 -c "import json,sys; print(json.load(sys.stdin)['agents']['defaults']['model']['primary'])"
```
