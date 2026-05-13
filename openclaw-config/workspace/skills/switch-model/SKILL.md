---
name: switch-model
description: Basculer le modèle IA actif (deepseek, ollm, gemma4, glm4, granite3b, lorawiki). Prévenir l'utilisateur avant d'exécuter — le gateway redémarre ~5s.
---

# Skill : switch-model

## Rôle

Basculer le modèle IA actif entre les différents modèles disponibles.
Le changement nécessite un redémarrage du gateway (~5 secondes d'indisponibilité).

## Modèles disponibles

| Alias commande | Modèle complet                                                              | Notes                  |
|----------------|-----------------------------------------------------------------------------|------------------------|
| `deepseek`     | deepseek/deepseek-chat                                                      | Modèle par défaut      |
| `ollm`         | ollm/near/DeepSeek-V3.1                                                     | DeepSeek V3.1 OLLM     |
| `gemma4`       | ollama/gemma4:latest                                                        | Gemma 4 8B local       |
| `glm4`         | ollama/glm4:latest                                                          | GLM 4 9B local         |
| `granite3b`    | ollama/granite4:3b                                                          | Granite 4 3B local     |
| `lorawiki`     | llamacpp/${HOME}/lora_local/test_wikipedia_gguf/model-Q6_K.gguf             | LoRA Wikipedia local   |

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
