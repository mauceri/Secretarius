# Rapport d'analyse Codex

## Perimetre
- Fichiers inspectes: README.md, lora_local_train.py, evaluate_merged.py, merge_and_quantize.py, requirements.txt, .gitignore

## Problemes potentiels (ordonnes par impact)
- [Eleve] lora_local_train.py: TrainingArguments utilise `eval_strategy` au lieu de `evaluation_strategy` (risque de TypeError au runtime). Voir lora_local_train.py:277.
- [Moyen] lora_local_train.py: l'option `--device` est ignoree; `device_map` est fixe par `torch.cuda.is_available()` (impossible de forcer CPU/CUDA). Voir lora_local_train.py:166-225.
- [Moyen] lora_local_train.py: `--packing` est `store_true` avec `default=True`, donc impossible a desactiver; incoherent avec un toggle optionnel. Voir lora_local_train.py:147-201.
- [Moyen] evaluate_merged.py: variables d'environnement allocateur definies apres `import torch`, donc sans effet sur l'allocateur ROCm/CUDA. Voir evaluate_merged.py:18-33.
- [Faible] lora_local_train.py: `save_steps` passe a `None` quand `save_strategy != "steps"`, ce qui peut etre invalide selon la version de Transformers. Voir lora_local_train.py:269-271.
- [Faible] requirements.txt: dependances NVIDIA CUDA listees dans un contexte ROCm; risque de conflits ou d'installations inutiles selon l'environnement.

## Recommandations
- Corriger les arguments Trainer (`evaluation_strategy`, `save_steps`) et respecter `--device`.
- Ajouter un flag `--no-packing` ou inverser la logique (`--packing` default False).
- Deplacer la configuration des variables d'environnement avant `import torch` dans evaluate_merged.py.
- Scinder ou documenter les requirements par cible (ROCm vs CUDA).
