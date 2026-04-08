# Analyse approfondie — `lora_local/`

*Générée le 2026-04-08*

---

## 1. Structure générale

Le répertoire est un projet complet de fine-tuning LoRA pour Phi-4-mini en environnement ROCm (AMD GPU).

| Répertoire | Rôle |
|-----------|------|
| `data/` | Corpus d'entraînement et de test (JSONL) |
| `models/phi4/` | Modèle de base Phi-4-mini-instruct (non fusionné) |
| `checkpoints_phi4_lora/` | Adaptateur LoRA entraîné (version stable nov 2025) |
| `checkpoints_phi4_lora.precieux/` | Sauvegarde du checkpoint initial |
| `gguf_out/phi4_merged/` | Modèles fusionnés et quantifiés (prod nov 2025) |
| `gguf_out.précieux/phi4_merged/` | Sauvegarde de la base fusionnée (HF uniquement) |
| `test_wikipedia/` | Checkpoint LoRA fine-tuné sur Wikipedia (jan 2026) |
| `test_wikipedia_gguf/` | Modèles GGUF quantifiés du checkpoint Wikipedia |
| `test_wikipedia_merged/` | Modèle fusionné Wikipedia (format HF) |
| `grenier/` | Archives notebooks et scripts anciens |
| `llenv/` | Environnement Python virtuel |

---

## 2. Versions des modèles GGUF

### Modèles actifs — `gguf_out/phi4_merged/` (nov 2025)

| Fichier | Taille | Quantization |
|---------|--------|-------------|
| `model-f16.gguf` | 7.2 GB | Float16 (non quantifié) |
| `model-Q4_K_M.gguf` | 2.4 GB | 4-bit |
| `model-Q5_K_M.gguf` | 2.7 GB | 5-bit |
| `model-Q6_K.gguf` | 3.0 GB | 6-bit |

Créés le 29 novembre 2025 à partir de `checkpoints/phi4-lora-v1`.

### Modèles Wikipedia — `test_wikipedia_gguf/` (jan 2026)

Mêmes quantizations, créés le 22 janvier 2026 à partir de :
- Base : `gguf_out.précieux/phi4_merged/merged_hf`
- LoRA : `test_wikipedia/` (fine-tuning Wikipedia FR 1000 chunks, 3 epochs)

**Modèle actif en production :** `test_wikipedia_gguf/model-Q6_K.gguf` (port 8989)

---

## 3. Processus d'entraînement (`lora_local_train.py`)

### Hyperparamètres par défaut

| Paramètre | Valeur |
|-----------|--------|
| `--model_path` | `models/phi4` |
| `--max_len` | 512 tokens |
| `--epochs` | 1 |
| `--per_device_batch` | 1 |
| `--grad_accum` | 16 |
| `--lr` | 2e-5 |
| `--optimizer` | adamw_torch |
| `--seed` | 42 |
| `--packing` | True |

### Configuration LoRA

```json
{
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
}
```

### Prompt d'entraînement (à utiliser impérativement)

**System prompt :**
```
Vous êtes un extracteur d'expressions clés. Répondez UNIQUEMENT par un tableau JSON de chaînes, sans commentaire. Incluez UNIQUEMENT les expressions, dates et lieux remarquables, évènements, qui apparaissent à l'identique dans le texte.
```

**User prefix :**
```
Quelles sont les expressions clés contenues à l'identique dans ce texte :
```

> ⚠️ Tout changement de ce prompt sort le modèle de son régime d'entraînement et dégrade les résultats.

### Format des données

Deux formats supportés :
1. **Format messages (ancien)** : `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
2. **Format chunks (nouveau)** : `{"source":"wiki40b", "titre":"...", "chunks":[{"chunk":"...", "expressions_caracteristiques":["..."]}], "meta":{...}}`

Tokenization : troncature à **gauche** pour préserver le tag `<|assistant|>:` en fin de séquence. Seule la partie assistant est entraînée (labels masqués à -100 pour le reste).

---

## 4. Corpus disponibles

| Fichier | Taille | Type |
|---------|--------|------|
| `corpus_wiki40b_fr_indexed_100.jsonl` | 100 exemples | Wikipedia FR |
| `corpus_wiki40b_fr_indexed_1000.jsonl` | 1000 exemples | Wikipedia FR (utilisé pour test_wikipedia) |
| `corpus_synth_indexed_500.jsonl` | 500 exemples | Synthétique |
| `corpus_synth_indexed_1000.jsonl` | 1000 exemples | Synthétique |
| `corpus_synth_indexed_4000.jsonl` | 4000 exemples | Synthétique |
| `corpus_gutenberg_indexed.jsonl` | — | Classiques français |
| `extrait_corpus1.jsonl` | 100 lignes | Format messages ancien |
| `extrait_corpus2.jsonl` | — | Format messages ancien |
| `data/train.jsonl` | 90% de wiki 1000 | Split entraînement |
| `data/test.jsonl` | 10% de wiki 1000 | Split test |

---

## 5. Évaluation

### Tableau comparatif des perplexités

| Date | Modèle | Dataset | Exemples | Loss | Perplexity |
|------|--------|---------|----------|------|-----------|
| 26 nov 2025 | gguf_out/phi4_merged | data/test.jsonl | 1026 | 0.4851 | 1.6243 |
| 21 jan 2026 | gguf_out.précieux (baseline) | data/test.jsonl | 239 | 0.9662 | 2.6280 |
| 22 jan 2026 | **test_wikipedia** | data/test_100.jsonl | 294 | 0.3057 | **1.3576** |
| 22 jan 2026 | test_wikipedia | tests_complets.jsonl | 1320 | 0.7595 | 2.1373 |
| 23 jan 2026 | test_wikipedia | corpus_synth_1000 | 1000 | 0.2648 | **1.3032** |

**Meilleure performance :** `test_wikipedia` sur corpus synthétique — PPL 1.3032.

### Fichiers de métriques
- `metrics_21_01_2026_1000.json` : loss=0.9662, perplexity=2.6280 (baseline)
- `metrics_21_01_2026_100_baseline.json` : loss=0.9519, perplexity=2.5907 (baseline)

---

## 6. Checkpoints LoRA

### `checkpoints_phi4_lora/` (nov 2025)
- Steps : 306, 612, 918 (final, epoch 1/1)
- Adapter : 35 MB (`adapter_model.safetensors`)

### `test_wikipedia/` (jan 2026)
- Steps : 162, 324, 486 (final, epoch 3/3)
- Loss initiale : ~1.78 → Loss finale : ~0.95
- Adapter : 35 MB

---

## 7. Scripts principaux

| Script | Rôle |
|--------|------|
| `lora_local_train.py` | Entraînement LoRA |
| `merge_and_quantize.py` | Fusion base+LoRA → GGUF quantifiés |
| `evaluate_merged.py` | Évaluation perplexité sur corpus test |
| `split_corpus.py` | Split train/test (seed=42, ratio 90/10) |
| `run_split_and_eval.py` | Pipeline complet split → train → eval |
| `convert_old_to_new.py` | Conversion format messages → chunks |
| `jaccard.py` | Similarité Jaccard entre expressions |

### Commande de création de `test_wikipedia`

```bash
HF_HOME=/home/mauceric/lora_local/.hf_cache \
python lora_local_train.py \
  --model_path gguf_out.précieux/phi4_merged/merged_hf \
  --data_file data/corpus_wiki40b_fr_indexed_1000.jsonl \
  --output_dir test_wikipedia \
  --epochs 3 \
  --log_file test_wikipedia/training.log
```

### Commande de fusion/quantization

```bash
python merge_and_quantize.py \
  --base gguf_out.précieux/phi4_merged/merged_hf/ \
  --lora test_wikipedia/ \
  --out test_wikipedia_merged/ \
  --gguf-dir test_wikipedia_gguf/ \
  --llama-cpp ../llama.cpp/ \
  --quantize-types Q4_K_M Q5_K_M Q6_K
```

---

## 8. Environnement

- Python 3.12 (virtualenv `llenv/`)
- ROCm 6.2 / iGPU AMD GFX900
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- PyTorch 2.5.1+rocm6.2

---

## 9. Problèmes connus (analye_codex.md)

| Sévérité | Problème |
|----------|---------|
| Élevé | `eval_strategy` vs `evaluation_strategy` (TypeError selon version Transformers) |
| Moyen | `--device` ignoré (toujours basé sur `torch.cuda.is_available()`) |
| Moyen | `--packing` impossible à désactiver (`store_true` + `default=True`) |
| Moyen | Variables d'environnement allocateur définies APRÈS `import torch` |
| Faible | `save_steps=None` invalide selon version Transformers |

---

## 10. Conclusion

Le modèle **`test_wikipedia_gguf/model-Q6_K.gguf`** est la version la plus aboutie :
- Fine-tuning 3 epochs sur Wikipedia FR 1000 chunks
- PPL 1.30 sur données synthétiques (meilleur résultat mesuré)
- Actif en production sur port 8989

Le prompt d'entraînement doit être utilisé **à l'identique** — toute modification dégrade les résultats car le modèle a été conditionné sur ce format exact.
