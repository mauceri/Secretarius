# lora_slm — Extraction d'expressions clés par LoRA

Fine-tuning d'un adaptateur LoRA sur n'importe quel modèle causal (Phi-4-mini,
Mistral, Qwen, Llama…) pour extraire les expressions caractéristiques d'un texte,
en vue d'indexer des notes et articles à la manière Zettelkasten.

## Pipeline complet

```
corpus/optimize_prompt.py     ← optimise le prompt via DSPy.GEPA + DeepSeek
corpus/generate.py            ← génère le corpus synthétique (format chunks, direct)
         ↓
   data/corpus_synth.jsonl
         ↓
src/split_corpus.py           ← découpe train / test
         ↓
src/train.py                  ← entraîne le LoRA
         ↓
src/merge.py                  ← fusionne + exporte GGUF
         ↓
src/evaluate.py               ← évalue loss / perplexité
src/jaccard.py                ← évalue score Jaccard (inférence GGUF)
```

## Structure

```
lora_slm/
├── corpus/
│   ├── common.py              # Chargeurs partagés + conversion → chunks
│   ├── generate.py            # Génération simple (prompt existant)
│   ├── optimize_prompt.py     # Optimisation GEPA + génération
│   ├── config/
│   │   ├── themes.json        # Liste de thèmes
│   │   ├── categories.jsonl   # Catégories de notes
│   │   ├── types_by_category.json  # Types de documents par catégorie
│   │   └── examples.json      # Exemples pour GEPA
│   └── prompts/
│       ├── prompt_init.txt    # Prompt initial
│       └── prompt_gepa.txt    # Meilleur prompt trouvé par GEPA
│
├── src/
│   ├── common.py              # Utilitaires partagés (prompts, dataset, collator)
│   ├── train.py               # Entraînement LoRA
│   ├── merge.py               # Fusion adaptateur + export GGUF
│   ├── evaluate.py            # Évaluation loss / perplexité (modèle HF)
│   ├── jaccard.py             # Évaluation score Jaccard (modèle GGUF)
│   ├── split_corpus.py        # Découpe train / test
│   ├── run_pipeline.py        # Pipeline : split → évaluation
│   └── convert_format.py      # Conversion format legacy → chunks
│
├── data/                      # Corpus JSONL (généré + train/test)
├── models/                    # Modèles de base (non versionnés)
├── checkpoints/               # Adaptateurs LoRA (non versionnés)
├── output/                    # Modèles fusionnés + GGUF (non versionnés)
├── logs/                      # Fichiers de log
├── metrics/                   # Résultats d'évaluation JSON
├── notebooks/                 # Notebooks Jupyter
├── docs/                      # Documentation complémentaire
└── archive/                   # Anciens scripts (référence)
```

## Environnements Python

Le projet utilise deux environnements séparés (dépendances incompatibles) :

| Répertoire | Environnement | Usage |
|---|---|---|
| `corpus/` | `corpus_env/` | DSPy, litellm, gepa, openai |
| `src/` | `llenv/` | torch, transformers, peft |

```bash
# Environnement corpus
python -m venv corpus_env
corpus_env/bin/pip install -r requirements-corpus.txt

# Environnement entraînement (ROCm)
python -m venv llenv
llenv/bin/pip install -r requirements-rocm.txt
llenv/bin/pip install -r requirements.txt
```

## 1. Génération du corpus

### Génération simple (prompt existant)

```bash
# Avec OpenAI
OPENAI_API_KEY=sk-... corpus_env/bin/python corpus/generate.py \
    --count 500 \
    --output data/corpus_synth.jsonl

# Avec DeepSeek (moins coûteux)
DEEPSEEK_API_KEY=sk-... corpus_env/bin/python corpus/generate.py \
    --provider deepseek --model deepseek-chat \
    --count 500 \
    --output data/corpus_synth.jsonl

# Avec un modèle local (serveur llama.cpp ou ollama)
corpus_env/bin/python corpus/generate.py \
    --api-base http://localhost:8080/v1 \
    --model Phi-4-mini-instruct-Q6_K.gguf \
    --count 100 --output data/corpus_local.jsonl
```

### Optimisation du prompt via GEPA puis génération

```bash
DEEPSEEK_API_KEY=sk-... corpus_env/bin/python corpus/optimize_prompt.py \
    --count 100 \
    --output data/corpus_gepa.jsonl \
    --gepa-prompt corpus/prompts/prompt_gepa.txt
```

Le meilleur prompt trouvé est sauvegardé dans `corpus/prompts/prompt_gepa.txt`
et peut ensuite être réutilisé :

```bash
DEEPSEEK_API_KEY=sk-... corpus_env/bin/python corpus/generate.py \
    --prompt corpus/prompts/prompt_gepa.txt \
    --count 2000 --output data/corpus_synth_v2.jsonl
```

## 2. Préparation des données

```bash
# Split train / test
llenv/bin/python src/split_corpus.py \
    --input data/corpus_synth.jsonl \
    --train data/train.jsonl \
    --test  data/test.jsonl \
    --test_ratio 0.1

# Conversion d'un corpus legacy (si besoin)
llenv/bin/python src/convert_format.py \
    --input old_corpus.jsonl --output data/train.jsonl
```

## 3. Entraînement

```bash
# Phi-4-mini (ROCm)
llenv/bin/python src/train.py \
    --model_path models/phi4-mini \
    --data_file  data/train.jsonl \
    --output_dir checkpoints/phi4-mini-lora-v1 \
    --max_len 512 --epochs 1 --bf16

# Mistral (tag assistant différent)
llenv/bin/python src/train.py \
    --model_path    models/mistral-7b \
    --data_file     data/train.jsonl \
    --output_dir    checkpoints/mistral-lora-v1 \
    --assistant_tag "<|im_start|>assistant" \
    --max_len 1024 --epochs 3
```

Paramètres clés :
- `--target_modules` : modules LoRA (défaut : `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`)
- `--lora_r` / `--lora_alpha` : rang / scaling LoRA (défaut : 16 / 32)
- `--packing` / `--no-packing` : constant-length packing (activé par défaut)
- `--log_file` : log des pertes (défaut : `logs/training.log`)

## 4. Fusion + quantization

```bash
# Fusion seule
llenv/bin/python src/merge.py \
    --base models/phi4-mini \
    --lora checkpoints/phi4-mini-lora-v1 \
    --out  output/phi4-mini_merged/merged_hf

# Fusion + GGUF
llenv/bin/python src/merge.py \
    --base           models/phi4-mini \
    --lora           checkpoints/phi4-mini-lora-v1 \
    --out            output/phi4-mini_merged/merged_hf \
    --gguf-dir       output/phi4-mini_merged/gguf \
    --llama-cpp      /chemin/vers/llama.cpp \
    --quantize-types Q4_K_M Q5_K_M Q6_K
```

## 5. Évaluation

```bash
# Loss / perplexité (modèle HF)
llenv/bin/python src/evaluate.py \
    --model       output/phi4-mini_merged/merged_hf \
    --data_file   data/test.jsonl \
    --metrics_out metrics/eval_v1.json

# Score Jaccard (modèle GGUF)
llenv/bin/python src/jaccard.py \
    --input data/test.jsonl \
    --model output/phi4-mini_merged/gguf/model-Q6_K.gguf

# Pipeline complet split → évaluation
llenv/bin/python src/run_pipeline.py \
    --input       data/corpus_synth.jsonl \
    --model       output/phi4-mini_merged/merged_hf \
    --metrics_out metrics/eval_v1.json
```

## Format de données (chunks)

Tous les scripts producteurs (corpus/) et consommateurs (src/) utilisent ce format :

```json
{
  "source": "synthetic",
  "titre": null,
  "chunks": [
    {"chunk": "Texte...", "expressions_caracteristiques": ["expr1", "expr2"]}
  ],
  "meta": {"theme": "...", "categorie": "...", "dataset": "synthetic", "lang": "fr"}
}
```

## Notes ROCm (AMD)

`src/common.py` positionne automatiquement :
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` (iGPU 680M / GFX1030)

Surcharger `HSA_OVERRIDE_GFX_VERSION` dans le shell si votre GPU est différent.
