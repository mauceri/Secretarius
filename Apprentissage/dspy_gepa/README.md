# dspy_gepa

Chantier dedie a la generation de corpus synthetique avec DSPy/GEPA.

- `schemas/` : schema JSON cible partage par les scripts et notebooks
- `prompts/` : prompts de generation, critique, filtrage
- `notebooks/` : prototypage et experimentation
- `scripts/` : pipelines reproductibles
- `datasets/source/` : donnees d'entree
- `datasets/generated/` : corpus synthetiques produits
- `artefacts/runs/` : sorties de runs, traces et configurations

Base minimale ajoutee pour un premier pipeline DSPy sans GEPA :

- `schemas/synthetic_record.schema.json` : format cible des exemples generes
- `datasets/source/manual_seed_examples.jsonl` : petites graines manuelles de depart
- `prompts/baseline_generation.md` : contrat de generation lisible
- `scripts/generate_baseline.py` : generation DSPy baseline
- `scripts/validate_generated.py` : validation locale des sorties JSONL
- `notebooks/01_baseline_generation.ipynb` : notebook de demarrage reproductible
