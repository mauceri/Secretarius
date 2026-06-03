# Harnais de routage par intention

Évalue, hors OpenClaw, la capacité à router un message vers le bon agent
(wikilm, gog, superpowers, clarify). Compare deux routeurs sur le même corpus.

## Prérequis
- venv : `../.venv` (sentence_transformers, numpy, openai, pytest)
- Routeur LLM : service `slm-llama-cpp` (Phi-4-mini) sur http://127.0.0.1:8998
- Génération de corpus : `DEEPSEEK_API_KEY` dans l'environnement

## Évaluer

    cd ~/Secretarius/Wiki_LM/routing
    ../.venv/bin/python eval_routing.py --router embed     # routeur embeddings BGE-M3
    ../.venv/bin/python eval_routing.py --router llm       # routeur Phi-4-mini local

## Enrichir le corpus (par agent)

    ../.venv/bin/python corpus_gen.py --agent gog --n 20   # génère candidates_gog.jsonl
    # relire/éditer candidates_gog.jsonl
    ../.venv/bin/python corpus_gen.py --commit --agent gog # ajoute au corpus.jsonl

## Tests

    ../.venv/bin/python -m pytest tests/ -v

## Fichiers
- `agents.json` — catalogue des agents
- `corpus.jsonl` — corpus étiqueté (= futur dataset LoRA)
- `router_embed.py` / `router_llm.py` — les deux routeurs
- `eval_routing.py` — évaluation (exactitude, matrice, erreurs)
- `corpus_gen.py` — génération assistée few-shot
