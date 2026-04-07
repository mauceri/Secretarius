# CLAUDE.md — Projet Secretarius

## Architecture

Agent documentaire personnel avec pipeline NLP local.

### Deux backends LLM (intentionnels, ne pas modifier)

| Backend | URL | Modèle | Rôle |
|---------|-----|--------|------|
| llama.cpp server | `127.0.0.1:8989` (alias `sanroque:8989`) | Phi-4-mini fine-tuné Wikipedia FR | `expression_extractor.py` |
| Ollama | `localhost:11434` | Qwen | `llm_ollama.py` (Chef d'Orchestre) |

### Composants principaux (dans `Prototype/secretarius_local/`)

- `expression_extractor.py` — extraction d'expressions via Phi-4/llama.cpp
- `embeddings.py` — BGE-M3 (BAAI/bge-m3), 1024 dim, L2-normalisé
- `semantic_graph.py` — Milvus 2.2 (COSINE → IP avec vecteurs normalisés), 1 row/expression
- `document_pipeline.py` — pipeline complet : chunking → expressions → embeddings → Milvus
- `chef_orchestre.py` — cycle ReAct, max 2 outils/cycle, recovery JSON
- `mcp_server.py` — 6 outils MCP exposés

### Répertoires lora (hors Prototype/)

- `~/lora_local/` — artefacts entraînement (NE PAS déplacer sans confirmation)
  - `test_wikipedia_gguf/model-Q6_K.gguf` — modèle actif dans llama.cpp service
  - `gguf_out/phi4_merged/` — variantes Q4/Q5/Q6
  - `checkpoints_phi4_lora/` — adaptateur LoRA (précieux, ne pas supprimer)
- `~/lora_slm/` — scripts pipeline LoRA

## Plan d'action v1 (voir `Plan_Action.md` pour détails)

| # | Phase | Statut |
|---|-------|--------|
| 1 | Late Interaction ColBERT (`aggregate_late_interaction`) | **En cours** |
| 2 | Consolidation tests | À faire |
| 3 | Fusion répertoires | À faire |
| 4 | Déploiement systemd + Milvus Docker | À faire |
| 5 | Documentation | À faire |

## Phase 1 — Late Interaction (priorité actuelle)

Formule : `score(doc) = Σᵢ maxⱼ cosine(qᵢ, dⱼ)`

Fichiers à modifier :
1. `Prototype/secretarius_local/semantic_graph.py` — ajouter `aggregate_late_interaction(hits, query_count)`
2. `Prototype/secretarius_local/document_pipeline.py` — réécrire `search_documents_by_text()` : extraction expressions → BGE-M3 → multi-vecteurs Milvus → agrégation MaxSim
3. `Prototype/secretarius_local/mcp_server.py` — brancher `_handle_search_text()`

Ce qui ne change pas : `semantic_graph_search_milvus()` (accepte déjà N vecteurs), `embeddings.py`, `expression_extractor.py`, structure Milvus.

## Commandes bash pré-approuvées (Prototype/)

- `python -m pytest tests/ *` — lancer les tests
- `./run_tests.sh` — suite complète
- `make test`, `make test-unit`, `make test-integration` — via Makefile
- `python -c "import *"` — vérifications rapides d'imports
- `git *` — toutes opérations git (sauf push --force)

## Règles importantes

- Ne jamais proposer de charger un GGUF dans Ollama (llama.cpp server est le backend, pas Ollama)
- Ne jamais modifier `llm_ollama.py` pour pointer vers llama.cpp (deux backends distincts intentionnels)
- Déploiement cible : systemd (pas Docker pour l'application, Docker uniquement pour Milvus)
- Confirmation requise avant : `systemctl start/stop/enable`, `docker compose up/down`, `git push`
