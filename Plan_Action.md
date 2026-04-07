# Plan d'action — Secretarius v1 opérationnelle

## État des lieux (23 mars 2026)

Le projet se compose de deux parties distinctes à consolider :

- **`Prototype/`** : application principale (architecture hexagonale) — bien avancée (~80-90%)
  - Chef d'Orchestre (ReAct) : fonctionnel
  - Guichet multi-canaux (TUI, OpenWebUI, Session Messenger, Memos) : fonctionnel
  - Serveur MCP + 5 outils : fonctionnel
  - Pipeline d'indexation (extraction expressions + embeddings BGE-M3 + Milvus) : fonctionnel
  - LLM configuré : `qwen3.5:2B` (à remplacer par Phi-4 fine-tuné, voir ci-dessous)
  - **Manque** : recherche late interaction (ColBERT) — seule une recherche vectorielle simple existe
- **`/home/mauceric/lora_local/`** : fine-tuning Phi-4-mini-instruct — **déjà accompli**
  - Adaptateur LoRA entraîné : `checkpoints_phi4_lora/` (checkpoints 306/612/918, version précieuse sauvegardée)
  - Modèle fusionné et quantifié : `gguf_out/phi4_merged/` — `model-Q4_K_M.gguf`, `model-Q5_K_M.gguf`, `model-Q6_K.gguf`
  - **Action immédiate** : charger le GGUF dans Ollama et mettre à jour `config.yaml`
- **`/home/mauceric/lora_slm/`** : réorganisation/nettoyage du pipeline lora_local (code, corpus, scripts)
  - Corpus synthétiques prêts (500/1000/4000 exemples) + Wikipedia + Gutenberg
  - Répertoires `checkpoints/` et `models/` vides (artefacts dans `lora_local/`, pas ici)

### Infrastructure de déploiement (pattern existant)

Les services tournent déjà via systemd sur cette machine :
- **llama.cpp server** (`llama-server`) : service systemd sur port **8989**, hostname `sanroque`
  - Modèle actif : `lora_local/test_wikipedia_gguf/model-Q6_K.gguf` — extracteur Phi-4 fine-tuné sur `corpus_fr` (Wikipedia FR), opérationnel
  - Autre variante : `/home/mauceric/Modèles/extracteur-phi-4-Q6_K.gguf`
  - Usage documenté dans `lora_local/test_model.ipynb` : chunking SentenceTransformer → POST `/v1/chat/completions` → filtrage verbatim
  - Config : `-ngl 32` (GPU AMD iGPU), `-c 20480` (contexte large), `--jinja`
- `session_bot.service` : bot Session Messenger — pattern systemd à suivre pour Secretarius

**Le déploiement cible est systemd, pas Docker** (sauf pour Milvus). Pas d'Ollama.

Deux backends LLM coexistent intentionnellement : Ollama/Qwen pour le routeur, llama.cpp server/Phi-4 pour l'extraction.

---

## Phase 1 — Recherche Late Interaction (ColBERT)

**Objectif** : compléter `search_text` pour exploiter la structure existante d'embeddings par expression.

### Pourquoi c'est déjà à moitié fait

L'architecture stocke déjà **1 embedding par expression** dans Milvus (`semantic_graph.py`, `_build_row()`). La fonction `semantic_graph_search_milvus()` accepte déjà une **liste** de vecteurs (`data=normalized`). Il manque uniquement l'agrégation MaxSim par document côté `search_text`.

### Principe

```
score(doc) = sum_i( max_j( cosine(q_i, d_j) ) )
```
où `q_i` = embeddings des expressions de la requête, `d_j` = embeddings des expressions du document `doc`.

### Ce que fait `search_text` actuellement

`search_text(query)` → encode la requête en **un seul vecteur** → 1 recherche Milvus → top_k résultats.

### Ce qu'il faut ajouter dans `document_pipeline.py` : `search_documents_by_text()`

1. **Extraire les expressions de la requête** via `extract_expressions()` (appel llama.cpp `127.0.0.1:8989`)
2. **Encoder chaque expression** via `embed_expressions_multilingual()` (BGE-M3, 1024 dim, normalisé)
3. **Passer tous les vecteurs d'un coup** à `semantic_graph_search_milvus()` — déjà supporté
4. **Agréger par `doc_id`** : pour chaque document, score = somme des MaxSim sur les vecteurs-requête
5. **Trier et retourner** les top_k documents

### Fichiers à modifier

| Fichier | Modification | Détail |
|---------|-------------|--------|
| `secretarius_local/document_pipeline.py` | `search_documents_by_text()` | Ajouter extraction + agrégation MaxSim |
| `secretarius_local/semantic_graph.py` | Ajouter `aggregate_late_interaction()` | Fonction pure : `hits[]` × `doc_id` → scores agrégés |
| `secretarius_local/mcp_server.py` | `_handle_search_text()` | Brancher sur la nouvelle logique |

### Ce qui ne change pas

- `semantic_graph_search_milvus()` : inchangée, accepte déjà N vecteurs
- `embeddings.py` : inchangé, BGE-M3 1024 dim normalisé
- `expression_extractor.py` : inchangé, déjà utilisé pour les requêtes
- Structure Milvus : inchangée, 1 row par expression

### Tests

Étendre `tests/test_semantic_graph.py` avec :
- `test_aggregate_late_interaction_sums_maxsim_per_doc` — logique d'agrégation pure (sans Milvus)
- `tests/test_document_pipeline.py` : `test_search_uses_late_interaction_scoring` — mock `semantic_graph_search_milvus`

---

## Phase 2 — Fusion des répertoires

**Objectif** : un seul repo cohérent sous `Secretarius/`.

### Structure cible

```
Secretarius/
├── Prototype/           # Application principale (inchangé)
├── lora_local/          # Artefacts entraînement (déplacé depuis /home/mauceric/lora_local/)
│   ├── checkpoints_phi4_lora/     # Adaptateur LoRA entraîné
│   ├── gguf_out/phi4_merged/      # GGUF prêts à l'emploi
│   └── models/phi4                # Modèle de base
├── lora_slm/            # Scripts pipeline (déplacé depuis /home/mauceric/lora_slm/)
│   ├── corpus/          # Génération corpus DSPy/GEPA
│   ├── src/             # Entraînement, fusion, évaluation
│   └── data/            # Corpus JSONL
├── Apprentissage/       # Exemples GEPA, Module LoRA (déjà présent)
├── deploy/              # Services systemd + config Milvus
├── infra/               # Config Milvus (déjà présent)
└── docs/                # Documentation (Phase 5)
```

### Étapes

1. Déplacer `lora_local/` → `Secretarius/lora_local/`
2. Déplacer `lora_slm/` → `Secretarius/lora_slm/`
3. Mettre à jour le `Modelfile` Ollama avec le nouveau chemin absolu du GGUF
4. Vérifier cohérence avec `Apprentissage/` (prompts, exemples GEPA partagés)
5. Mettre à jour `README.md` avec la structure complète
6. Un seul `git init` / `git remote` pour l'ensemble du projet

---

## Phase 3 — Consolidation des tests

**Objectif** : `pytest tests/ -m unit` passe sans services externes.

### État actuel (lu dans le code)

- 15 fichiers de tests dans `Prototype/tests/`
- Tests existants couvrent déjà : filtrage par score, upsert/suppression Milvus, IDs stables, filtres keywords, parsing doc_id/type_note, cycle ReAct JSON recovery, blocage oracle
- `pytest.ini` présent mais markers non déclarés
- Mocks Milvus utilisent `MagicMock` mais pas toujours cohérents entre fichiers

### Stratégie de mock

| Service | Approche |
|---------|----------|
| Milvus (`MilvusClient`) | `unittest.mock.patch` sur `semantic_graph.py` — déjà fait partiellement |
| llama.cpp server (`httpx`/`aiohttp`) | Mock `expression_extractor._call_llama_cpp()` avec réponses JSON fixes |
| Ollama (`OllamaAdapter`) | Mock `generate_response()` avec JSON ReAct fixes |
| MCP stdio | Transport stdio avec `mcp_server` lancé en subprocess de test |

### Tests prioritaires à compléter

- `test_semantic_graph.py` — ajouter `test_aggregate_late_interaction_sums_maxsim_per_doc`
- `test_document_pipeline.py` — ajouter `test_search_uses_late_interaction_scoring`
- `test_chef_orchestre.py` — compléter le cycle complet avec vrai MCP mock
- `test_mcp_client.py` — vérifier les 6 outils exposés

### Configuration pytest

```ini
# pytest.ini à compléter
[pytest]
markers =
    unit: tests sans services externes
    integration: tests nécessitant Milvus et Ollama
    slow: tests longs (entraînement, embeddings réels)
```

Étendre le `Makefile` :
```makefile
test-unit:
    pytest tests/ -m unit -v

test-integration:
    pytest tests/ -m integration -v
```

---

## Phase 4 — Déploiement systemd + Milvus Docker

**Objectif** : démarrage fiable et automatique de tous les services au boot.

### Architecture de déploiement

Le pattern existant sur la machine (Ollama en systemd, session_bot en systemd) est le bon modèle :

| Service | Mode | Fichier |
|---------|------|---------|
| Ollama | systemd | `ollama.service` (existant) |
| Milvus | Docker Compose | `infra/` (existant à compléter) |
| Secretarius | systemd | `deploy/secretarius.service` (à créer) |
| Session bot | systemd | `session_bot.service` (existant, référence) |

### Étapes

1. Finaliser `deploy/secretarius.service` (voir tâches transversales)
2. Vérifier/compléter le `docker-compose.yml` Milvus dans `infra/` (etcd + minio + milvus-standalone)
3. Script `deploy/start.sh` : démarrer Milvus (docker compose up -d) puis activer le service systemd
4. Script `deploy/stop.sh` : arrêt propre
5. Documenter la configuration Tailscale pour accès distant chiffré

---

## Phase 5 — Squelette de documentation

**Objectif** : documentation minimale pour rendre le projet appropriable.

### Structure

```
Secretarius/
├── README.md                  # Vue d'ensemble + installation rapide (5 commandes)
└── docs/
    ├── architecture.md        # Architecture hexagonale, explication des composants
    ├── installation.md        # Prérequis (Ollama, Milvus, Python), pas-à-pas
    ├── configuration.md       # config.yaml commenté section par section
    ├── canaux.md              # Guide par canal (TUI, OpenWebUI, Telegram, Session Messenger)
    ├── indexation.md          # Pipeline : chunking sémantique → expressions → embeddings → Milvus
    ├── lora.md                # Fine-tuning Phi-4-mini : corpus → entraînement → GGUF → Ollama
    └── zettelkasten.md        # Vision à terme : usage Zettelkasten avec Secretarius
```

---

## Tâches transversales

### Deux backends LLM distincts — aucune modification nécessaire

L'architecture utilise deux backends selon les rôles, ce qui est intentionnel :

| Backend | URL | Modèle | Rôle |
|---------|-----|--------|------|
| Ollama | `localhost:11434` | Qwen | Chef d'Orchestre / routeur (`llm_ollama.py`) |
| llama.cpp server | `sanroque:8989` | Phi-4 fine-tuné (Wikipedia FR) | Extraction d'expressions (`expression_extractor.py`) |

Les deux sont opérationnels. Rien à modifier côté LLM pour la v1.

### Service systemd pour Secretarius (pattern : session_bot.service)

Créer `deploy/secretarius.service` sur le modèle de `session_bot.service` :

```ini
[Unit]
Description=Secretarius Agent Service
After=network.target

[Service]
Type=simple
User=mauceric
WorkingDirectory=/home/mauceric/Secretarius/Prototype
ExecStart=/home/mauceric/Secretarius/Prototype/.venv/bin/python server_secretarius.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Résilience Milvus

Ajouter retry/backoff dans `Prototype/secretarius_local/semantic_graph.py` pour gérer les démarrages lents ou redémarrages de Milvus.

---

## Ordre de priorité

| # | Tâche | Dépendances | Complexité |
|---|-------|-------------|------------|
| 1 | Late Interaction ColBERT | — | Moyenne |
| 2 | Consolidation tests | Phase 1 | Moyenne |
| 3 | Fusion répertoires | Tests stables | Faible |
| 4 | Déploiement systemd + Milvus Docker | — | Faible |
| 5 | Documentation | Phases précédentes | Faible |

---

## Critères de succès de la v1

- [ ] Les deux backends LLM (Ollama/Qwen pour le routeur, llama.cpp/Phi-4 pour l'extracteur) fonctionnent ensemble dans un test de bout en bout
- [ ] `pytest tests/ -m unit` passe en totalité sans services externes
- [ ] Une recherche avec late interaction retourne de meilleurs résultats qu'une recherche vectorielle simple sur un corpus de test
- [ ] `deploy/start.sh` lance tous les services (Milvus + Secretarius) en moins de 2 minutes
- [ ] Secretarius redémarre automatiquement au boot via systemd
- [ ] Un nouveau collaborateur peut installer et lancer Secretarius en suivant `docs/installation.md` uniquement
