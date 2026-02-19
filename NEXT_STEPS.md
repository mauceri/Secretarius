# Secretarius - Document de travail (19 fevrier 2026)

## 1) Ce qui est en place depuis le dernier commit

### Serveur MCP
- Serveur MCP fonctionnel via `run_secretarius_mcp.py` + `secretarius/mcp_server.py`.
- Compatible OpenClaw (`openclaw_mcp_adapter`) en transport `stdio`.
- Outils exposes:
  1. `extract_expressions`
  2. `expressions_to_embeddings`
  3. `semantic_graph_search`

### Outil 1 - Extraction d'expressions
- Chunking semantique actif (vendor local: `secretarius/vendor/chunk_data.py`).
- Prompt local interne au repo:
  - `secretarius/prompts/prompt.txt`
- Appel LLM via `llama.cpp` (`/v1/chat/completions`).
- Filtre verbatim applique:
  - une expression est conservee seulement si elle apparait telle quelle dans le texte.
- Sortie enrichie:
  - `chunks`
  - `by_chunk` (expressions par chunk)
  - `expressions` (dedup)
  - `request_fingerprint`
  - `inference_params`
  - debug optionnel (`raw_llm_outputs`)

### Outil 2 - Plongements
- `expressions_to_embeddings` implemente (plus un stub).
- Modele par defaut:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Espace latent:
  - dimension `384`
- Parametres exposes:
  - `model` (optionnel)
  - `normalize` (bool)
  - `batch_size` (int)

### Outil 3 - Graphe semantique sur base
- `semantic_graph_search` implemente sur Milvus (plus un stub).
- Flux unifie:
  - option d'insertion (documents fournis) + recherche dans le meme appel.
- Retour:
  - `graph.nodes`, `graph.edges`
  - `hits`
  - `inserted_count`, `query_count`
  - `warning`
- Compatibilite Milvus 2.2:
  - `COSINE` mappe vers `IP` (avec embeddings normalises).

### Infra Milvus
- Stack docker compose prete dans `infra/milvus`.
- Validation realisee:
  - demarrage OK
  - healthcheck OK
  - test insertion+recherche OK via `semantic_graph_search`.

## 2) Orientation produit (idees valides)

### Tout est "note"
Unite documentaire unique: la note.
- une note peut etre brute, lecture, synthese, capture, brouillon, etc.
- les chunks sont une vue technique de la note.
- les expressions sont une vue semantique de granularite fine.

### Switch de persistance
Pour `semantic_graph_search`, garder un commutateur simple:
- `upsert=false`: requete seule
- `upsert=true`: insertion + requete

## 3) Proposition de structure de metadonnees (bio-friendly)

Objectif: garder des champs humains comprehensibles, sans perdre la tracabilite technique.

### Niveau note (`source_metadata`)
```json
{
  "source_id": "note:2026-02-19:001",
  "title": "Titre humain",
  "note_type": "lecture",
  "status": "draft",
  "importance": 3,
  "confidentiality": "internal",
  "themes": ["histoire", "poesie"],
  "keywords": ["mort", "seigneurie"],
  "language": "fr",
  "author": "mauceric",
  "created_at": "2026-02-19T15:00:00Z",
  "updated_at": "2026-02-19T15:00:00Z"
}
```

### Niveau chunk (`chunk_metadata`)
```json
{
  "chunk_id": "chunk:sha256:...",
  "source_id": "note:2026-02-19:001",
  "chunk_index": 0,
  "char_start": 0,
  "char_end": 420,
  "status": "draft",
  "confidentiality": "internal",
  "themes": ["histoire", "poesie"]
}
```

### Niveau expression (`expression_metadata`)
```json
{
  "expression_id": "expr:sha256:...",
  "source_id": "note:2026-02-19:001",
  "chunk_id": "chunk:sha256:...",
  "expression_text": "chambre aux deniers",
  "origin": "auto",
  "human_score": null
}
```

## 4) Orchestration cible des 3 outils

1. `extract_expressions(text, source_metadata)`  
Sorties:
- chunks
- expressions par chunk
- liste globale dedup

2. `expressions_to_embeddings(expressions, metadata_refs)`  
Sorties:
- embeddings alignes expression par expression

3. `semantic_graph_search(embeddings, documents, upsert, filters)`  
Sorties:
- hits
- graphe de similarite ponderee

## 5) Prochaines actions recommandees

1. Ajouter `source_id` en entree de `extract_expressions` et le propager partout.
2. Ajouter `upsert` et `filters` dans `semantic_graph_search`.
3. Formaliser un schema JSON versionne (`contract_version`).
4. Ajouter tests d'integration e2e (extract -> embed -> graph).
