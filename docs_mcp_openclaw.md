# Secretarius MCP server (stubs)

Ce serveur MCP expose 3 outils `stub` pour l'integration initiale avec OpenClaw via `openclaw_mcp_adapter`, et reste reutilisable depuis tout client MCP compatible.

## Outils exposes

1. `extract_expressions` (chunking + appel llama.cpp)
- Entree: `text` (string) ou `document` (objet)
- Chunking: `SemanticChunker` embarque dans `secretarius/vendor/chunk_data.py`
- Prompt: `secretarius/prompts/prompt.txt`
- Backend LLM: `llama_cpp` via endpoint chat completions
- Option: `llama_url`, `model`, `timeout_s`, `max_tokens`, `prompt_path`
- En cas d'echec: `expressions = []` (aucune heuristique)

2. `expressions_to_embeddings`
- Entree: `expressions` (liste de strings)
- Backend: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingue)
- Dimension: `384`
- Parametres: `model` (optionnel), `normalize` (bool), `batch_size` (int)
- Sortie: `embeddings`, `embedding_count`, `dimension`, `model`, `warning`

3. `semantic_graph_search`
- Entree: `embeddings` (liste de vecteurs), `documents` optionnels, `top_k`
- Backend: Milvus (`milvus_uri`, `collection_name`, `metric_type`, `milvus_token` optionnels)
- Flux unifie: si `documents` est fourni, insertion + recherche dans le meme appel
- Sortie: `graph`, `hits`, `inserted_count`, `query_count`, `warning`

## Lancement

```bash
cd /home/mauceric/Secretarius
source .venv/bin/activate
python /home/mauceric/Secretarius/run_secretarius_mcp.py
```

Transport: `stdio` (JSON-RPC/MCP).

## Exemple integration OpenClaw (`openclaw_mcp_adapter`)

A adapter selon le format exact de configuration OpenClaw dans votre environnement:

```json
{
  "mcpServers": {
    "secretarius": {
      "command": "/home/mauceric/Secretarius/.venv/bin/python",
      "args": ["-u", "/home/mauceric/Secretarius/run_secretarius_mcp.py"],
      "cwd": "/home/mauceric/Secretarius",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Notes

- Cette version est volontairement une base de contrat, sans logique metier.
- Les schemas d'entree sont deja poses pour guider le remplissage progressif des outils.
