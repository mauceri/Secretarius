# Continuation - Secretarius Prototype

## Contexte
Objectif: rendre `Prototype` autonome, avec serveur MCP fonctionnel et outils intégrés localement (sans dépendance code vers `../secretarius`).

## Ce qui a été fait
- Correction du client MCP:
  - `adapters/output/mcp_client.py`
  - API stdio MCP corrigée (`read_stream`, `write_stream`) + `cwd` explicite.
- Durcissement orchestration:
  - `core/chef_orchestre.py`
  - prompt JSON valide, parsing plus robuste, verrou async anti-course, trimming mémoire.
- Durcissement LLM/config:
  - `adapters/output/llm_ollama.py` (timeout HTTP)
  - `main.py` (fallback config, `sys.executable`, `cwd`)
- Serveur MCP local:
  - `tools/oracle_server.py`
  - expose `ask_oracle` + délègue aux outils locaux vendorisés.
- Intégration locale des outils MCP (copiés dans `Prototype`):
  - `secretarius_local/mcp_server.py`
  - `secretarius_local/expression_extractor.py`
  - `secretarius_local/embeddings.py`
  - `secretarius_local/semantic_graph.py`
  - `secretarius_local/document_schema.py`
  - `secretarius_local/prompts/prompt.txt`
  - `secretarius_local/vendor/chunk_data.py`
  - `secretarius_local/__init__.py`

## Tests ajoutés
- `tests/test_mcp_client.py` (smoke MCP connect/list/call/disconnect)
- `tests/test_chef_orchestre.py` (parsing, validation ReAct, trimming)
- `run_tests.sh`
- `Makefile` (`make test`)

## Validation
Commande:
```bash
make test
```
Résultat actuel: 4 tests OK.

Vérification d’autonomie code:
```bash
rg -n "../secretarius|from secretarius import|import secretarius" . -g "*.py"
```
Résultat attendu: aucun match.

## Points d’attention
- Les outils `expressions_to_embeddings` / `semantic_graph_search` dépendent toujours de libs/runtime externes (`sentence-transformers`, `pymilvus`, modèles locaux), mais plus de code importé depuis `../secretarius`.
- Un appel à `expressions_to_embeddings` peut charger le modèle et afficher beaucoup de logs (normal).

## Prochaines étapes proposées
1. Ajouter un test explicite qui vérifie la présence des 4 outils dans `tools/list`.
2. Ajouter un mode de logs configurable (fichier) pour le serveur MCP local.
3. Ajouter une doc `README` dans `Prototype` pour lancer le TUI + prérequis runtime (ollama, embeddings, milvus).

---

## Mise à jour du 2026-03-03

### Avancement du jour
- Passage du modèle par défaut à `qwen3:0.6b`.
- Ajout des logs persistants:
  - `logs/guichet.log` (chat + pensées)
  - `logs/llm_raw.log` (sortie brute LLM)
- Correctifs TUI/Textual:
  - échappement du texte affiché (plus de disparition liée au markup),
  - correction du dispatch UI (`call_from_thread` / thread UI).
- Durcissement ReAct (`core/chef_orchestre.py`):
  - meilleure tolérance aux réponses JSON imparfaites de petits modèles,
  - garde-fous anti-répétition d’appel outil,
  - récupération progressive quand `action` et `final_answer` sont fournis ensemble.
- Désactivation du mode thinking Qwen côté Ollama (`think: false`).
- Architecture canaux:
  - refactor vers **plusieurs canaux, un seul guichet** via `adapters/input/guichet_unique.py`,
  - TUI converti en canal (`TUIChannel`),
  - OpenWebUI branché comme canal OpenAI-compatible (`openwebui_api.py`),
  - runtime partagé (`app_runtime.py`).
- Nouveau lancement mono-process multi-canaux:
  - `main_multicanal.py` (TUI + OpenWebUI),
  - arrêt plus propre des composants,
  - fallback automatique de port OpenWebUI (`port`, `port+1`, ...).

### État actuel
- Les 4 outils MCP sont bien vus au runtime:
  - `ask_oracle`
  - `extract_expressions`
  - `expressions_to_embeddings`
  - `semantic_graph_search`
- `make test` reste au vert (4 tests).
- Le système fonctionne, mais le comportement de Qwen reste parfois instable sur certaines formulations (réponses contradictoires / refus non pertinents).

### Problèmes encore ouverts
- En mode OpenWebUI, l’envoi de tout l’historique peut perturber l’agent (ancrage sur un contexte précédent).
- Malgré les garde-fous, certaines sorties restent incohérentes sémantiquement (bonne tool-selection puis mauvaise finalisation verbale).

### Priorités pour la prochaine session
1. Ajouter une stratégie explicite de gestion d’historique par canal OpenWebUI (par défaut: dernier message utilisateur, option configurable).
2. Ajouter des tests d’intégration ReAct pour scénarios “action+final_answer” répétés avec Qwen.
3. Ajouter un test de non-régression sur la sélection d’outil (`extract_expressions` vs `ask_oracle`) selon l’intention utilisateur.
