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

---

## Mise à jour du 2026-03-04

### Changements effectués
- Observabilité LLM renforcée (`adapters/output/llm_ollama.py`):
  - `logs/llm_raw.log` contient maintenant:
    - `STREAM REQUEST` (payload exact envoyé à Ollama),
    - `STREAM RESPONSE_RAW` (JSON brut reçu),
    - `STREAM RESPONSE_CONTENT` (contenu extrait).
- Robustesse parsing ReAct (`core/chef_orchestre.py`):
  - normalisation de `final_answer` non-string (JSON -> string),
  - garde-fou `action_input` non-dict.
- Politique oracle (`core/chef_orchestre.py`):
  - `ask_oracle` est traité comme outil terminal: retour direct de l’observation outil sans nouveau tour LLM.
- Blindage des arguments outils (`core/chef_orchestre.py`):
  - ajout d’un filtrage/whitelist des `action_input` par outil,
  - suppression des clés halluciné(e)s (ex: `llama_cpp_url`, `llama_url` injectés par le modèle).
- OpenWebUI (`openwebui_api.py`):
  - extraction `Query:` quand OpenWebUI envoie `History: ... Query: ...`,
  - filtrage des tâches auxiliaires OpenWebUI (follow_ups / title / tags) pour éviter la pollution du flux agent.
- TUI:
  - traces visibles réactivées par défaut (`ui.show_thoughts: true`).
- Modèle:
  - essai de `qwen3:8B`, puis retour à `qwen3:0.6b` (latence trop élevée).

### Tests / validation
- `make test` OK (4/4) après mise à jour du test orchestration:
  - adaptation du test `ask_oracle` terminal dans `tests/test_chef_orchestre.py`.

### Incidents observés
- Erreurs `unknown url type: 'llama.cpp'` sur `extract_expressions`:
  - cause: arguments outil halluciné(e)s par le LLM (`llama_cpp_url: "llama.cpp"`),
  - atténuation: filtrage strict des arguments côté orchestrateur.
- Logs Ollama avec `aborting completion request due to client closing the connection`:
  - cause observée: timeout client/cancellation côté application, pas arrêt volontaire d’Ollama.

### Problème à traiter en priorité demain
- Le cycle ReAct relance encore parfois le LLM alors qu’une réponse exploitable est déjà disponible.
- Action attendue: simplifier la boucle (moins de retries, policy plus déterministe) pour réduire coût et latence.

### Piste R&D (à cadrer)
- Étudier un adaptateur LoRA léger pour contraindre `qwen3:0.6b` au contrat JSON/outils:
  - dataset ciblé: sorties valides `action/action_input/final_answer`, cas négatifs (clés interdites, formats invalides),
  - objectif: réduire hallucinations d’arguments outil et sorties hors schéma,
  - livrable minimal: protocole d’évaluation (taux de JSON valide, taux d’appels outil valides, coût latence).

---

## Mise à jour du 2026-03-05

### Changements effectués
- OpenWebUI port:
  - suppression du fallback automatique de port (plus de bascule 8001),
  - démarrage strict sur le port configuré.
- Orchestrateur en mode routeur:
  - Qwen utilisé pour la sélection d’outil uniquement,
  - suppression de la reformulation post-outil par Qwen (retour direct de l’observation outil),
  - fallback standard si aucun outil valide: `Aucun outil ne correspond a votre demande.`,
  - limite `MAX_TOOL_CALLS_PER_TURN = 2`.
- Robustesse d’entrées outil:
  - filtrage strict des arguments par outil conservé,
  - récupération automatique pour `extract_expressions` quand le modèle n’envoie pas de `text` exploitable,
  - nettoyage du préfixe d’instruction utilisateur avant extraction (payload hygiene).
- Déduplication des requêtes:
  - déduplication concurrente in-flight par `(channel, texte_normalise)`,
  - déduplication post-réponse avec cache court (TTL 15s) pour absorber les doubles soumissions rapprochées.
- Observabilité OpenWebUI:
  - ajout d’un `request_id` par requête,
  - fingerprint de prompt,
  - marquage `retry_suspect` sur fenêtre courte,
  - propagation de `X-Request-Id`.
- Catalogue outils MCP revu (orientation métier):
  - conservation de `extract_expressions` (analyse),
  - conservation de `semantic_graph_search`,
  - ajout de `index_text` (extraction + insertion Milvus),
  - ajout de `search_text` (requête textuelle -> recherche Milvus),
  - masquage de `expressions_to_embeddings` du `tools/list` public (reste interne).

### Tests / validation
- Nouveaux tests:
  - `tests/test_guichet_unique.py` (dédup concurrente + dédup post-réponse),
  - `tests/test_mcp_tools_catalog.py` (contrat public des outils MCP),
  - compléments dans `tests/test_chef_orchestre.py` (router mode, sanitation, prefix stripping).
- Exécution locale validée:
  - `python -m unittest -q tests/test_chef_orchestre.py tests/test_guichet_unique.py tests/test_mcp_tools_catalog.py`
  - résultat: OK.

### Décisions produit entérinées
- `extract_expressions` reste exposé pour usage d’analyse.
- `expressions_to_embeddings` n’est plus un outil final utilisateur.
