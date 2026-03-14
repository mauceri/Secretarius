# ATTENTION - REGLE DE TRAVAIL OBLIGATOIRE
NE FAIRE AUCUNE MODIFICATION DE CODE, DE CONFIGURATION OU DE FICHIER SANS ACCORD EXPLICITE PREALABLE DE L'UTILISATEUR.
TOUTE PROPOSITION DE CHANGEMENT DOIT D'ABORD ETRE VALIDEE PAR L'UTILISATEUR AVANT EDITION.
POUR LES COMMANDES PYTHON SUR CE PROJET, UTILISER L'ENVIRONNEMENT VIRTUEL : /home/mauceric/Secretarius/.venv

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
  - `tools/secretarius_server.py`
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
- IMPORTANT (architecture): la logique de retry/dédup spécifique OpenWebUI doit rester dans le canal `openwebui_api.py` (transport), pas dans les outils MCP; l’orchestrateur ne garde qu’une protection générique minimale anti-boucle.

---

## Mise à jour du 2026-03-06

### Changements effectués
- Nettoyage API extraction:
  - suppression des wrappers legacy autour de `extract_expressions`,
  - simplification du chargement côté `secretarius_local/mcp_server.py` pour ne résoudre qu'une seule fonction publique d'extraction.
- Factorisation des chemins runtime:
  - ajout de `secretarius_local/runtime_paths.py`,
  - suppression des chemins locaux codés en dur pour NLTK et le cache Hugging Face,
  - réutilisation de cette résolution dans `secretarius_local/expression_extractor.py` et `secretarius_local/embeddings.py`.
- Externalisation des prompts:
  - renommage de `secretarius_local/prompts/prompt.txt` vers `secretarius_local/prompts/prompt_extracteur.txt`,
  - ajout de `secretarius_local/prompts/prompt_routeur.txt`,
  - chargement disque du prompt routeur dans `core/chef_orchestre.py`,
  - fallback de compatibilité conservé côté extracteur vers l'ancien `prompt.txt`.
- Observabilité renforcée:
  - journalisation dans `logs/guichet.log` du prompt envoyé à `llama.cpp` pour `extract_expressions`,
  - journalisation dans `logs/guichet.log` du prompt routeur final envoyé à Qwen (`system_prompt` + `messages`).
- Prompt routeur révisé:
  - version française dédiée au rôle de routeur MCP,
  - ajout de contraintes explicites contre la troncature du texte multiligne,
  - interdiction explicite de pseudo-valeurs comme `document: "text"`,
  - ajout d'exemples few-shot, dont un cas d'extraction multiligne.

### Diagnostic confirmé
- Le problème de troncature observé sur `extract_expressions` ne venait ni d'OpenWebUI ni de l'outil d'extraction.
- Les logs montrent que le routeur recevait bien l'intégralité du poème dans `messages`, puis que Qwen renvoyait un `action_input.text` tronqué à la première ligne, parfois avec un faux `document: "text"`.
- L'amélioration doit donc rester concentrée sur le prompt routeur et l'observabilité de son entrée/sortie.

### Tests / validation
- Régression détectée après externalisation du prompt routeur:
  - cause: utilisation de `.format(...)` sur un prompt fichier contenant des accolades JSON littérales,
  - symptôme: `KeyError: '"action"'` dans `tests/test_chef_orchestre.py`.
- Correctif appliqué:
  - remplacement du formatage global par une substitution ciblée de `{tools_schema}`.
- Validation effectuée dans l'environnement virtuel projet:
  - `source ../.venv/bin/activate && python -m unittest discover -s tests -p 'test_*.py' -v`
  - résultat: 17 tests OK.

### Point de reprise pour demain
- Rejouer un cas d'extraction multiligne via OpenWebUI et vérifier dans `logs/guichet.log`:
  - le bloc `Router LLM request`,
  - la ligne `Calling tool: extract_expressions ...`,
  - le bloc `llama.cpp extract request`.
- Si Qwen tronque encore `action_input.text`, décider entre:
  - renforcement supplémentaire du prompt routeur,
  - pré-routage déterministe pour certains cas très reconnaissables d'extraction.

---

## Mise à jour du 2026-03-11

### Décision d'architecture entérinée
- Séparation explicite entre :
  - fonctions métier internes,
  - outils MCP exposés au routeur,
  - infrastructure interne.
- Les outils MCP publics doivent refléter l'intention utilisateur, pas l'implémentation.
- Contrat visé :
  - `extract_expressions(text)` : extraction seule sur texte brut,
  - `index_text(text)` : indexation documentaire à partir d'une chaîne,
  - `search_text(query)` : interrogation documentaire par similarité.
- La structure `Document` est désormais considérée comme structure métier interne canonique, pas comme schéma d'entrée public à exposer au routeur.

### Spécification ajoutée
- Nouveau document de référence :
  - `spec_mcp_documentaire.md`
- Cette note formalise :
  - les fonctions métier attendues,
  - les outils MCP minimaux,
  - le rôle du `Document`,
  - et l'interdiction de mélanger backend LLM, structure documentaire et schémas MCP publics.

### Refactor métier / MCP
- Nouveau module métier :
  - `secretarius_local/document_pipeline.py`
- Fonctions ajoutées :
  - `analyse_texte_documentaire(...)`
  - `index_document_text(...)`
  - `search_documents_by_text(...)`
- `mcp_server.py` a été allégé :
  - les handlers MCP `index_text` et `search_text` délèguent maintenant au pipeline métier,
  - l'extraction d'expressions reste exposée séparément avec une sortie minimale,
  - les schémas publics MCP ont été simplifiés.

### Simplification des schémas MCP publics
- `extract_expressions` :
  - entrée publique minimale : `text`
  - suppression du faux schéma documentaire public et des paramètres LLM du `tools/list`.
- `index_text` :
  - entrée publique minimale : `text`
- `search_text` :
  - entrée publique minimale : `query`
- Les détails techniques restent gérés en interne par les handlers et/ou via l'environnement.

### Sorties d'outils
- `extract_expressions` :
  - en régime normal : sortie minimale (`expressions`, `warning`, `document` si nécessaire),
  - en debug : diagnostics détaillés seulement si `debug_return_raw=True`.
- `index_text` :
  - conserve un résumé compact,
  - détail `extract/index` seulement si `debug_full=True`.
- `search_text` :
  - conserve un résumé compact et les documents trouvés.

### Pipeline documentaire
- `index_text` :
  - analyse la chaîne en document,
  - extrait les expressions à partir du texte documentaire,
  - calcule les plongements de toutes les expressions,
  - insère en base le document enrichi via ces plongements.
- Validation explicite :
  - le document indexé n'est plus réduit conceptuellement à une simple liste de snippets,
  - le pipeline insère bien le document enrichi en s'appuyant sur les plongements de l'ensemble de ses expressions.

### Prompt routeur
- Réécriture du prompt routeur :
  - `secretarius_local/prompts/prompt_routeur.txt`
- Objectif :
  - réduire la verbosité,
  - rendre explicite la priorité de l'intention principale,
  - empêcher qu'une phrase contenue dans un document ("extraire les expressions...") soit prise pour l'instruction à exécuter.
- Cas observé et corrigé :
  - une demande commençant par `indexer :` contenant ensuite une phrase de type `Extraire les expressions...` était routée à tort vers `extract_expressions`,
  - le nouveau prompt donne priorité à `index_text` si l'intention principale est l'indexation.

### Nouveau canal notebook
- Ajout d'un nouveau canal OpenAI-compatible pour usage depuis un carnet Jupyter :
  - `notebook_api.py`
  - `server_notebook.py`
- Intégration dans le lancement multicanal :
  - `main_multicanal.py`
  - `config.yaml`
- Configuration par défaut :
  - port `8001`
  - modèle `secretarius-notebook`
  - journal `logs/notebook.log`
- Ce canal est distinct d'OpenWebUI :
  - canal logique `notebook`,
  - déduplication propre par canal,
  - pas d'heuristiques auxiliaires OpenWebUI (follow-ups, title, tags).

### Tests ajoutés / adaptés
- `tests/test_document_pipeline.py`
  - vérifie que l'indexation s'appuie sur les plongements de toutes les expressions.
- `tests/test_notebook_api.py`
  - vérifie que le canal notebook route bien les requêtes via `gateway.submit("notebook", ...)`.
- `tests/test_mcp_server_compact_responses.py`
  - adaptés au nouveau découpage métier / MCP.
- `tests/test_mcp_tools_catalog.py`
  - vérifie aussi la minimalité des schémas publics.

### Validation effectuée
- Dans la venv du projet :
```bash
../.venv/bin/python -m unittest tests.test_notebook_api -v
../.venv/bin/python -m unittest tests.test_document_pipeline tests.test_mcp_server_compact_responses tests.test_mcp_tools_catalog tests.test_chef_orchestre tests.test_guichet_unique -v
```
- Résultat :
  - `tests.test_notebook_api` : OK
  - `tests.test_document_pipeline` : OK
  - `tests.test_mcp_server_compact_responses` : OK
  - `tests.test_mcp_tools_catalog` : OK
  - `tests.test_chef_orchestre` : OK
  - `tests.test_guichet_unique` : OK

### Points d'attention restants
- `semantic_graph_search` et `expressions_to_embeddings` existent encore côté serveur MCP comme outils internes / bas niveau.
- Une décision reste à prendre :
  - les conserver cachés dans `mcp_server.py`,
  - ou les sortir plus franchement de la surface MCP si l'on veut une séparation encore plus stricte.
- Le canal notebook nécessite évidemment un redémarrage de `main_multicanal.py` pour ouvrir le port `8001`.

---

## Mise à jour du 2026-03-13

### Changements effectués
- Serveur MCP unifié autour de `secretarius_local/mcp_server.py` :
  - `secretarius_local/mcp_server.py` est maintenant le vrai point d'entrée MCP,
  - `tools/secretarius_server.py` est réduit a un shim,
  - `ask_oracle` a ete reintegre dans `mcp_server.py`,
  - `config.yaml` lance desormais `python -m secretarius_local.mcp_server`.
- Journalisation MCP :
  - `adapters/output/mcp_client.py` accepte maintenant un `env` explicite,
  - `app_runtime.py` injecte `SECRETARIUS_MCP_LOG`,
  - fichier de log cible : `logs/mcp_server.log`.
- Parsing documentaire enrichi dans `secretarius_local/document_pipeline.py` :
  - support de plusieurs URLs dans une meme chaine documentaire,
  - stockage dans `source.urls` avec conservation de `source.url` comme URL primaire,
  - support de `type_note: fugace|lecture|permanente`,
  - support de `doc_id: ...` dans la chaine documentaire.
- Indexation / mise a jour des notes :
  - ajout de `update_document_text(...)` dans `document_pipeline.py`,
  - ajout de l'outil MCP `update_text`,
  - semantique choisie : remplacement complet d'une note a partir d'un `doc_id` explicite.
- Couche Milvus (`secretarius_local/semantic_graph.py`) :
  - abandon des IDs temporels au profit d'identifiants stables par note/expression,
  - ajout de champs explicites `doc_id`, `expression_id`, `expression_norm`, `type_note`,
  - suppression des expressions obsoletes d'une note,
  - `upsert` des expressions encore presentes.
- Recherche documentaire :
  - `search_text` continue a partir des expressions caracteristiques de la requete,
  - ajout d'un seuil configurable `mcp_servers.secretarius.search_min_score` dans `config.yaml`,
  - renvoi de la structure document complete dans les resultats,
  - deduplication des resultats par `doc_id`,
  - reranking hybride :
    - base semantique,
    - bonus si des keywords de la requete matchent `user_fields.keywords`,
    - petit bonus si le titre matche des termes de requete.

### Tests / validation
- Tests ajoutes / adaptes :
  - `tests/test_document_pipeline.py`
  - `tests/test_semantic_graph.py`
  - `tests/test_mcp_server_compact_responses.py`
  - `tests/test_mcp_tools_catalog.py`
- Verifications executees dans la venv du projet :
```bash
../.venv/bin/python -m py_compile \
  secretarius_local/document_pipeline.py \
  secretarius_local/document_schema.py \
  secretarius_local/semantic_graph.py \
  secretarius_local/mcp_server.py

../.venv/bin/python -m unittest -q \
  tests/test_document_pipeline.py \
  tests/test_semantic_graph.py \
  tests/test_mcp_server_compact_responses.py \
  tests/test_mcp_tools_catalog.py
```
- Resultat :
  - validations OK sur le parsing documentaire,
  - validations OK sur la couche Milvus,
  - validations OK sur les sorties MCP compactes,
  - validations OK sur le catalogue d'outils publics.

### Decisions produit / technique enterinees
- Travailler au niveau note entiere plutot qu'au niveau "update partiel d'expressions".
- Introduire `/update` avant un eventuel `/upsert`.
- Contrat utilisateur vise pour les chaines documentaires :
  - `doc_id: ...` optionnel pour `/index`, obligatoire pour `/update`,
  - `type_note: ...`,
  - URLs multiples,
  - hashtags / keywords,
  - corps libre.
- La recherche reste d'abord semantique, puis rerankee au niveau note.

### Etat actuel utile pour demain
- La structure `Document` est maintenant suffisante pour servir de noeud canonique d'un graphe de notes.
- Les informations utiles pour un graphe de notes sont deja disponibles ou presque :
  - `doc_id`
  - `type_note`
  - `user_fields.title`
  - `user_fields.keywords`
  - `source.url` / `source.urls`
  - `derived.expressions`
- `search_text` renvoie deja la note complete, ce qui facilitera une presentation Markdown orientee note plutot qu'orientee hit vectoriel.

### Point d'attention pour la prochaine session
- Le prochain chantier doit porter sur la gestion et la presentation en Markdown des graphes de notes, pas sur la couche d'indexation de base.
- Il faudra clarifier des le depart :
  - quel est le "graphe" a presenter :
    - voisins semantiques retournes par Milvus,
    - liens bases sur mots-cles partages,
    - liens bases sur expressions partagees,
    - ou combinaison de ces criteres.
- Il faudra aussi decider du format Markdown cible :
  - simple liste de notes trouvees,
  - sections par voisin / score / mots-cles communs,
  - ou rendu plus explicite de type noeuds + aretes.

### Proposition de point de depart pour demain
1. Definir une structure interne "note graph view model" a partir des documents complets deja recuperes.
2. Produire un premier rendu Markdown lisible et compact pour ce graphe.
3. N'ajouter qu'ensuite des raffinements de scoring, d'explication des liens et de presentation.

---

## Mise a jour du 2026-03-14

### Reprise de session locale
- Ajout de `Prototype/session_resume.py` :
  - `python session_resume.py resume --last`
  - `python session_resume.py snapshot ...`
  - `python session_resume.py snapshot-auto ...`
- Ajout de cibles Makefile :
  - `make resume-last`
  - `make snapshot-session TITLE="..." SUMMARY="..." NEXT="..." NOTES="..."`
  - `make snapshot-auto`
- Ajout du dossier `Prototype/sessions/` pour stocker les snapshots Markdown.
- Premier snapshot cree :
  - `Prototype/sessions/SESSION_20260314_153256.md`
- Politique actuelle du resume local :
  - si un snapshot existe, `resume --last` affiche le plus recent ;
  - sinon fallback vers un resume construit a partir de `continuation.md` + etat git.

### Snapshot automatique periodique
- Ajout d'un mode auto avec garde-fous :
  - snapshot seulement si `git status --short -- Prototype` contient des changements ;
  - pas de nouveau snapshot si le precedent est trop recent.
- Script :
  - `Prototype/scripts/snapshot_auto.sh`
- Units `systemd --user` ajoutees :
  - `deploy/systemd-user/secretarius-prototype-session-snapshot.service`
  - `deploy/systemd-user/secretarius-prototype-session-snapshot.timer`
- Script d'installation :
  - `deploy/scripts/setup_prototype_session_snapshot_timer.sh`
- Etat actuel :
  - timer active cote utilisateur ;
  - frequence configuree : `OnBootSec=10m`, `OnUnitActiveSec=30m` ;
  - le timer survit aux redemarrages de VSCode, car il est gere par `systemd --user`.

### Services `systemd --user` operationnels
- Ollama :
  - unit : `deploy/systemd-user/secretarius_ollama.service`
  - commande reelle utilisee : `/usr/local/bin/ollama serve`
  - variables :
    - `HSA_OVERRIDE_GFX_VERSION=10.3.0`
    - `OLLAMA_HOST=0.0.0.0:11434`
    - `OLLAMA_CONTEXT_LENGTH=8192`
  - important : `ollama` n'existe pas dans `.venv/bin`, il faut utiliser `/usr/local/bin/ollama`.
- Serveur Secretarius Prototype :
  - unit : `deploy/systemd-user/secretarius_server.service`
  - commande :
    - `/home/mauceric/Secretarius/.venv/bin/python /home/mauceric/Secretarius/Prototype/server_secretarius.py`
  - le service demarre correctement et lance aussi `python -m secretarius_local.mcp_server`.
- Open WebUI :
  - choix retenu : service utilisateur generique, non prefixe `secretarius_`, car Open WebUI n'est pas propre au projet ;
  - unit : `deploy/systemd-user/open-webui.service`
  - commande :
    - `/home/mauceric/Secretarius/.venv/bin/open-webui serve`
  - script d'installation :
    - `deploy/scripts/setup_open_webui_service.sh`

### Scripts d'installation services
- `deploy/scripts/setup_secretarius_services.sh`
  - installe et active :
    - `secretarius_ollama.service`
    - `secretarius_server.service`
- `deploy/scripts/setup_open_webui_service.sh`
  - installe et active :
    - `open-webui.service`
- Tous ces services sont en `enabled`, donc relances automatiquement a la prochaine session utilisateur.

### Commandes utiles de verification
```bash
systemctl --user status --no-pager secretarius_ollama.service
systemctl --user status --no-pager secretarius_server.service
systemctl --user status --no-pager open-webui.service
systemctl --user status --no-pager secretarius-prototype-session-snapshot.timer
systemctl --user list-timers | grep secretarius-prototype-session-snapshot
```

### Point d'attention memoire / operations
- En cas de services deja lances a la main, les units `systemd --user` peuvent echouer au premier demarrage (port deja pris / process deja present).
- Procedure correcte :
  1. arreter les processus manuels ;
  2. `systemctl --user restart ...service` ;
  3. verifier avec `systemctl --user status --no-pager ...service`.

### Milvus Docker remis d'aplomb
- Inspection realisee dans `/home/mauceric/milvus`.
- Diagnostic retenu :
  - la pile Docker Milvus ne definissait aucune politique `restart`;
  - les conteneurs avaient donc `restart=no`;
  - cela explique bien des arrets non suivis de redemarrage automatique.
- Etat constate avant correction :
  - `docker compose -f /home/mauceric/milvus/docker-compose.yml ps`
  - les services etaient sains quand l'inspection a ete faite, mais rien n'assurait leur reprise automatique.
- Corrections appliquees dans `/home/mauceric/milvus/docker-compose.yml` :
  - suppression de `version: '3.5'` devenue obsolete ;
  - ajout de `restart: unless-stopped` sur :
    - `etcd`
    - `minio`
    - `standalone`
- Verification apres recreation :
  - `milvus-etcd`: `restart=unless-stopped`
  - `milvus-minio`: `restart=unless-stopped`
  - `milvus-standalone`: `restart=unless-stopped`

### Service utilisateur Milvus
- Ajout d'un service `systemd --user` :
  - `deploy/systemd-user/milvus-compose.service`
- Script d'installation :
  - `deploy/scripts/setup_milvus_compose_service.sh`
- Commandes du service :
  - `ExecStart=/usr/bin/docker compose -f /home/mauceric/milvus/docker-compose.yml up -d`
  - `ExecStop=/usr/bin/docker compose -f /home/mauceric/milvus/docker-compose.yml down`
- Etat actuel :
  - `milvus-compose.service` est `enabled`
  - le service est `active (exited)` avec `RemainAfterExit=yes`
  - la stack Docker est recreee / remontee correctement par `systemd --user`

### Commandes utiles Milvus
```bash
systemctl --user status --no-pager milvus-compose.service
docker compose -f /home/mauceric/milvus/docker-compose.yml ps
docker inspect -f '{{.Name}} restart={{.HostConfig.RestartPolicy.Name}} status={{.State.Status}}' milvus-etcd milvus-minio milvus-standalone
```

### Utilitaire export / import Milvus
- Ajout de `Prototype/scripts/milvus_collection_io.py`.
- Objectif :
  - exporter une collection Milvus dans un fichier JSON ;
  - recharger une collection depuis un fichier du meme format ;
  - supprimer proprement une collection de test ou de restauration.
- Format de dump :
  - `format`
  - `collection_name`
  - `schema`
  - `indexes`
  - `rows`
- Commandes utiles :
```bash
cd /home/mauceric/Secretarius/Prototype
source /home/mauceric/Secretarius/.venv/bin/activate

python scripts/milvus_collection_io.py export \
  --collection-name secretarius_semantic_graph \
  --output /tmp/secretarius_semantic_graph_dump.json

python scripts/milvus_collection_io.py import \
  --input /tmp/secretarius_semantic_graph_dump.json \
  --collection-name secretarius_semantic_graph_restored \
  --drop-if-exists

python scripts/milvus_collection_io.py drop \
  --collection-name secretarius_semantic_graph_restored \
  --require-exists
```
- Validation deja effectuee :
  - export reel de `secretarius_semantic_graph` OK ;
  - import reel vers `secretarius_semantic_graph_restore_test` OK.
