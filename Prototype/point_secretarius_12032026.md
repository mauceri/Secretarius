# Point Secretarius - 12/03/2026

## Objet Du Document

Ce document synthétise l'état du projet `Secretarius/Prototype` au 12 mars 2026.

Il reprend l'historique de [continuation.md](/home/mauceric/Secretarius/Prototype/continuation.md) et le prolonge avec les changements effectués ensuite, en particulier :
- la consolidation du serveur MCP local
- le découplage du TUI et du backend serveur
- l'ajout du canal `session_messenger`
- les ajustements du routeur et de l'extracteur
- la préparation d'une migration future du serveur MCP

L'objectif est de fournir un point de reprise lisible sans devoir relire toute la conversation ni l'historique détaillé des commits.

## Contexte General

L'objectif initial du `Prototype` a été de rendre `Secretarius` autonome dans ce répertoire, avec :
- un serveur MCP local fonctionnel
- des outils embarqués localement
- un runtime sans dépendance code directe vers `../secretarius`

Au fil des sessions, le projet a évolué vers un système multi-canaux piloté par un seul backend agentique, avec séparation progressive entre :
- orchestration
- outils MCP
- pipeline documentaire
- canaux d'entrée

## Base Initiale Consolidée

Les travaux déjà documentés dans [continuation.md](/home/mauceric/Secretarius/Prototype/continuation.md) peuvent être résumés ainsi :

### Client MCP Et Runtime

- correction du client MCP dans `adapters/output/mcp_client.py`
- API stdio MCP stabilisée
- `cwd` explicite
- `main.py` et `app_runtime.py` durcis pour mieux gérer les chemins et le fallback config

### Outils Locaux

Import local dans `Prototype` des briques nécessaires :
- `secretarius_local/mcp_server.py`
- `secretarius_local/expression_extractor.py`
- `secretarius_local/embeddings.py`
- `secretarius_local/semantic_graph.py`
- `secretarius_local/document_schema.py`
- `secretarius_local/vendor/chunk_data.py`
- `secretarius_local/runtime_paths.py`

### Orchestrateur

Durcissement progressif de `core/chef_orchestre.py` :
- meilleure tolérance aux sorties JSON imparfaites du LLM
- politique plus déterministe de sélection d'outil
- garde-fous anti-boucle
- retour direct de l'observation outil sans reformulation LLM
- sanitation technique des arguments d'outils

### Prompt Routeur

Externalisation du prompt routeur dans :
- [prompt_routeur.txt](/home/mauceric/Secretarius/Prototype/secretarius_local/prompts/prompt_routeur.txt)

Objectifs poursuivis :
- rendre explicite la logique de choix d'outil
- empêcher qu'un texte fourni comme document soit interprété comme une instruction
- imposer une sortie JSON stricte

### Observabilité

Mise en place et enrichissement des journaux :
- [guichet.log](/home/mauceric/Secretarius/Prototype/logs/guichet.log)
- [llm_raw.log](/home/mauceric/Secretarius/Prototype/logs/llm_raw.log)
- [openwebui.log](/home/mauceric/Secretarius/Prototype/logs/openwebui.log)
- [notebook.log](/home/mauceric/Secretarius/Prototype/logs/notebook.log)

Les journaux ont servi à diagnostiquer :
- la troncature du texte par le routeur
- les erreurs de sélection d'outil
- les charges auxiliaires envoyées par OpenWebUI
- les appels effectifs à `llama.cpp`

### Evolution Des Outils MCP Publics

Le catalogue public a été orienté métier :
- `extract_expressions(text)`
- `index_text(text)`
- `search_text(query)`
- `ask_oracle(question)`

Les outils plus techniques comme `expressions_to_embeddings` ou les structures documentaires détaillées ont été masqués du contrat public du routeur.

### Pipeline Documentaire

Introduction de [document_pipeline.py](/home/mauceric/Secretarius/Prototype/secretarius_local/document_pipeline.py) pour :
- analyser une chaîne documentaire
- produire un document enrichi
- extraire les expressions
- calculer les embeddings
- insérer et interroger le graphe sémantique

Cette étape a clarifié la séparation entre :
- fonction métier interne
- outil MCP exposé
- infrastructure MCP

### Canaux

Introduction progressive de plusieurs canaux autour d'un guichet unique :
- TUI Textual
- OpenWebUI
- notebook API

Le point pivot est [guichet_unique.py](/home/mauceric/Secretarius/Prototype/adapters/input/guichet_unique.py), qui gère :
- journalisation par canal
- déduplication concurrente
- déduplication courte durée
- collecte de la réponse finale

## Changements Importants Ulterieurs Au 11/03/2026

Les points suivants ne figuraient pas encore entièrement dans `continuation.md` et constituent l'essentiel du travail complémentaire réalisé ensuite.

## 1. Canal Session Messenger

### Objectif

Ajouter un canal `session_messenger` inspiré du bot `session_bot_openclaw`, mais intégré au `Prototype`.

### Résultat

Ajout de :
- [session_messenger_api.py](/home/mauceric/Secretarius/Prototype/session_messenger_api.py)
- [server_session_messenger.py](/home/mauceric/Secretarius/Prototype/server_session_messenger.py)
- [session_messenger_bridge/](/home/mauceric/Secretarius/Prototype/session_messenger_bridge)

Le bridge Session Messenger comprend :
- `session_bot.ts`
- `db.ts`
- `package.json`
- `tsconfig.json`
- `README.md`
- `session-messenger-bot.service`
- `session_bot_secretarius.service`

### Fonctionnement

Le bridge :
- ouvre une session Session Messenger
- déduplique les messages entrants via SQLite
- transmet les messages vers l'API Python locale
- journalise l'identifiant Session du bot dans le log de canal

Le backend Python :
- expose `/session/message`
- appelle `gateway.submit("session_messenger", ...)`
- renvoie le texte de réponse au bridge

### Service Systemd Utilisateur

Préparation et installation du fichier :
- [session_bot_secretarius.service](/home/mauceric/.config/systemd/user/session_bot_secretarius.service)

Cela permet de lancer le bridge Session en service utilisateur, sans dépendre d'un lancement manuel permanent.

## 2. Découplage Du TUI Et Du Backend

### Problème initial

Le lancement `main_multicanal.py` mélangeait :
- TUI local
- backend serveur
- APIs réseau

Cela rendait difficile :
- le lancement autonome du backend
- l'intégration avec Session Messenger
- l'usage en service

### Solution mise en place

Création d'un backend serveur sans TUI :
- [server_secretarius.py](/home/mauceric/Secretarius/Prototype/server_secretarius.py)

Création d'un TUI client séparé :
- [tui_secretarius.py](/home/mauceric/Secretarius/Prototype/tui_secretarius.py)

Ajout d'une API locale dédiée au TUI :
- [tui_api.py](/home/mauceric/Secretarius/Prototype/tui_api.py)

Adaptation du TUI pour devenir client HTTP local :
- [tui_guichet.py](/home/mauceric/Secretarius/Prototype/adapters/input/tui_guichet.py)

### Effet architectural

Le backend peut désormais :
- tourner sans interface locale
- servir OpenWebUI, notebook, Session Messenger et TUI API

Le TUI devient un client comme les autres, au lieu d'être fusionné au runtime backend.

## 3. Enrichissement Du Guichet Unique Pour Le TUI Distant

Ajout dans [guichet_unique.py](/home/mauceric/Secretarius/Prototype/adapters/input/guichet_unique.py) de la capacité à collecter :
- les messages
- les pensées intermédiaires

Ajout d'une méthode de type `submit_with_trace(...)` pour permettre au TUI distant :
- d'afficher la réponse finale
- d'afficher les traces de raisonnement visibles

Cela a permis de conserver une expérience TUI proche de l'ancienne, malgré le découplage.

## 4. Simplification De L'Orchestrateur

### Constat

`ChefDOrchestre` avait accumulé trop de logique de réécriture :
- suppression de préfixes d'instruction
- récupération implicite du texte
- correction sémantique locale de certains payloads

Cela entrait en contradiction avec la ligne voulue :
- le routeur doit décider
- l'orchestrateur ne doit pas réinterpréter sémantiquement l'intention

### Changement effectué

Dans [chef_orchestre.py](/home/mauceric/Secretarius/Prototype/core/chef_orchestre.py) :
- retrait des prétraitements sémantiques
- conservation des garde-fous techniques
- conservation de la sanitation stricte des arguments autorisés
- maintien du blocage explicite de `ask_oracle`

### Conséquence

Le routeur reçoit et transmet désormais la responsabilité de choisir l'outil et de fournir le bon texte brut. L'orchestrateur redevient plus proche d'un adaptateur d'exécution.

## 5. Robustesse Du Parsing JSON Du Routeur

Avec des modèles plus petits ou intermédiaires, certaines réponses du routeur étaient quasi valides mais légèrement tronquées, par exemple avec une accolade finale manquante.

Un rattrapage minimal a été ajouté dans [chef_orchestre.py](/home/mauceric/Secretarius/Prototype/core/chef_orchestre.py) :
- nettoyage des blocs Markdown
- tentative de `json.loads`
- réparation minimale des accolades ou crochets finaux manquants

Important :
- aucune heuristique métier n'a été réintroduite
- seule la tolérance structurelle du parseur a été améliorée

## 6. Commandes Directes Sans Passage Par Le LLM

Ajout d'un mode de bypass explicite du routeur dans [chef_orchestre.py](/home/mauceric/Secretarius/Prototype/core/chef_orchestre.py) avec les préfixes :
- `/exp <texte>` -> `extract_expressions`
- `/index <texte>` -> `index_text`
- `/req <requete>` -> `search_text`

Ces commandes :
- court-circuitent complètement le routeur LLM
- évitent la surcharge quand l'intention utilisateur est explicite
- rendent le système plus déterministe pour les usages avancés

Un correctif a ensuite été ajouté pour accepter également :
- `/index` suivi d'un saut de ligne
- et plus généralement tout séparateur blanc, pas seulement l'espace

## 7. Nouveau Fallback Par Defaut Du Routeur

Quand le routeur ne trouve pas d'outil exploitable, le comportement a été changé :
- au lieu de répondre immédiatement `Aucun outil ne correspond a votre demande.`
- le système tente désormais `index_text` avec l'intégralité de la chaîne utilisateur

Ce choix a été introduit explicitement à la demande de l'utilisateur pour fournir un comportement de sauvegarde orienté indexation.

## 8. Evolution De L'Extracteur

### Problème identifié

Le pipeline d'extraction contenait un paramètre interne `per_chunk_llm` permettant de choisir entre :
- un seul appel LLM sur le texte complet
- un appel LLM par chunk

Ce paramètre a été jugé contraire à l'architecture souhaitée, où :
- les textes doivent être systématiquement chunkés
- le chunker doit décider de la taille
- l'inférence doit travailler sur les chunks

### Changements effectués

Suppression de `per_chunk_llm` dans :
- [expression_extractor.py](/home/mauceric/Secretarius/Prototype/secretarius_local/expression_extractor.py)
- [document_pipeline.py](/home/mauceric/Secretarius/Prototype/secretarius_local/document_pipeline.py)
- [mcp_server.py](/home/mauceric/Secretarius/Prototype/secretarius_local/mcp_server.py)

Conséquence :
- l'extraction se fait désormais toujours chunk par chunk
- les résultats sont agrégés ensuite

### Alignement Des Defaults `max_tokens`

Les défauts locaux `512` ont été remplacés par `20480` pour s'aligner sur la configuration du serveur `llama.cpp` utilisée dans l'environnement cible.

Cela concerne notamment :
- `expression_extractor.py`
- `document_pipeline.py`
- `mcp_server.py`

## 9. Ajustements De Modeles Routeurs

Plusieurs essais de modèles routeurs ont été faits au fil du travail :
- `qwen3:0.6b`
- `qwen3:8b`
- `qwen3.5:2B`

Constats :
- les petits modèles ont besoin d'un prompt très contraint
- certains modèles choisissent le bon outil mais produisent parfois un JSON légèrement invalide
- les modèles plus gros réduisent certaines erreurs mais ne suppriment pas tous les problèmes de troncature ou de reformulation

Le système a donc été rendu plus robuste sur le parsing et l'observabilité, sans remettre de logique métier cachée dans l'orchestrateur.

## 10. Planification D'Une Refonte Du Serveur MCP

Un document de plan détaillé a été rédigé :
- [migration_serveur_MCP.md](/home/mauceric/Secretarius/Prototype/migration_serveur_MCP.md)

Objectif :
- préparer une future migration du serveur MCP vers un registre d'outils
- éviter de lancer cette refonte immédiatement
- disposer d'un plan de reprise propre pour plus tard

Le document couvre :
- l'état actuel
- l'architecture cible
- les invariants à préserver
- les étapes de migration
- le plan de tests

## Validation Et Tests

Au fil des changements récents, plusieurs validations ont été exécutées dans la venv du projet :

### Orchestrateur Et Canaux

```bash
../.venv/bin/python -m unittest -q tests/test_chef_orchestre.py
../.venv/bin/python -m unittest -q tests/test_guichet_unique.py tests/test_notebook_api.py tests/test_session_messenger_api.py tests/test_tui_api.py
```

### Extraction Et MCP

```bash
../.venv/bin/python -m unittest -q tests/test_mcp_server_compact_responses.py tests/test_document_pipeline.py tests/test_chef_orchestre.py
```

### Compilation Python

```bash
../.venv/bin/python -m py_compile core/chef_orchestre.py
../.venv/bin/python -m py_compile secretarius_local/expression_extractor.py secretarius_local/document_pipeline.py secretarius_local/mcp_server.py
```

### Résultat Global

Les tests ciblés ajoutés ou adaptés au cours de cette phase sont passés au vert au moment du point.

## Sauvegarde Git

L'état correspondant à cette consolidation a été committé et poussé sur `main` avec :

- commit : `7f8abfb`
- message : `Sauvegarde et decouplage du serveur Secretarius`

Les artefacts runtime non committés ont été volontairement laissés hors historique :
- logs
- SQLite du bridge Session
- fichier de configuration/mnémonique Session local

## Etat Fonctionnel Au 12/03/2026

Le système peut désormais fonctionner avec :
- un backend serveur découplé
- un TUI client séparé
- OpenWebUI
- notebook API
- Session Messenger via bridge dédié

Les points fonctionnels importants sont :
- outils MCP publics orientés intention métier
- routeur davantage recentré sur son rôle
- extraction toujours chunkée
- parsing JSON plus robuste
- commandes directes `/exp`, `/index`, `/req`
- fallback par défaut du routeur vers `index_text`

## Points D'Attention Pour La Suite

### 1. Routeur

Le principal point de fragilité restant est la qualité du routeur selon le modèle choisi :
- choix de l'outil
- recopie intégrale du texte brut
- JSON strict

### 2. Fallback D'Indexation

Le fallback automatique vers `index_text` quand le routeur ne choisit rien est pratique, mais il modifie la sémantique de l'échec. Il faudra vérifier qu'il ne masque pas trop de cas d'erreur ou de mauvais routage.

### 3. Serveur MCP

La structure actuelle reste encore largement monolithique, même si elle est devenue plus propre. La migration future vers un registre d'outils reste pertinente.

### 4. Milvus

Le prochain thème de travail annoncé est le passage ou le recentrage autour de Milvus. La base actuelle est déjà compatible avec cette orientation puisque :
- `index_text`
- `search_text`
- le pipeline documentaire
- et `semantic_graph_search`

s'appuient déjà sur cette logique documentaire et vectorielle.

## Resume Executif

Au 12/03/2026, `Secretarius/Prototype` est dans un état significativement plus structuré qu'au départ :

- autonome localement
- multi-canaux
- plus observable
- plus découplé
- plus robuste sur le plan opérationnel

Le système est désormais suffisamment stabilisé pour :
- être sauvegardé proprement
- servir de base de reprise
- et engager ensuite un nouveau chantier thématique, notamment autour de Milvus, sans repartir d'une base fragile.
