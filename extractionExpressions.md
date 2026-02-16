## Module d’analyse sémantique

## Contexte
Nous dépendons de plus en plus d’Internet ; il faut pourtant envisager des situations où l’on serait coupé du réseau : convulsions sociales, gouvernement autoritaire, catastrophes naturelles. Doit-on alors renoncer aux commodités qu’offre l’IA ? Il existe de grands modèles de langage suffisamment petits pour fonctionner sur des ordinateurs bon marché destinés au grand public. Ces modèles ne sont certes pas comparables à ceux que proposent les grands fournisseurs, qui nécessitent d’immenses centres de données ; ils se comparent pourtant avantageusement à ce qui se faisait de mieux il y a trois ans à peine. Leur plus grande limitation est l’étendue de leur connaissance, leur incapacité à connaître leurs limites et leur propension à inventer des réponses.

Si nous devenons, pour le meilleur et pour le pire, dépendants de l’IA, nous l’avons toujours été de notre mémoire, et prendre des notes ne pallie pas toujours nos limitations naturelles. Comment retrouve-t-on un texte écrit des années auparavant, comment savons-nous même que nous l’avons écrit ?

L’empire de la mémoire règne aussi bien sur nos esprits que sur le cerveau artificiel des systèmes d’IA. Nous nous proposons de montrer qu’il est possible, avec un petit budget, de construire un assistant capable de gérer intelligemment une grande quantité d’informations dans la plus grande confidentialité. Un assistant qui se souvienne d’anciennes notes que nous avions écrites longtemps auparavant, un assistant doté d’un savoir encyclopédique, un agent cognitif autonome, frugal et confidentiel en somme. Nous avons appelé cet assistant **Secretarius**.

## Architecture
Secretarius se décompose en trois modules principaux :
1. un module d’indexation ;
2. un module d’interrogation ;
3. un module d’analyse sémantique.

Le module d’analyse sémantique, utilisé par les deux autres, est le cœur de l’application. C’est lui que nous allons décrire dans ce document, avec un niveau de détail technique croissant.

### Le découpage en passages sémantiquement cohérents d’un texte
Description du *chunking* sémantique.

### L’extraction d’expressions caractéristiques de passages sémantiquement cohérents
Commençons par un exemple. Nous voulons être capables, pour le passage suivant :

> La jeune femme qui avait ces yeux se leva, et montra jusqu’à la ceinture sa taille enveloppée d’un camail à la turque (féredjé) aux plis longs et rigides. Le camail était de soie verte, orné de broderies d’argent. Un voile blanc enveloppait soigneusement la tête, n’en laissant paraître que le front et les grands yeux. Les prunelles étaient bien vertes, de cette teinte vert de mer d’autrefois chantée par les poètes d’Orient.

d’extraire la liste d’expressions suivante :
- camail,
- féredjé,
- soie verte,
- broderies d’argent,
- voile blanc,
- front,
- grands yeux,
- prunelles,
- teinte vert de mer.

On pourra toujours objecter qu’une telle liste n’est pas parfaite, mais même chez des indexeurs humains professionnels utilisant un thésaurus, il existait une grande variabilité de l’indexation. Nous n’utilisons pas de thésaurus, car ils introduisent une grande rigidité (acquisition, maintenance, domaines d’application, etc.), et parce que des techniques comme l’utilisation de vecteurs de mots, inaugurée par Gerard Salton, ont fait la preuve d’une très grande efficacité.

Nous prenons la peine d’extraire des expressions caractéristiques plutôt que l’ensemble des mots du texte (au filtrage par un antidictionnaire près : mots vides, mots trop fréquents, par exemple), car nous avons constaté expérimentalement qu’elles fournissent de meilleurs index.

En général, les méthodes d’indexation utilisées dans les systèmes d’information actuels, comme les RAG, utilisent la moyenne des plongements (c’est-à-dire des vecteurs beaucoup plus sophistiqués que les vecteurs de mots de G. Salton) de tous les mots du passage comme index (N.B. : à vérifier, existe-t-il d’autres techniques). Nous pensons que chaque passage devrait plutôt être considéré comme un sac de plongements (par analogie avec les sacs de mots de G. Salton), un peu comme le fait Omar Khattab dans ColBERT, à ceci près que nous utilisons des plongements d’unités sémantiquement significatives.

Les grands modèles de langage (LLM) s’acquittent assez bien de ce genre de tâches. Dans l’exemple donné plus haut, les expressions ont été extraites par DeepSeek V3 (DeepSeek dorénavant), mais l’exigence de confidentialité et de localité est une contrainte forte de Secretarius. Nous utilisons donc le modèle open source de Microsoft, Phi-4-mini-instruct (Phi4 dorénavant). Phi4 n’a pas les capacités de DeepSeek ; il nous faut donc utiliser une technique de distillation pour entraîner Phi4 à imiter le comportement de DeepSeek dans cette tâche d’extraction d’expressions significatives de passages. Pour ce faire, il nous faut constituer des corpus qui seront découpés en passages, dont DeepSeek extraira les expressions significatives afin de former un corpus d’apprentissage. Ce corpus sera ensuite utilisé pour entraîner Phi4. Nous présentons ces différentes étapes dans les paragraphes suivants :

#### Découpage de textes en passages sémantiquement cohérents (chunker sémantique)
La procédure utilisée pour découper le texte en passages peut être décrite ainsi :  
le texte est segmenté en phrases et chaque phrase est encodée avec *SentenceTransformer* pour obtenir des plongements normalisés. Pour chaque passage, un vecteur centre représentant le thème courant est actualisé.  
À l’arrivée d’une nouvelle phrase, la similarité cosinus avec ce centre est calculée ; si elle passe sous un seuil (après une taille minimale), cela signale un changement de sujet et une coupure est déclenchée.  
Le centre est mis à jour par moyenne exponentielle, ce qui rend la décision robuste aux transitions locales.  
Un recouvrement et des fusions a posteriori évitent les ruptures sèches et la production de passages trop courts.

#### Corpus d’apprentissage
Nous avons créé deux corpus, l’un synthétique, l’autre à partir d’un extrait en français de Wiki-40B, une collection de textes Wikipédia préparée par Google Research.

##### Corpus synthétique
Ce corpus synthétique est construit à partir d’un prompt initial (annexe A), affiné par la procédure d’optimisation GEPA de DSPy (annexe B). Ce prompt affiné a été utilisé pour demander à GPT-5 de fournir 1 000 documents.  
Les textes de ce corpus sont découpés en passages, et un prompt (annexe C) demandant à DeepSeek d’en extraire les expressions caractéristiques est affiné grâce à GEPA (annexe D). DeepSeek extrait alors les expressions caractéristiques de chaque passage, constituant ainsi un corpus d’apprentissage synthétique.

**Remarque importante** : dans un premier temps, nous avons demandé à GPT-5 de générer directement les expressions caractéristiques en même temps que le texte, sans découpage en passages. Il s’est avéré que cette approche produisait une indexation de moindre qualité en aval. Afin de ne pas disperser nos efforts, nous n’avons conservé que les textes générés, que nous avons ensuite découpés en passages, dont nous avons extrait les expressions caractéristiques. L’apprentissage est nettement meilleur en procédant ainsi.

##### Corpus Wikipédia
Nous avons extrait 1 000 textes de l’archive Wiki-40B, que nous avons convertis au format ISO-Latin-1, puis découpés en passages. Nous avons ensuite demandé à DeepSeek d’en extraire les expressions caractéristiques à l’aide du prompt optimisé par GEPA (annexe E).

**Frugalité** : le coût de construction de ces deux corpus s’est élevé à 10 € avec DeepSeek.

#### Entraînement d’un adaptateur LoRA de Phi4
L’utilisation d’adaptateurs de faible rang (Low Rank Adapters, ou LoRA) pour affiner (*fine-tuning*) un modèle de langage est une technique populaire et éprouvée. Le principe consiste à geler les poids du modèle et à n’entraîner que de petits réseaux additionnels (les adaptateurs), insérés à la sortie de certains sous-réseaux spécifiques. Ces adaptateurs apprennent alors à répondre correctement à des prompts dérivés du corpus d’apprentissage.

Par exemple, pour le passage donné en illustration au début du document, le prompt serait :

> **Quelles sont les expressions caractéristiques du texte** :  
> « La jeune femme qui avait ces yeux se leva, et montra jusqu’à la ceinture sa taille enveloppée d’un camail à la turque (féredjé) aux plis longs et rigides. Le camail était de soie verte, orné de broderies d’argent. Un voile blanc enveloppait soigneusement la tête, n’en laissant paraître que le front et les grands yeux. Les prunelles étaient bien vertes, de cette teinte vert de mer d’autrefois chantée par les poètes d’Orient. »

La réponse attendue serait la liste :
`['camail', 'féredjé', 'soie verte', 'broderies d’argent', 'voile blanc', 'front', 'grands yeux', 'prunelles', 'teinte vert de mer']`

**Entraînement local** : dans l’esprit de frugalité et de confidentialité de Secretarius, l’entraînement des deux adaptateurs a pu être effectué en 72 heures sur une machine de jeu AMD dotée de 30 Go de RAM et d’un GPU local de 8 Go de VRAM.

#### Analyse des résultats préliminaires
Après l’entraînement, nous avons effectué pour les deux corpus un test sur 20 % des données, obtenant une perplexité de 1,3. Nous avons également réalisé un test sur un corpus totalement étranger aux deux premiers, constitué des romans *Aziyadé* de Pierre Loti et *Notre-Dame de Paris* de Victor Hugo, et obtenu une perplexité de 1,4.

## Première phase de Secretarius : compréhension opérationnelle

Dans sa première phase, **Secretarius** se compose de deux modules applicatifs :
1. un module d’indexation ;
2. un module de requête.

Ces deux modules partagent la même brique linguistique : **Phi-4-mini-instruct** servi localement via `llama-server`, chargé avec un **adaptateur LoRA** spécialisé dans l’extraction d’expressions caractéristiques.

Le service d’inférence local est aligné avec la configuration `systemd` fournie :
- modèle GGUF `model-Q6_K.gguf` (Phi-4 + LoRA fusionné) ;
- fenêtre de contexte `-c 8192` ;
- offload GPU `-ngl 32` ;
- endpoint HTTP `http://0.0.0.0:8080`.

L’idée centrale de la phase 1 est la suivante : un texte (à indexer ou une requête utilisateur) n’est pas traité comme un bloc unique. Il est d’abord **découpé en passages sémantiquement cohérents**, puis chaque passage est **indexé par ses expressions caractéristiques**. Ces expressions sont ensuite vectorisées (plongements) et stockées/recherchées dans une base sémantique.

### Pipeline cible (phase 1)

1. **Ingestion/normalisation du texte**
- nettoyage des artefacts de format (`b"..."`, échappements, espaces, retours ligne) ;
- filtrage des contenus trop courts ou trop bruités.

2. **Découpage sémantique (chunking)**
- segmentation en phrases ;
- encodage des phrases avec `SentenceTransformer` (embeddings normalisés) ;
- maintien d’un centre thématique de chunk ;
- coupure si dérive sémantique (cosinus sous seuil) après une taille minimale ;
- recouvrement (overlap) et fusion a posteriori pour éviter des micro-chunks.

3. **Extraction d’expressions caractéristiques**
- envoi des chunks au modèle d’extraction (Phi-4 + LoRA en local) ;
- retour attendu par chunk : liste d’expressions nominales/collocations pertinentes ;
- format de sortie cible : `{"chunk": "...", "expressions_caracteristiques": [...]}`.

4. **Préparation à la base sémantique**
- calcul de plongements pour les expressions (et éventuellement pour les chunks) ;
- persistance avec métadonnées (document, chunk, offsets, source) ;
- cette étape sera l’objet de la phase suivante (implémentation de la BD sémantique).

### Ce que montre déjà `indexation_wiki40b`

Le répertoire `indexation_wiki40b` fournit un prototype utile de la chaîne :
- `chunk_data.py` : chunker sémantique robuste (nettoyage, embeddings de phrases, seuil de similarité, overlap, fusion/filtrage) ;
- `index_data.py` : extraction d’expressions par batch et structuration des chunks indexés ;
- `run_pipeline.sh` : enchaînement de la pipeline (nettoyage -> chunking -> indexation).

Ce prototype valide la logique méthodologique. La principale évolution pour Secretarius est le passage d’une extraction distante (DeepSeek dans le prototype) vers une extraction locale par `llama-server` sur `:8080`.

## Plan de mise en oeuvre (avant la BD sémantique)

### Étape 1 : figer le contrat de données commun
1. Définir le schéma d’entrée document (`id_doc`, `source`, `titre`, `contenu`).
2. Définir le schéma de sortie chunkée/indexée :
   `id_doc`, `id_chunk`, `chunk`, `expressions_caracteristiques`, `meta`.
3. Définir la représentation intermédiaire commune aux modules indexation/requête.

### Étape 2 : industrialiser le chunker sémantique
1. Reprendre la logique de `chunk_data.py` dans un module `secretarius/chunking/`.
2. Exposer les hyperparamètres (seuil cosinus, min/max mots, overlap, bruit).
3. Ajouter des tests de non-régression (nombre moyen de chunks, taux de micro-chunks).

### Étape 3 : brancher l’extracteur local Phi-4 + LoRA
1. Implémenter un client HTTP `llama-server` (`http://localhost:8080`).
2. Standardiser le prompt d’extraction (format JSON strict attendu).
3. Gérer robustement les erreurs (timeouts, JSON partiel, retries, logs).

### Étape 4 : assembler les deux modules applicatifs
1. **Module d’indexation** : document -> chunks -> expressions.
2. **Module de requête** : requête utilisateur -> pseudo-chunks -> expressions.
3. Garantir la symétrie de traitement pour réduire l’écart indexation/requête.

### Étape 5 : instrumenter et valider la phase 1
1. Jeu de test fixe (Wikipédia + textes littéraires hors domaine).
2. Métriques : latence par document, coût CPU/GPU, stabilité JSON, qualité perçue des expressions.
3. Rapport de validation phase 1 et gel des paramètres avant phase 2 (BD sémantique).

### Livrables de fin de phase 1
1. Un pipeline local reproductible `ingestion -> chunking -> extraction`.
2. Un format de sortie stable prêt à être vectorisé.
3. Un lot de référence permettant de mesurer les régressions.

## Tests de fonctionnement prévus (phase 1)

Les tests ci-dessous sont prévus pour une exécution sur **Sanroque** (référence de chemins : `~/` = `/home/mauceric`).

### 1) Tests smoke (démarrage et santé)
1. Vérifier que `llama-server` répond sur `http://localhost:8080`.
2. Exécuter une extraction sur un chunk court de référence.
3. Valider que la réponse est parsable en JSON et contient une liste d’expressions.

**Critère d’acceptation** : service joignable, réponse < 10 s sur chunk court, JSON valide.

### 2) Tests unitaires chunking
1. Entrée vide, bruitée, et texte court : vérifier les garde-fous.
2. Texte multi-thèmes : vérifier qu’au moins une coupure sémantique est produite.
3. Vérifier l’absence de micro-chunks après fusion (selon `min_chunk_words/min_chunk_chars`).

**Critère d’acceptation** : aucun crash, sorties déterministes à paramètres constants.

### 3) Tests d’intégration indexation
1. Prendre un mini-corpus (ex. 20 documents) ;
2. Exécuter le pipeline complet : nettoyage -> chunking -> extraction ;
3. Vérifier la structure de sortie :
   `id_doc`, `id_chunk`, `chunk`, `expressions_caracteristiques`.

**Critère d’acceptation** : 100 % des documents traités, et < 2 % de chunks sans expressions.

### 4) Tests d’intégration requête
1. Soumettre un lot de requêtes réalistes (phrases courtes, longues, ambiguës).
2. Vérifier que la même chaîne chunking/extraction est appliquée côté requête.
3. Vérifier la stabilité du format de sortie pour brancher ensuite la BD sémantique.

**Critère d’acceptation** : format strictement identique à l’indexation (hors métadonnées de contexte).

### 5) Tests de robustesse
1. Timeouts réseau locaux simulés (serveur lent).
2. Réponses mal formées (JSON partiel) et stratégie de retry.
3. Charge modérée (lots de chunks) pour observer latence et erreurs.

**Critère d’acceptation** : pas d’arrêt brutal du pipeline, erreurs journalisées, reprise automatique.

### 6) Tests de non-régression
1. Geler un corpus de référence (Wiki + littérature).
2. Stocker les métriques de base (latence, nb moyen de chunks/doc, expressions/chunk).
3. Comparer automatiquement chaque nouvelle version aux métriques gelées.

**Critère d’acceptation** : aucune dérive majeure non expliquée (seuils à fixer avant phase 2).
