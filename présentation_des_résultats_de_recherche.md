# Présentation des résultats de recherche

## Objectif

Alléger fortement la réponse utilisateur de `/req`.

La structure actuelle est trop orientée debug / interne et pas assez orientée lecture. La réponse de recherche doit devenir une vue de consultation claire, avec un niveau de détail limité aux informations utiles pour juger rapidement de la pertinence d'un document.

## Premier nettoyage proposé

Pour la présentation standard des résultats, ne conserver que :

- les métadonnées utiles du document ;
- une liste simple des expressions du document ;
- l'expression du document qui a servi à l'indexation du hit retenu ;
- le score de similarité associé ;
- un score global de pertinence utilisé pour ordonner les résultats.

À retirer de la réponse standard :

- `chunks` ;
- les `hash` ;
- la structure complète `derived` ;
- dans `expressions` : `norm`, `weight`, `span`, `embedding_ref` ;
- dans `indexing` : tout sauf l'expression retenue et son score.

## Forme cible minimale d'un résultat

Chaque document retourné pourrait ressembler à ceci :

```json
{
  "doc_id": "...",
  "title": "...",
  "type": "...",
  "document_date": "...",
  "source": {
    "url": "..."
  },
  "text": "...",
  "keywords": ["..."],
  "expressions": [
    "expression 1",
    "expression 2"
  ],
  "best_match": {
    "document_expression": "...",
    "score": 0.96
  },
  "global_score": 1.08
}
```

## Évolution souhaitable ensuite

Deuxième étape, plus orientée usage :

- regrouper explicitement les résultats par document logique ;
- afficher le document une seule fois ;
- rattacher à chaque document les correspondances significatives avec la requête ;
- exposer les triplets `(expression_requete, expression_document, score_similarite)` ;
- ne garder que les triplets dont le score dépasse le seuil configuré.

Cela donnerait une structure plus utile, par exemple :

```json
{
  "doc_id": "...",
  "title": "...",
  "text": "...",
  "expressions": ["..."],
  "matches": [
    {
      "query_expression": "...",
      "document_expression": "...",
      "score": 0.91
    }
  ],
  "global_score": 1.14
}
```

## Idée de commande complémentaire

`/req` pourrait devenir une vue courte de résultats.

`/doc` pourrait ensuite renvoyer le document complet, pour inspection détaillée.

Séparation proposée :

- `/req` : recherche et synthèse légère ;
- `/doc` : détail complet d'un document.

## Score actuel : `combined_score`

Dans l'implémentation actuelle, `combined_score` n'est pas une pure similarité vectorielle.

Il est calculé comme :

```python
combined_score = semantic_score + keyword_bonus + title_bonus
```

avec :

- `semantic_score` : score renvoyé par Milvus ;
- `keyword_bonus = len(keyword_matches) * _KEYWORD_MATCH_BONUS` ;
- `title_bonus = _TITLE_MATCH_BONUS if title_matches else 0.0`.

Donc `combined_score` est déjà un score hybride :

- similarité sémantique ;
- bonus lexical sur les mots-clés ;
- bonus lexical sur le titre.

## Nouveau besoin : score global avec intersection requête / document

Il faut ajouter un score global qui prenne explicitement en compte le recouvrement entre :

- les expressions extraites de la requête ;
- les expressions du document ;
- et, plus largement, les mots-clés ou termes saillants partagés.

Ce score global servirait au reranking final des documents.

Intuition :

- un document ne doit pas seulement remonter parce qu'un vecteur isolé matche bien ;
- il doit être favorisé si plusieurs éléments de la requête trouvent des correspondances cohérentes dans le document ;
- un document dont plusieurs expressions recoupent la requête doit être mieux classé qu'un document porté par un seul hit fort mais isolé.

## Proposition de reranking

Après la récupération initiale des hits Milvus, effectuer un reranking par document logique.

### Signaux à combiner

Pour chaque document :

- meilleur score de similarité vectorielle ;
- moyenne des meilleurs scores de correspondance ;
- nombre de couples `(expression_requete, expression_document)` au-dessus du seuil ;
- taux de couverture des expressions de requête ;
- intersection des mots-clés ;
- bonus de titre éventuel.

### Score global proposé

Première formule simple et explicable :

```text
global_score =
  0.50 * best_vector_score
  + 0.20 * avg_match_score
  + 0.20 * query_coverage
  + 0.10 * keyword_overlap
  + title_bonus
```

où :

- `best_vector_score` = meilleur score de similarité document ;
- `avg_match_score` = moyenne des scores des triplets conservés ;
- `query_coverage` = proportion d'expressions de la requête ayant au moins un match document au-dessus du seuil ;
- `keyword_overlap` = ratio d'intersection entre keywords requête et document ;
- `title_bonus` = petit bonus fixe si le titre contient un terme saillant de la requête.

Cette formule a l'avantage d'être :

- simple à comprendre ;
- stable ;
- pilotable par configuration ;
- compatible avec l'état actuel du pipeline.

## Variante de reranking recommandée

Une bonne approche pragmatique serait :

1. récupérer les hits Milvus comme aujourd'hui ;
2. regrouper par `doc_id` ;
3. reconstruire pour chaque document les correspondances entre expressions de requête et expressions indexées du document ;
4. ne conserver que les couples au-dessus d'un seuil de similarité ;
5. calculer un `global_score` ;
6. trier les documents sur `global_score` et non plus seulement sur le meilleur hit.

## Structure cible pour le reranking

La réponse pourrait embarquer les correspondances utiles :

```json
{
  "doc_id": "...",
  "title": "...",
  "expressions": ["..."],
  "matches": [
    {
      "query_expression": "cavalerie soviétique",
      "document_expression": "unités de cavalerie",
      "score": 0.96
    },
    {
      "query_expression": "Boudienny",
      "document_expression": "marche de Boudienny",
      "score": 0.91
    }
  ],
  "best_match": {
    "document_expression": "unités de cavalerie",
    "score": 0.96
  },
  "global_score": 1.14
}
```

## Bénéfices attendus

- résultats plus lisibles ;
- documents moins dupliqués ;
- meilleur classement des documents réellement pertinents ;
- séparation plus nette entre vue utilisateur et données techniques internes ;
- base claire pour une future commande `/doc`.

## Spécification technique pour la refonte de `/req`

### Périmètre

Cette refonte concerne uniquement la présentation et le reranking des résultats de `search_text` pour `/req`.

Ne pas modifier dans un premier temps :

- la logique d'extraction d'expressions ;
- la génération des embeddings ;
- l'appel bas niveau à Milvus ;
- le contrat de `/index` et `/update`.

### Principe général

Conserver la recherche initiale actuelle :

1. extraction des expressions de la requête ;
2. génération des embeddings ;
3. appel Milvus ;
4. récupération des hits bruts.

Puis ajouter une nouvelle phase applicative :

5. regroupement par `doc_id` ;
6. reconstruction d'un objet document allégé ;
7. calcul des correspondances requête / document ;
8. calcul d'un `global_score` ;
9. reranking final sur `global_score` ;
10. sérialisation d'une réponse allégée pour `/req`.

### Sortie cible de `/req`

La réponse de `/req` doit exposer :

- un résumé global de recherche ;
- une liste de documents dédupliqués par `doc_id` ;
- pour chaque document, uniquement les champs utiles à l'utilisateur.

Proposition de forme cible :

```json
{
  "status": "ok",
  "tool": "search_text",
  "message": "Recherche semantique executee.",
  "query": "...",
  "summary": {
    "collection_name": "...",
    "query_count": 0,
    "document_count": 0,
    "top_k": 10,
    "min_score": 0.75
  },
  "documents": [
    {
      "doc_id": "...",
      "title": "...",
      "type": "...",
      "document_date": "...",
      "url": "...",
      "text": "...",
      "keywords": ["..."],
      "expressions": ["..."],
      "best_match": {
        "query_expression": "...",
        "document_expression": "...",
        "score": 0.96
      },
      "matches": [
        {
          "query_expression": "...",
          "document_expression": "...",
          "score": 0.96
        }
      ],
      "global_score": 1.14
    }
  ],
  "warning": null
}
```

### Champs à supprimer de la sortie standard

Ne plus exposer dans la sortie standard de `/req` :

- `chunks` ;
- `hash` ;
- `derived` détaillé ;
- `embedding_ref` ;
- `norm` ;
- `weight` ;
- `span` ;
- les détails internes complets de `indexing` ;
- l'identifiant Milvus de ligne, sauf besoin explicite de debug.

### Champs à conserver ou dériver

Pour chaque document, conserver ou dériver :

- `doc_id` ;
- `title` ;
- `type` ;
- `document_date` ;
- `url` principale si disponible ;
- `text` ;
- `keywords` ;
- `expressions` sous forme de liste de chaînes ;
- `best_match` ;
- `matches` ;
- `global_score`.

### Déduplication par document

Le regroupement doit se faire par `doc_id`.

Pour un même `doc_id`, plusieurs hits Milvus peuvent exister, car plusieurs expressions du document peuvent avoir été indexées.

La nouvelle logique doit :

- agréger tous les hits associés au même `doc_id` ;
- reconstruire les différentes expressions document qui ont matché ;
- calculer les meilleures correspondances avec la requête ;
- produire un seul résultat final par document.

### Reconstruction des matches

Pour chaque hit regroupé sous un même document :

- récupérer l'expression de document depuis `indexing.source_expression` quand elle est disponible ;
- récupérer le score vectoriel du hit ;
- relier ce hit à l'expression de requête qui l'a produit si cette information est disponible par position ou par structure de retour ;
- sinon, conserver au minimum l'expression document et le score.

Structure cible minimale d'un match :

```json
{
  "query_expression": "...",
  "document_expression": "...",
  "score": 0.91
}
```

### Seuil de conservation des matches

Les couples `(expression_requete, expression_document)` ne doivent être conservés que si leur score dépasse le seuil configuré.

Ce seuil peut être :

- le `min_score` déjà existant ;
- ou un seuil dédié de reranking si l'on veut distinguer recherche initiale et présentation finale.

### Définition de `best_match`

`best_match` est le meilleur triplet conservé pour le document, selon le score de similarité.

Il doit contenir :

- `query_expression` ;
- `document_expression` ;
- `score`.

### Définition de `global_score`

`global_score` doit servir de score final de classement document.

Il ne remplace pas le score vectoriel brut ; il le complète.

Formule initiale recommandée :

```text
global_score =
  0.50 * best_vector_score
  + 0.20 * avg_match_score
  + 0.20 * query_coverage
  + 0.10 * keyword_overlap
  + title_bonus
```

Définitions :

- `best_vector_score` = meilleur score de hit du document ;
- `avg_match_score` = moyenne des scores des matches conservés ;
- `query_coverage` = nombre d'expressions de requête couvertes / nombre total d'expressions de requête ;
- `keyword_overlap` = ratio de recouvrement des keywords requête / document ;
- `title_bonus` = petit bonus si le titre contient un terme saillant de la requête.

### Ordonnancement final

L'ordre final des documents doit être déterminé par :

1. `global_score` décroissant ;
2. `best_vector_score` décroissant en cas d'égalité ;
3. ordre stable par `doc_id` en dernier recours.

### Compatibilité et debug

Pour éviter de casser les usages internes, il est recommandé de conserver un mode détaillé activable, par exemple via `debug_full=true`.

Comportement proposé :

- sortie standard allégée par défaut ;
- sortie riche historique uniquement en mode debug.

### Points d'implémentation probables

Les zones les plus probables pour implémenter cette refonte sont :

- `Prototype/secretarius_local/mcp_server.py`
- `Prototype/secretarius_local/document_pipeline.py`

En particulier :

- conserver `semantic_graph_search_milvus(...)` tel quel dans un premier temps ;
- modifier la couche qui transforme les hits bruts en documents utilisateurs ;
- remplacer la logique actuelle de `_extract_search_documents(...)` par une logique d'agrégation et de reranking par document.

### Étapes d'implémentation recommandées

1. Introduire une nouvelle fonction d'agrégation par `doc_id`.
2. Produire une structure document allégée.
3. Ajouter la reconstruction des matches.
4. Ajouter le calcul de `global_score`.
5. Trier les documents sur `global_score`.
6. Garder un mode `debug_full` pour la structure détaillée historique.
7. Ajuster les tests de contrat de `search_text`.

### Tests à prévoir

- un document indexé par plusieurs expressions ne doit apparaître qu'une fois ;
- la sortie standard ne doit plus contenir `chunks`, `hash`, `embedding_ref`, `span`, `norm`, `weight` ;
- `best_match` doit être cohérent avec les hits conservés ;
- le classement final doit changer quand la couverture de requête augmente ;
- deux documents proches en score vectoriel doivent pouvoir être départagés par le reranking ;
- `debug_full=true` doit conserver la sortie détaillée.

### Suite possible

Après stabilisation de cette refonte :

- introduire `/doc <doc_id>` pour le détail complet d'un document ;
- éventuellement exposer les identifiants Milvus uniquement dans cette vue détaillée ou en debug.

## Décision de conception provisoire

À court terme :

- alléger fortement la sortie de `/req` ;
- regrouper les résultats par document ;
- afficher une liste simple d'expressions ;
- conserver une notion de meilleur match ;
- introduire un `global_score` pour le reranking.

À moyen terme :

- enrichir chaque document avec les triplets `(expression_requete, expression_document, score_similarite)` ;
- ajouter `/doc` pour la consultation détaillée.
