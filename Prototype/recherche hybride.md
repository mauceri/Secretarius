# Plan Recherche Hybride Avec Operateurs De Mots-Cles

## Resume
Etendre la recherche documentaire pour dupliquer les `keywords` dans Milvus par expression et permettre une recherche hybride semantique + filtrage lexical. Le contrat MCP public reste inchangé: `search_text(query)` continue de prendre une simple chaîne. Les operateurs utilisateur sur hashtags sont interpretes directement dans `query`:
- `#mot` : mot-clé optionnel en groupe `OR`
- `+#mot` : mot-clé obligatoire (`AND`)
- `-#mot` : mot-clé exclu (`NOT`)

## Changements d'implementation
- Etendre les lignes Milvus dans `secretarius_local/semantic_graph.py` pour stocker explicitement :
  - `keywords`: liste dedupee issue de `document.user_fields.keywords`
  - `type_note`, `doc_id`, `expression_id`, `expression_norm`: inchanges
  - `payload_json`: conserve pour restituer la note complete
- Ajouter un parseur interne de requête hashtags dans la couche recherche :
  - extraire trois groupes depuis `query`: `optional_keywords`, `required_keywords`, `excluded_keywords`
  - normaliser chaque mot-clé au format existant `#mot`
  - retirer les opérateurs du texte libre avant extraction sémantique des expressions
- Définir la sémantique de requête :
  - texte libre seul : recherche vectorielle seule
  - hashtags nus `#mot` : filtre lexical optionnel en `OR`
  - hashtags `+#mot` : filtre lexical obligatoire en `AND`
  - hashtags `-#mot` : exclusion en `NOT`
  - combinaison :
    - les `required_keywords` doivent tous matcher
    - aucun `excluded_keyword` ne doit matcher
    - les `optional_keywords` s’appliquent en `OR` si au moins un est présent
- Étendre `semantic_graph_search_milvus(...)` pour accepter un filtre interne structuré, par exemple :
  - `required_keywords: list[str] | None = None`
  - `optional_keywords: list[str] | None = None`
  - `excluded_keywords: list[str] | None = None`
- Construire le filtre Milvus à partir de ces groupes :
  - `required`: conjonction `AND`
  - `optional`: disjonction `OR`
  - `excluded`: négation `NOT`
  - si seuls des `excluded_keywords` existent, appliquer uniquement l’exclusion sans bloquer la recherche sémantique
- Adapter `search_documents_by_text(...)` dans `secretarius_local/document_pipeline.py` :
  - parser la requête utilisateur
  - envoyer à l’extracteur seulement le texte libre nettoyé des hashtags/opérateurs
  - transmettre les groupes de keywords à Milvus
- Conserver le reranking applicatif dans `secretarius_local/mcp_server.py` :
  - `keyword_matches` reste calculé sur la note restituée
  - `title_matches` reste inchangé
  - `combined_score` reste utilisé pour ordonner les documents après filtrage
- Ne pas changer le contrat MCP public :
  - `tools/list` reste minimal
  - `search_text` garde uniquement `query`
  - aucun opérateur n’est ajouté au schéma JSON, seulement dans la syntaxe texte de `query`

## Exemples de requêtes cibles
- `memoire autobiographique`
  - recherche sémantique seule
- `memoire autobiographique #psychologie #trauma`
  - recherche sémantique + filtre Milvus `(#psychologie OR #trauma)`
- `memoire autobiographique +#psychologie +#trauma`
  - recherche sémantique + filtre Milvus `(#psychologie AND #trauma)`
- `memoire autobiographique #psychologie -#brouillon`
  - recherche sémantique + filtre Milvus `(#psychologie) AND NOT (#brouillon)`
- `jung +#psychologie #symbolisme -#brouillon`
  - recherche sémantique sur `jung`
  - `#psychologie` requis
  - `#symbolisme` optionnel
  - `#brouillon` exclu

## Tests
- Ajouter dans `tests/test_semantic_graph.py` :
  - vérification que `_build_row(...)` stocke bien `keywords`
  - déduplication et nettoyage des keywords invalides
  - construction correcte des filtres Milvus pour `OR`, `AND`, `NOT` et leurs combinaisons
- Ajouter dans `tests/test_document_pipeline.py` :
  - parsing correct de `#mot`, `+#mot`, `-#mot`
  - séparation correcte entre texte libre et opérateurs
  - absence de régression quand la requête ne contient aucun hashtag
- Ajouter ou adapter dans `tests/test_mcp_server_compact_responses.py` :
  - `search_text` continue de renvoyer des documents compacts complets
  - `keyword_query_count` reflète le nombre total de hashtags utiles dans la requête
- Garder `tests/test_mcp_tools_catalog.py` inchangé et vert pour garantir l’absence d’évolution du contrat MCP public

## Assumptions
- Les mots-clés restent stockés sous forme de hashtags normalisés dans `user_fields.keywords`.
- Les hashtags nus `#mot` sont traités comme un groupe `OR` par défaut, pour éviter un filtrage trop agressif.
- Les opérateurs s’appliquent uniquement aux hashtags; aucun parseur booléen général sur mots libres n’est introduit.
- Aucune migration automatique des données Milvus existantes n’est incluse; une réindexation sera nécessaire pour que les anciennes notes bénéficient du filtrage par keywords.
- `payload_json` reste conservé dans Milvus pour reconstruire les notes renvoyées sans ajouter une seconde base documentaire dans ce lot.
