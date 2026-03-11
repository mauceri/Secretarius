# Proposition d'evolution de l'indexation

## Objet
Ce document propose une evolution du pipeline d'indexation de `Secretarius` pour mieux gerer :

- les notes informelles,
- les fragments de texte courts,
- les textes plus structures avec titre, tags, metadata,
- la tracabilite entre expressions extraites et texte source,
- la recherche semantique avec retour de contexte utile.

Il inclut aussi une proposition de module tres leger de reconnaissance de la morphologie des textes soumis, avec une option basee sur un micro-modele de type Qwen, mais sans rendre le systeme dependant d'un gros LLM.

## Constat sur l'etat actuel

### Ce que le systeme fait deja bien
- `index_text` extrait des expressions caracteristiques puis les pousse dans Milvus.
- La recherche semantique sur expressions fonctionne deja de maniere utile.
- Le schema local `secretarius.document.v0.1` sait deja porter :
  - `doc_id`,
  - `source`,
  - `content`,
  - `user_fields`,
  - `derived.expressions`,
  - des metadonnees d'indexation.

### Limites actuelles
- Quand l'utilisateur soumet un simple `text`, on n'indexe pas vraiment un document parent riche.
- En pratique, le systeme fabrique surtout des `snippet` auto-generes, un par expression.
- Les resultats de recherche renvoient donc des fragments isoles, sans lien fort vers un texte source parent.
- On perd la difference entre :
  - une note rapide,
  - un extrait litteraire,
  - un document structure avec titre et tags,
  - une citation,
  - un brouillon.
- Il n'existe pas aujourd'hui de couche explicite de "morphologie documentaire".

## Reponse courte a votre question
Oui, il manque aujourd'hui deux choses structurantes :

1. un document parent stable, avec `doc_id`, qui represente le texte soumis ;
2. un lien explicite entre les expressions/snippets indexes et ce document parent.

Sans cela, la recherche renvoie des morceaux semantiquement proches, mais pas une memoire documentaire vraiment exploitable.

## Principe directeur recommande
Le pipeline doit indexer **d'abord un document parent**, puis indexer **des vues derivees** de ce document :

- document parent,
- expressions extraites,
- snippets derives,
- embeddings associes.

Autrement dit : on ne doit plus considerer l'expression comme l'unite primaire du systeme. L'unite primaire doit etre le document soumis, meme s'il est tres court et informel.

## Typologie de textes a viser
Le systeme doit fonctionner d'abord sur des cas simples et frequents.

### Classe A : fragment informel
Exemples :
- pensee rapide,
- note de travail,
- phrase isolee,
- brouillon,
- citation collee sans contexte,
- bout de texte libre.

Caracteristiques :
- pas de titre,
- pas de structure explicite,
- longueur courte ou moyenne,
- metadata absentes.

### Classe B : note semi-structuree
Exemples :
- note avec un titre implicite,
- petite fiche,
- note avec tags,
- morceau de journal,
- bloc de texte avec ligne de tete identifiable.

Caracteristiques :
- eventuel titre,
- eventuels mots-cles,
- parfois listes ou sections courtes.

### Classe C : document structure
Exemples :
- article,
- page web,
- note formatee,
- fiche de lecture,
- texte avec auteur, date, source, tags.

Caracteristiques :
- titre probable,
- structure plus nette,
- metadata exploitables.

## Ce qu'il faut stocker

### 1. Document parent
Chaque soumission a l'indexation doit produire un document parent canonique.

Proposition :

```json
{
  "schema": "secretarius.document.v0.2",
  "doc_id": "doc:...",
  "type": "note|fragment|poem_excerpt|article|quote|other",
  "morphology": {
    "class": "fragment_informel|note_semi_structuree|document_structure",
    "confidence": 0.0,
    "method": "heuristic|micro_llm|hybrid"
  },
  "source": {
    "source_id": "src:...",
    "url": null,
    "canonical_url": null,
    "authors": [],
    "origin": "manual_input|import|web|api"
  },
  "content": {
    "mode": "inline",
    "text": "...",
    "hash": "sha256:...",
    "length_chars": 1234,
    "language": "fr"
  },
  "structure": {
    "title": null,
    "subtitle": null,
    "headings": [],
    "keywords": [],
    "has_lists": false,
    "has_stanzas": false
  },
  "user_fields": {
    "tags": [],
    "keywords": [],
    "status": "draft",
    "created_at": "...",
    "updated_at": "..."
  },
  "derived": {
    "expressions": [],
    "chunks": []
  },
  "indexing": {
    "pipeline_version": "v0.2",
    "state": "done",
    "errors": []
  }
}
```

### 2. Objets derives lies au parent
Au lieu d'inserer seulement des snippets anonymes, chaque expression derivee doit pointer vers le parent.

Proposition minimale :

```json
{
  "schema": "secretarius.derived_expression.v0.1",
  "derived_id": "expr:...",
  "parent_doc_id": "doc:...",
  "content": {
    "text": "trou de verdure"
  },
  "derived": {
    "expression": "trou de verdure",
    "norm": "trou de verdure",
    "span": [12, 28],
    "weight": 0.81
  },
  "source_context": {
    "preview": "C'est un trou de verdure ou chante une riviere..."
  }
}
```

### 3. Snippets contextuels eventuels
Si vous voulez garder une granularite fine, il faut indexer non seulement l'expression, mais aussi un petit contexte de quelques phrases ou quelques vers autour d'elle.

Cela donne de meilleurs resultats d'affichage que l'expression seule.

## Recommendation centrale
Indexer en parallelle deux niveaux :

- **niveau parent** : le texte complet ou la note complete ;
- **niveau derive** : expressions et snippets contextuels.

Puis, lors d'une recherche :

1. on interroge les derives ;
2. on regroupe les hits par `parent_doc_id` ;
3. on affiche d'abord le document parent ;
4. on montre ensuite les expressions/snippets qui ont servi de preuve.

Cela change completement l'utilite du resultat.

## Proposition d'architecture de pipeline

### Etape 1 : normalisation d'entree
Transformer toute entree en document parent canonique.

Cas 1 :
- l'utilisateur fournit un vrai document structure ;
- on preserve les metadata existantes.

Cas 2 :
- l'utilisateur fournit un simple texte brut ;
- on fabrique un document parent minimal.

Regle :
- meme une note tres informelle devient un document parent ;
- on n'indexe jamais directement une simple liste d'expressions sans parent.

### Etape 2 : analyse morphologique legere
But :
- reconnaitre la forme probable du texte ;
- extraire uniquement quelques metadonnees simples ;
- choisir une strategie de derivation adaptee.

Sorties attendues :
- classe morphologique,
- titre probable ou absence de titre,
- mots-cles probables,
- detecteurs simples : liste, poemes/vers, citation, note, document structure.

### Etape 3 : extraction d'expressions
Conserver l'extraction existante, mais la rattacher au parent.

Chaque expression extraite doit porter :
- `parent_doc_id`,
- `span` si disponible,
- `chunk_id` si disponible,
- une petite fenetre de contexte.

### Etape 4 : derivation de snippets
Generer des snippets contextuels seulement si utile :
- autour des expressions saillantes,
- ou par chunk semantique.

Pour des notes tres courtes, ce n'est pas necessaire.
Pour des textes plus longs, c'est tres utile.

### Etape 5 : embeddings et indexation
Indexer au minimum :
- les derives `expression`,
- les derives `snippet`,
- eventuellement le document parent lui-meme.

Recommandation :
- collection unique possible au debut, avec un champ `record_type`,
- mais logique de regroupement obligatoire par `parent_doc_id`.

## Ce qu'il faut changer concretement dans le schema local

### Evolution minimale, compatible avec l'existant
Je recommande une transition legere, pas une refonte brutale.

#### A. Conserver `secretarius.document.v0.1` a court terme
Mais ajouter progressivement :
- `morphology`,
- `structure`,
- `content.language`,
- `source.origin`,
- `derived[*].parent_doc_id` pour les objets derives.

#### B. Modifier l'auto-generation de snippets
Aujourd'hui, `semantic_graph_search` fabrique des snippets auto-generes depourvus de vrai parent quand il n'a qu'une liste d'expressions.

Il faudrait que `index_text` :
- cree d'abord un parent document minimal,
- enrichisse ce parent,
- puis cree des derives rattachables a ce parent.

#### C. Faire de `index_text` un vrai pipeline documentaire
Le pipeline cible devient :

1. `text` ou `document` recu ;
2. creation/normalisation du parent ;
3. analyse morphologique legere ;
4. extraction ;
5. creation de derives relies au parent ;
6. embedding ;
7. insertion ;
8. retour d'un resume avec `doc_id`, `parent_doc_id`, nombres de derives inseres.

## Comment differencier note informelle et note structuree
Oui, il faut une reconnaissance de morphologie, mais elle doit rester tres legere.

## Recommandation de design pour la morphologie

### Option recommandee : pipeline hybride
Le meilleur compromis n'est pas "tout heuristique" ni "tout LLM".

Je recommande :

1. **heuristiques tres bon marche** en premier niveau ;
2. **micro-modele** seulement pour les cas ambigus ;
3. **fallback sans blocage** si le micro-modele echoue.

### Niveau 1 : heuristiques locales
Exemples de signaux peu couteux :
- longueur en caracteres,
- nombre de lignes,
- taux de lignes courtes,
- presence d'une premiere ligne courte suivie d'un blanc,
- presence de puces,
- presence de doubles retours ligne,
- presence de metadata visibles (`Titre:`, `Tags:`, `Auteur:`),
- presence de strophes ou de vers courts,
- ratio ponctuation / mots,
- densite de majuscules ou de dates.

Sorties possibles :
- `fragment_informel`,
- `note_semi_structuree`,
- `document_structure`,
- `poetic_or_verse_like`,
- `list_like`,
- `metadata_rich`.

Avantages :
- tres rapide,
- deterministic,
- facile a tester,
- zero dependance GPU.

### Niveau 2 : micro-modele de desambiguation
Le micro-modele n'intervient que si les heuristiques sont indecises.

Tache tres simple proposee :
- classifier la morphologie ;
- proposer un titre seulement si evident ;
- proposer quelques keywords seulement si elles sont manifestes.

Sortie JSON stricte :

```json
{
  "morphology_class": "fragment_informel",
  "has_title": false,
  "title": null,
  "keywords": [],
  "confidence": 0.78
}
```

### Pourquoi cette approche est preferable
- Elle limite le nombre d'appels LLM.
- Elle garde un comportement stable.
- Elle evite de sur-interpreter des notes courtes.
- Elle est compatible avec une machine locale modeste.

## Faut-il utiliser un micro-modele Qwen ?
Oui, possiblement, mais pas comme premier reflexe.

## Position recommandee
Un micro-Qwen peut etre utile pour la **desambiguation morphologique**, pas pour toute la chaine.

Il ne faut pas lui demander :
- de faire l'extraction complete,
- de resumer le document,
- de reconstruire la structure profonde,
- de decider seul de toutes les metadata.

Il faut lui demander une tache **petite, bornee, testable**.

## Pourquoi Qwen est une option credible
D'apres les sources officielles disponibles :
- Qwen3 existe en petites tailles, notamment `0.6B` et `1.7B`, avec support multilingue et bon rapport capacite/cout ;
- la famille Qwen met en avant un mode "thinking" activable/desactivable, utile pour garder un comportement court et contraint ;
- les modeles Qwen sont presentes comme adaptes a des usages d'instruction, d'agent et de tool use, ce qui est proche d'une classification JSON simple.

Observation pratique :
- pour une tache de morphologie documentaire en JSON court, `Qwen3:0.6B` peut etre suffisant comme second niveau ;
- si vous constatez trop d'instabilite sur les sorties JSON, `Qwen3:1.7B` sera probablement un meilleur compromis ;
- il ne faut pas confier a `0.6B` des decisions trop semantiques ou editoriales.

## Ce que le micro-modele devrait faire exactement
Proposition de contrat strict :

- entree : texte brut ;
- sortie : petit JSON ferme ;
- classes fermees ;
- aucune prose ;
- pas plus de 5 keywords ;
- titre seulement si explicitement detectable.

Exemple :

```json
{
  "morphology_class": "note_semi_structuree",
  "document_kind": "note",
  "title": "Le Dormeur du val",
  "keywords": ["soldat", "nature", "mort"],
  "confidence": 0.84
}
```

## Ce qu'il ne faut pas faire
- Ne pas lancer le micro-modele pour tous les textes.
- Ne pas en faire une brique obligatoire de l'indexation.
- Ne pas ecrire les tags ou mots-cles proposes comme verite source.
- Ne pas laisser le modele inventer auteur, date, genre ou provenance.

## Alternatives encore plus legeres
Si votre vrai objectif est seulement de distinguer :
- fragment,
- note,
- document structure,

alors un petit classifieur non generatif peut etre encore meilleur.

### Option 1 : heuristiques seules
C'est probablement suffisant pour une V1.

### Option 2 : fastText ou classifieur lineaire tres compact
Pour une tache de classification courte et fermee, un petit modele supervise de type fastText ou equivalent peut etre plus robuste, plus rapide et plus simple a tester qu'un micro-LLM.

### Conclusion sur ce point
Je ne pense pas qu'il faille commencer par "un module LLM". Je pense qu'il faut commencer par :

1. heuristiques,
2. puis eventuellement micro-Qwen uniquement sur les cas ambigus,
3. et garder ouverte la possibilite de remplacer ce niveau 2 plus tard par un classifieur specialise plus petit.

## Proposition de schema de relations

### Document parent
- `doc_id`
- `morphology.class`
- `structure.title`
- `content.text`

### Expression derivee
- `derived_id`
- `parent_doc_id`
- `expression`
- `span`
- `context_preview`

### Snippet derive
- `snippet_id`
- `parent_doc_id`
- `chunk_id`
- `text`
- `start`
- `end`

### Search result ideal
La recherche devrait renvoyer un objet de ce type :

```json
{
  "status": "ok",
  "tool": "search_text",
  "query": "trou de verdure",
  "results": [
    {
      "parent_doc_id": "doc:abc",
      "score_max": 0.92,
      "document": {
        "type": "poem_excerpt",
        "title": null,
        "text_preview": "C'est un trou de verdure ou chante une riviere..."
      },
      "matches": [
        {
          "record_type": "expression",
          "text": "trou de verdure",
          "score": 0.92
        },
        {
          "record_type": "snippet",
          "text": "C'est un trou de verdure ou chante une riviere",
          "score": 0.81
        }
      ]
    }
  ]
}
```

Ce format est bien plus utile que 10 snippets isoles.

## Strategie d'implementation par etapes

### Etape 1 : corrigible sans rupture
- Creer un document parent minimal meme pour `text`.
- Lui donner un `doc_id` stable.
- Propager `doc_id` dans tous les derives.
- Ajouter `parent_doc_id` aux snippets/expressions indexes.

Impact :
- faible,
- gain fort de tracabilite.

### Etape 2 : enrichissement morphologique leger
- Ajouter un module `text_morphology.py`.
- Commencer par heuristiques pures.
- Sortie JSON locale et testable.

Impact :
- faible a moyen,
- pas de dependance lourde.

### Etape 3 : mode hybride
- Si heuristiques ambiguës, appeler un micro-modele.
- Stocker `method` et `confidence`.

Impact :
- moyen,
- optionnel.

### Etape 4 : regroupement des resultats de recherche
- Regrouper les hits par `parent_doc_id`.
- Afficher le parent et les preuves.

Impact :
- fort benefice utilisateur.

## Proposition de module `text_morphology.py`

### API proposee

```python
def analyze_text_morphology(text: str) -> dict:
    return {
        "class": "fragment_informel",
        "document_kind": "note",
        "title": None,
        "keywords": [],
        "signals": {
            "line_count": 8,
            "has_blank_lines": True,
            "looks_like_poetry": True
        },
        "confidence": 0.82,
        "method": "heuristic"
    }
```

### Regles simples de V1
- Si texte tres court et sans structure : `fragment_informel`
- Si premiere ligne courte et reste du texte plus long : `note_semi_structuree`
- Si nombreuses lignes courtes et retours ligne reguliers : `poetic_or_verse_like`
- Si presence de labels type `Titre:`, `Tags:`, `Auteur:` : `document_structure`
- Si puces ou numerotation importante : `list_like`

### Role exact de cette couche
Elle ne decide pas du sens du texte.
Elle decide seulement de sa **forme documentaire probable**.

## Ce qu'il faut afficher au retour de `index_text`
Le retour de `index_text` devrait evoluer vers :

```json
{
  "status": "ok",
  "tool": "index_text",
  "summary": {
    "doc_id": "doc:...",
    "morphology_class": "fragment_informel",
    "expressions_count": 12,
    "snippets_count": 4,
    "inserted_count": 16,
    "collection_name": "secretarius_semantic_graph"
  }
}
```

## Ce qu'il faut afficher au retour de `search_text`
Le retour utilisateur devrait privilegier :
- preview du texte parent,
- titre si disponible,
- expressions/snippets matches,
- score max par document,
- eventuellement deduplication de plusieurs snippets provenant du meme parent.

## Risques a anticiper

### 1. Sur-interpretation des notes informelles
Le systeme pourrait inventer des titres ou des tags.
Mitigation :
- ne remplir que les champs a forte confiance ;
- sinon laisser `null` ou liste vide.

### 2. Surcout de pipeline
Si vous ajoutez trop d'etapes, l'indexation devient lourde.
Mitigation :
- heuristiques d'abord ;
- micro-modele uniquement si necessaire.

### 3. Complexite Milvus
La relation parent/enfant n'est pas native comme dans une base graphe/document.
Mitigation :
- porter explicitement `parent_doc_id` dans les payloads JSON indexes ;
- faire le regroupement applicatif cote Python.

### 4. Ambiguite entre "expression", "snippet" et "document"
Mitigation :
- ajouter un champ `record_type` explicite dans chaque payload indexe.

## Recommendation finale
Je recommande fortement de faire evoluer l'indexation dans ce sens :

1. **Le parent documentaire devient l'unite primaire.**
2. **Les expressions et snippets deviennent des derives lies a ce parent.**
3. **La morphologie documentaire est geree par une couche legere, d'abord heuristique.**
4. **Un micro-Qwen n'est utile qu'en second niveau, pour desambiguïser, pas pour piloter toute la chaine.**
5. **La recherche doit regrouper les hits par document parent et non afficher seulement des fragments isoles.**

## Plan de mise en oeuvre concret recommande

### Sprint 1
- Creer un parent document minimal pour tout `text`.
- Ajouter `parent_doc_id` aux derives indexes.
- Ajouter `record_type`.
- Modifier `search_text` pour regrouper les resultats par parent.

### Sprint 2
- Ajouter `text_morphology.py` heuristique.
- Stocker `morphology.class`, `method`, `confidence`.
- Remonter ce resume dans `index_text`.

### Sprint 3
- Ajouter un mode optionnel `micro_llm` pour cas ambigus.
- Limiter a classification JSON stricte.
- Evaluer `Qwen3:0.6B`, puis `1.7B` si `0.6B` est trop instable.

## Observations issues de la recherche web

### Ce que confirment les sources
- La famille Qwen3 existe bien en petites tailles et vise explicitement des usages d'instruction/agent ; cela rend plausible son emploi comme classifieur JSON local de second niveau.
- Les docs officielles Qwen mettent en avant le controle du mode "thinking", utile pour des sorties courtes et bornees.
- Le modele d'embedding deja utilise, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, est bien un modele oriente recherche semantique multilingue, ce qui valide votre base actuelle pour la similarite.
- Pour une classification de texte fermee et tres legere, des approches specialisees comme fastText restent tres competitives en cout et simplicite.

### Interpretation pratique pour Secretarius
- Pour la recherche semantique : votre choix d'embeddings est coherent.
- Pour la morphologie documentaire : commencer par un micro-LLM serait premature si vous n'avez pas d'abord un schema parent/enfant propre.
- Pour la V1 : heuristiques > micro-LLM.
- Pour la V2 : hybride heuristiques + micro-Qwen sur cas ambigus.

## Sources
- Qwen3 official README / model family:
  - https://github.com/QwenLM/Qwen3
- Qwen on Hugging Face:
  - https://huggingface.co/Qwen
- Sentence-Transformers model card, `paraphrase-multilingual-MiniLM-L12-v2`:
  - https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- fastText official documentation:
  - https://fasttext.cc/docs/en/supervised-tutorial.html
