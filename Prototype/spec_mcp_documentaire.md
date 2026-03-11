# Specification MCP Documentaire

Date de reference : 2026-03-11

## Objet

Cette note fixe la separation des responsabilites entre :
- les fonctions metier internes de Secretarius,
- les outils MCP exposes au routeur,
- les structures de donnees manipulees lors de l'indexation et de l'interrogation documentaire.

L'objectif est de supprimer le melange actuel entre :
- extraction d'expressions,
- analyse documentaire,
- indexation vectorielle,
- parametres techniques de backend,
- schemas MCP trop riches ou trompeurs.

## 1. Principes d'architecture

1. Les fonctions metier ne se confondent pas avec les outils MCP.
2. Les outils MCP sont des interfaces d'orchestration minimales exposees au routeur.
3. Les schemas MCP doivent refleter l'intention utilisateur, pas l'implementation interne.
4. L'extraction d'expressions travaille sur un texte brut et ne suppose aucun statut documentaire.
5. L'indexation et l'interrogation travaillent sur une chaine fournie par l'utilisateur, interpretee comme document potentiel.
6. L'analyse documentaire est une etape interne utilisee par l'indexation et l'interrogation, pas par l'outil d'extraction d'expressions.
7. Les details techniques de backend comme `llama_url`, `model`, structure Milvus ou pseudo-document interne ne doivent pas polluer les schemas MCP exposes au routeur, sauf necessite absolue.

## 2. Fonctions metier internes

### 2.1. Fonction d'extraction d'expressions

Signature conceptuelle :

```text
extraire_expressions(texte: str) -> list[str]
```

Role :
- prendre en entree une chaine de caracteres,
- extraire les expressions caracteristiques presentes dans ce texte,
- renvoyer uniquement une liste d'expressions.

Contraintes :
- cette fonction ignore la notion de document,
- elle ne depend pas de la base de donnees,
- elle ne construit pas de structure documentaire,
- elle n'indexe rien,
- elle ne recherche rien.

### 2.2. Fonction d'analyse documentaire

Signature conceptuelle :

```text
analyser_texte_documentaire(texte: str) -> Document
```

Role :
- prendre en entree une chaine representant un document potentiel,
- en extraire la structure documentaire implicite,
- produire un objet `Document` canonique.

Contenu attendu de `Document` dans cette premiere phase :
- texte principal,
- titre eventuel,
- mots-cles eventuels,
- date eventuelle,
- URL eventuelle,
- identifiant documentaire si la strategie retenue le justifie,
- eventuellement un champ `expressions` vide ou absent a ce stade.

Remarque :
- un champ `expressions` dans `Document` est juge utile pour eviter de faire circuler separement le texte, les expressions et le document au cours du pipeline.

### 2.3. Fonction de calcul de plongements

Signature conceptuelle :

```text
calculer_plongements(expressions: list[str]) -> list[vector]
```

Role :
- prendre une liste d'expressions,
- calculer les plongements associes,
- renvoyer la liste des vecteurs.

Contraintes :
- cette fonction ne connait ni l'indexation documentaire ni la recherche,
- elle ne recoit pas un document en entree,
- elle ne decide pas quoi inserer dans la base.

### 2.4. Fonction d'indexation

Signature conceptuelle :

```text
indexer_document(texte_documentaire: str) -> ResultatIndexation
```

Pipeline :
1. analyser la chaine avec `analyser_texte_documentaire`,
2. obtenir un `Document`,
3. extraire les expressions caracteristiques a partir du texte du `Document`,
4. stocker ces expressions dans `Document.expressions` si ce champ existe,
5. calculer les plongements a partir de ces expressions,
6. inserer dans la base le `Document`, enrichi des informations utiles au pipeline.

Role :
- indexer un document, pas une simple liste d'expressions.

Point essentiel :
- ce qui est insere dans la base est le `Document`,
- les plongements sont calcules a partir des expressions du texte de ce `Document`,
- les expressions ne sont pas elles-memes le document.

Sortie minimale possible :
- succes / echec,
- nombre d'expressions extraites,
- eventuellement la liste des expressions,
- message ou warning si necessaire.

### 2.5. Fonction d'interrogation

Signature conceptuelle :

```text
interroger_documents(texte_requete: str) -> ResultatRecherche
```

Pipeline initial :
1. analyser la chaine fournie comme document potentiel ou requete documentaire,
2. obtenir un `Document` de requete,
3. extraire les expressions du texte de ce document,
4. calculer les plongements de ces expressions,
5. utiliser ces plongements pour rechercher des documents similaires dans la base.

Role :
- rechercher dans la base les documents similaires au document decrit par la chaine d'entree.

Premiere phase :
- la similarite repose sur les plongements des expressions caracteristiques extraites du texte,
- l'usage precis des mots-cles, dates, URL et autres metadonnees pourra etre ajoute plus tard.

## 3. Outils MCP

### 3.1. Outil MCP d'extraction d'expressions

Nom conceptuel :

```text
extract_expressions
```

Entree :

```text
text: str
```

Role :
- fournir la liste des expressions caracteristiques contenues dans une chaine de caracteres,
- sans supposer qu'il s'agit d'un document,
- sans interaction avec la base.

Sortie normale :
- liste d'expressions,
- eventuellement warning si necessaire.

Cet outil ne doit pas :
- exposer un schema documentaire interne,
- exposer des parametres techniques de backend au routeur sans necessite,
- melanger extraction et indexation.

### 3.2. Outil MCP d'indexation

Nom conceptuel :

```text
index_text
```

Entree :

```text
text: str
```

Interpretation :
- la chaine contient un document potentiellement structure par l'utilisateur,
- avec eventuellement mots-cles, date, URL, titre, etc.

Role :
- appeler la fonction metier d'indexation,
- indexer dans la base le document correspondant a cette chaine.

Sortie attendue :
- indicateur de succes ou d'echec,
- informations de deroulement utiles,
- eventuellement nombre ou liste des expressions extraites,
- warning eventuel.

Cet outil ne doit pas :
- exposer au routeur la structure interne complete du `Document`,
- demander a l'utilisateur des objets documentaires techniques,
- exposer inutilement les parametres internes de backend.

### 3.3. Outil MCP d'interrogation

Nom conceptuel :

```text
search_text
```

Entree :

```text
query: str
```

ou eventuellement `text: str` si une harmonisation future est decidee.

Role :
- recevoir une chaine semblable a celle utilisee pour l'indexation,
- appeler la fonction metier d'interrogation,
- rechercher dans la base les documents similaires.

Premiere phase :
- l'interrogation exploite principalement les expressions du texte,
- les metadonnees documentaires seront prises en compte plus tard.

Sortie attendue :
- liste de documents trouves,
- scores ou equivalent,
- warning eventuel,
- informations minimales de deroulement si necessaire.

## 4. Structure documentaire canonique

Le `Document` est une structure interne metier.
Il ne doit pas etre confondu avec le schema minimal d'entree des outils MCP.

Proprietes conceptuelles possibles :
- `doc_id`
- `text`
- `title`
- `keywords`
- `date`
- `url`
- `expressions`
- `embeddings_ref` ou equivalent si necessaire
- metadonnees de pipeline si vraiment utiles en interne

Regle importante :
- une seule structure canonique doit porter l'identite documentaire,
- eviter les doublons du type `id` / `doc_id`, `texte` / `content.text`, `mots_clefs` / `keywords`, `url` / `source.url`.

## 5. Separation stricte des niveaux

Ne doivent pas apparaitre dans les schemas MCP orientes routeur sauf necessite forte :
- `llama_url`
- `llama_cpp_url`
- `backend`
- structure Milvus
- structure interne complete du `Document`
- etats internes d'indexation
- details de chunking ou de fingerprint, hors debug

Ces elements relevent :
- soit de l'infrastructure,
- soit du debug,
- soit de l'implementation interne.

## 6. Consequences pratiques

1. `extract_expressions` doit rester un outil simple centre sur le texte.
2. `index_text` doit traiter un texte documentaire et inserer un `Document` en base.
3. `search_text` doit traiter une requete documentaire et rechercher des `Document`.
4. Le routeur MCP doit voir des outils simples et clairement distincts.
5. Le schema d'entree MCP ne doit pas exposer un faux document technique non stabilise.
6. Les structures internes riches doivent rester internes au pipeline.

## 7. Decisions retenues au 2026-03-11

- L'extraction d'expressions est independante de la base.
- L'analyse documentaire est distincte de l'extraction d'expressions.
- L'indexation utilise l'analyse documentaire puis l'extraction d'expressions.
- L'interrogation suit le meme principe general que l'indexation, mais au lieu d'inserer, elle recherche par similarite.
- Les outils MCP sont des facades minces.
- Le `Document` est une structure metier interne canonique, pas une entree brute a imposer au routeur ou a l'utilisateur.
