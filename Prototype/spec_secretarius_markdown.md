# Specification du DSL Markdown `secretarius`

Date de reference : 2026-03-17

## Objet

Cette note definit un mini-format Markdown executable pour Secretarius.

Objectif :
- permettre a un document Markdown de porter une intention machine explicite ;
- eviter les ambiguities des commandes implicites dans du texte libre ;
- rester compatible avec les commandes existantes `/index`, `/req`, `/update`, `/exp`.

Principe :
- le bloc `secretarius` porte l'intention ;
- le reste du document porte la matiere documentaire.

## 1. Regles generales

1. Un document executable contient exactement un bloc de code Markdown de type `secretarius`.
2. Le premier bloc `secretarius` est la commande a executer.
3. Si plusieurs blocs `secretarius` sont presents, le document est considere invalide.
4. Si aucun bloc `secretarius` n'est present, le document n'est pas executable au titre de ce DSL.
5. Le contenu du bloc suit un format minimal `cle: valeur`, une ligne par paire.
6. Le YAML complexe n'est pas autorise dans cette premiere version :
   - pas de listes YAML,
   - pas de dictionnaires imbriques,
   - pas de multi-lignes YAML.
7. Tout le texte hors bloc `secretarius` est considere comme `content`.

## 2. Actions supportees

Actions autorisees dans cette premiere version :

- `action: index`
- `action: req`
- `action: update`
- `action: exp`

Correspondance avec l'existant :

- `action: index` <-> `/index`
- `action: req` <-> `/req`
- `action: update` <-> `/update`
- `action: exp` <-> `/exp`

## 3. Grammaire minimale

Exemple de forme generale :

````markdown
```secretarius
action: index
doc_id: doc:exemple-001
type_note: lecture
title: Exemple
tags: histoire, cavalerie
```

Contenu documentaire libre...
````

Regles de parsing :

1. Le parser extrait le premier bloc ````secretarius````.
2. Chaque ligne du bloc contenant `:` est analysee comme `cle: valeur`.
3. Les espaces en tete et fin sont supprimes pour la cle et la valeur.
4. Le texte complet du document, prive du bloc `secretarius`, devient `content`.

## 4. Cles autorisees

### 4.1. Cle commune

- `action` : obligatoire

### 4.2. Pour `action: index`

Cles reconnues :
- `doc_id` : optionnel
- `type_note` : optionnel
- `title` : optionnel
- `tags` : optionnel

Semantique :
- `content` contient le corps documentaire ;
- les cles du bloc sont projetees vers une chaine documentaire compatible avec `index_text`.

### 4.3. Pour `action: update`

Cles reconnues :
- `doc_id` : obligatoire
- `type_note` : optionnel
- `title` : optionnel
- `tags` : optionnel

Semantique :
- `content` contient le corps documentaire corrige ;
- les cles du bloc sont projetees vers une chaine documentaire compatible avec `update_text`.

### 4.4. Pour `action: req`

Cles reconnues :
- `query` : obligatoire
- `top_k` : optionnel, reserve pour un usage futur

Semantique :
- `query` est la requete envoyee a `search_text` ;
- `content` est vide ou ignore.

Recommendation initiale :
- si `content` est non vide pour `req`, retourner une erreur explicite plutot que d'inventer une interpretation.

### 4.5. Pour `action: exp`

Cles reconnues :
- `text` : optionnel

Semantique :
- si `text` est present, il est utilise ;
- sinon, `content` est utilise ;
- si les deux sont vides, retourner une erreur explicite.

## 5. Representation interne cible

Le parser minimal peut produire une structure conceptuelle de cette forme :

```text
{
  "action": "index|req|update|exp",
  "args": { ...cles du bloc... },
  "content": "...texte hors bloc..."
}
```

Cette structure n'est pas un contrat public externe.
Elle decrit seulement le resultat de parsing attendu.

## 6. Mapping recommande vers les commandes existantes

### 6.1. `index`

Le systeme reconstruit un texte compatible avec l'entree actuelle de `/index`, par exemple :

```text
doc_id: ...
type_note: ...
title: ...
#tag1 #tag2

content
```

Puis appelle `index_text`.

### 6.2. `update`

Le systeme reconstruit un texte compatible avec l'entree actuelle de `/update`, puis appelle `update_text`.

### 6.3. `req`

Le systeme appelle `search_text` avec :

```text
query
```

### 6.4. `exp`

Le systeme appelle `extract_expressions` avec le texte retenu.

## 7. Validations minimales

Le parser ou l'adaptateur de canal doit produire une erreur explicite dans les cas suivants :

- plusieurs blocs `secretarius` ;
- absence de `action` ;
- `action` inconnue ;
- `update` sans `doc_id` ;
- `index` sans contenu documentaire exploitable ;
- `update` sans contenu documentaire exploitable ;
- `req` sans `query` ;
- `exp` sans texte exploitable.

Politique initiale recommandee pour les cles inconnues :
- option A : les ignorer silencieusement ;
- option B : retourner une erreur stricte.

Recommendation pour la premiere implementation :
- ignorer les cles inconnues en journalisant leur presence.

## 8. Exemples

### 8.1. Indexation

````markdown
# Cavalerie rouge

```secretarius
action: index
doc_id: doc:boudienny-001
type_note: lecture
title: Cavalerie rouge
tags: URSS, cavalerie
```

Texte de la note...
````

### 8.2. Recherche

````markdown
```secretarius
action: req
query: cavalerie rouge URSS
```
````

### 8.3. Mise a jour

````markdown
# Cavalerie rouge corrige

```secretarius
action: update
doc_id: doc:boudienny-001
type_note: lecture
```

Texte corrige...
````

### 8.4. Extraction

````markdown
```secretarius
action: exp
```

Le regiment de cavalerie progresse vers l'est.
````

## 9. Positionnement architectural

Ce DSL ne remplace pas les commandes directes actuelles.

Il ajoute une entree plus structuree, particulierement adaptee a :
- Memos,
- Obsidian,
- fichiers Markdown,
- carnets ou brouillons documentaires.

Decision recommandee :
- conserver `/index`, `/req`, `/update`, `/exp` pour l'usage direct ;
- ajouter le DSL `secretarius` comme mode structure complementaire.

## 10. Decision de conception

La philosophie retenue est volontairement stricte :

- bloc = intention
- texte = matiere

Cette separation doit rester simple, deterministe et facile a parser.
Le systeme doit eviter toute interpretation heuristique floue quand un bloc `secretarius` est present.
