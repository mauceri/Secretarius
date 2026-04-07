# Canal Memos pour Secretarius

Ce document explique comment relier une instance Memos locale a Secretarius Prototype.

## Principe

Le canal Memos recoit des webhooks emis par Memos quand un memo est cree ou modifie.
Si le contenu du memo commence par une commande supportee (`/exp`, `/index`, `/req`, `/update`), Secretarius traite la demande puis publie la reponse en commentaire du memo source.
Il accepte aussi un bloc Markdown ````secretarius```` conforme a [`Prototype/spec_secretarius_markdown.md`](/home/mauceric/Secretarius/Prototype/spec_secretarius_markdown.md).

Ce canal n'interprete pas les memos ordinaires.
Il ignore aussi les commentaires Memos pour eviter les boucles.

## Preconditions

- Memos tourne deja chez vous, par exemple sur `http://sanroque:5230`
- Secretarius Prototype fonctionne deja localement
- vous disposez d'un token API Memos avec droit de creation de commentaires
- la venv projet existe : `/home/mauceric/Secretarius/.venv`

## Configuration Secretarius

Editez [`Prototype/config.yaml`](/home/mauceric/Secretarius/Prototype/config.yaml) :

```yaml
memos:
  enabled: true
  host: "127.0.0.1"
  port: 8004
  base_url: "http://sanroque:5230"
  access_token: "VOTRE_TOKEN_API_MEMOS"
  webhook_token: "UN_SECRET_LONG_ALEATOIRE"
  ignored_creator: ""
  response_visibility: "PRIVATE"
  journal_file: "logs/memos.log"
  request_timeout_s: 120
  publish_timeout_s: 30
```

Notes :
- `base_url` doit pointer vers votre API Memos ; dans votre cas, `http://sanroque:5230`
- `access_token` est utilise par Secretarius pour poster la reponse dans Memos
- `webhook_token` protege l'endpoint Secretarius ; il est passe dans l'URL du webhook
- `response_visibility` controle la visibilite du commentaire de reponse
- `ignored_creator` peut servir a ignorer un auteur technique particulier si vous en avez besoin
- `base_url` est une URL sortante utilisee par Secretarius pour appeler Memos ; elle peut etre differente de l'adresse utilisee par Memos pour joindre le webhook Secretarius

## Demarrage

Depuis `Prototype` :

```bash
cd /home/mauceric/Secretarius/Prototype
source /home/mauceric/Secretarius/.venv/bin/activate
python main_multicanal.py
```

Le canal Memos expose alors :

```text
http://127.0.0.1:8004/memos/webhook?token=UN_SECRET_LONG_ALEATOIRE
```

Healthcheck :

```text
http://127.0.0.1:8004/health
```

## Configuration Memos

Dans Memos, configurez un webhook utilisateur pointant vers l'URL Secretarius.

URL recommandee :

```text
http://127.0.0.1:8004/memos/webhook?token=UN_SECRET_LONG_ALEATOIRE
```

Evenements utiles :
- memo created
- memo updated

Si Memos tourne dans Docker et Secretarius hors conteneur sur la meme machine, `127.0.0.1` suffit souvent cote Secretarius.
Si Memos doit joindre Secretarius depuis un autre conteneur, utilisez une adresse reseau joignable depuis ce conteneur.

## Utilisation

Deux modes d'entree sont supportes :
- commandes directes `/exp`, `/index`, `/req`, `/update`
- bloc Markdown structure ````secretarius````

Le mode recommande pour Memos est le bloc ````secretarius````, car il est plus robuste et moins ambigu.

Ecrivez un memo dont le contenu commence par une commande supportee.

Exemples :

```text
/req cavalerie rouge #URSS
```

```text
/index
doc_id: note-cavalerie-001
type_note: permanente
title: Cavalerie rouge
#URSS #cavalerie
Notes sur l'organisation de la cavalerie sovietique.
```

```text
/update
doc_id: note-cavalerie-001
type_note: permanente
title: Cavalerie rouge
#URSS #cavalerie
Version corrigee de la note.
```

```text
/exp Le regiment de cavalerie progresse vers l'est.
```

Comportement :
- Secretarius recoit le webhook
- la commande est envoyee au guichet `memos`
- la reponse est publiee en commentaire du memo source

## Utilisation avec le DSL Markdown `secretarius`

Le memo peut aussi contenir un bloc de code Markdown `secretarius`.

Principe :
- le bloc `secretarius` porte l'intention executable
- le reste du memo porte la matiere documentaire

Actions supportees :
- `action: index`
- `action: req`
- `action: update`
- `action: exp`

Exemples :

Indexation :

````markdown
# Cavalerie rouge

```secretarius
action: index
doc_id: doc:boudienny-001
type_note: lecture
title: Cavalerie rouge
tags: URSS, cavalerie
```

Notes sur l'organisation de la cavalerie sovietique.
````

Recherche :

````markdown
```secretarius
action: req
query: cavalerie rouge URSS
```
````

Mise a jour :

````markdown
# Cavalerie rouge corrige

```secretarius
action: update
doc_id: doc:boudienny-001
type_note: lecture
```

Version corrigee de la note.
````

Extraction :

````markdown
```secretarius
action: exp
```

Le regiment de cavalerie progresse vers l'est.
````

Regles importantes :
- un seul bloc `secretarius` par memo
- pour `index` et `update`, le texte hors bloc devient le contenu documentaire
- pour `req`, la requete vient du champ `query`
- pour `update`, `doc_id` est obligatoire
- un bloc invalide est rejete avec une erreur webhook

## Comportement et limites

- seuls les memos commencant par `/exp`, `/index`, `/req` ou `/update` sont traites en mode direct
- les memos contenant un bloc ````secretarius```` valide sont aussi traites
- les commentaires Memos sont ignores pour eviter une boucle de reponse
- si la publication du commentaire echoue, le webhook est repondu en erreur HTTP
- le journal du canal est ecrit dans [`Prototype/logs/memos.log`](/home/mauceric/Secretarius/Prototype/logs/memos.log)

## Test manuel rapide

1. lancer `python main_multicanal.py`
2. verifier `http://127.0.0.1:8004/health`
3. creer dans Memos un memo avec :

```text
/req cavalerie rouge
```

Ou avec le DSL Markdown :

````markdown
```secretarius
action: req
query: cavalerie rouge
```
````

4. verifier qu'un commentaire Secretarius apparait sous le memo
5. verifier les traces dans [`Prototype/logs/memos.log`](/home/mauceric/Secretarius/Prototype/logs/memos.log)
