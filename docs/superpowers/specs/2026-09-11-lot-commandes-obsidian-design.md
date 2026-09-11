# Exécution en lot de commandes wiki depuis une note Obsidian — conception

## Contexte et motivation

Aujourd'hui, une commande `/` wiki (`/c`, `/q`, `/ingest`…) se tape une par
une, sur Telegram ou via le plugin `wikilm-capture` (qui ne fait que
capturer la note ouverte). L'utilisateur veut pouvoir préparer une **liste**
de commandes dans une note Obsidian — par exemple une session de captures
groupées, ou une séquence capture+ingestion+vérification — et les lancer
toutes d'un geste, avec le résultat de chacune visible directement dans la
note.

Périmètre de cette conception : **commandes wiki uniquement** (`kind:
"wiki"` dans `dispatch.ts`) — pas gog, pas scout. Les commandes
destructrices (`/supprimer`, seule à ce jour) sont explicitement exclues du
lot ; elles gardent leur flux normal (essai à blanc + `/confirm`), qui
n'a pas de sens en exécution automatique non supervisée.

## Format dans la note

Chaque commande est un bloc de code délimité par ` ```wiki ` / ` ``` ` :

````
```wiki
/c
#zoologue https://en.wikipedia.org/wiki/Dmitry_Belyayev_(zoologist)
```

```wiki
/q
Qu'est-ce que la domestication du renard argenté ?
```
````

- La première ligne à l'intérieur du bloc est la commande (`/c`, `/q`,
  `/ingest`, `/wikistatus`, `/r`, `/tags`, `/kbupdate`, `/relire`,
  `/verifie`).
- Le reste du bloc, jusqu'à la fermeture, est l'argument — multi-lignes
  préservées telles quelles (jointes par `\n`, pas aplaties en une seule
  ligne).
- Un bloc sans deuxième ligne = commande sans argument (`/wikistatus`,
  `/tags`, `/kbupdate`, `/relire`).
- Le texte hors des blocs ```wiki (prose, notes) est ignoré.
- Plusieurs blocs dans une même note = plusieurs commandes, exécutées
  dans l'ordre d'apparition dans le document.

## Côté serveur — nouvelle route `POST /run`

`Wiki_LM/tools/server.py` gagne une route `POST /run`, requête
`{"command": "/c", "arg": "#zoologue https://..."}`.

Table de correspondance commande → opération `wiki.py`, sous-ensemble
wiki de celle déjà en place dans `derisk-deleg/src/dispatch.ts`
(dupliquée ici en Python — une table stable, courte, peu de risque de
divergence) :

| Commande | Opération |
|---|---|
| `/c` | `capture` |
| `/q` | `query` |
| `/ingest` | `ingest` |
| `/wikistatus` | `status` |
| `/r` | `search` |
| `/tags` | `tags` |
| `/kbupdate` | `kb_update` |
| `/relire` | `review` |
| `/verifie` | `verify` |

**`/supprimer` (opération `delete`/`delete_preview`) est explicitement
refusée par cette route** — pas seulement absente de la table, un test
dédié vérifie qu'une requête `{"command": "/supprimer", ...}` renvoie une
erreur claire, jamais un dispatch. Le garde-fou est porté par le serveur,
pas seulement par la discipline du client : si un autre client parlait un
jour à cette route, il ne pourrait pas non plus contourner la suppression.

La route appelle directement les fonctions `op_*` de `wiki.py` (import,
comme le fait déjà `server.py` pour `/capture` et `/query`), et renvoie
le JSON de l'opération tel quel — aucune synthèse/formatage côté serveur,
c'est le rôle du plugin.

## Côté plugin — nouvelle commande

Nouvelle entrée dans `Wiki_LM/obsidian-wikilm-capture/` (palette de
commandes + barre latérale), distincte de la capture existante — pas de
détection automatique du contenu de la note, l'utilisateur choisit
explicitement laquelle des deux actions il déclenche.

Comportement :

1. Parcourt la note ouverte, extrait chaque bloc ` ```wiki `.
2. Pour chaque bloc, dans l'ordre : première ligne = commande, reste =
   argument.
3. Si la commande est `/supprimer` (ou toute commande absente de la
   table) : n'envoie rien au serveur, insère directement un message
   d'erreur sous le bloc.
4. Sinon, POST vers `/run` avec `{command, arg}`.
5. Insère le résultat formaté juste après le bloc, avant de passer au
   bloc suivant.
6. Une commande en échec (erreur réseau, erreur métier) n'interrompt pas
   le lot — le message d'erreur est inséré comme un résultat normal, la
   commande suivante s'exécute quand même.

**Relance sur une note déjà exécutée** : le résultat inséré est encadré
par des marqueurs invisibles (`<!-- wikilm-run:start -->` /
`<!-- wikilm-run:end -->`). Au prochain déclenchement, un résultat déjà
présent sous un bloc est **remplacé**, pas dupliqué en dessous — repérable
grâce à ces marqueurs, sans toucher à tout texte que l'utilisateur aurait
écrit lui-même entre deux blocs.

Formatage des résultats dans le plugin (TypeScript, module dédié dans
`obsidian-wikilm-capture/src/`, volontairement plus simple que
`derisk-deleg/src/wiki-ops.ts` pour ce premier jet — pas de régime
bref/complet, on est déjà dans Obsidian) :

- `capture` : liste des fichiers mis en attente, ou « Rien à capturer. »
- `query` : la synthèse complète (`synthesis`).
- `status` : état d'ingestion, comptage en attente/bloqués.
- `ingest` : statut (lancé / rien à faire / déjà en cours).
- `search` : liste numérotée titre + extrait.
- `tags` : liste des tags.
- `kb_update` : confirmation de mise à jour.
- `review` : contenu de la page à relire (ou « Rien à relire. »).
- `verify` : confirmation du slug marqué vérifié.
- Toute erreur (`json.error` ou HTTP non 200) : le message d'erreur
  verbatim.

## Hors périmètre (explicite)

- Commandes gog et scout — seulement wiki pour ce premier jet.
- Support de `/supprimer` en lot, même via deux blocs consécutifs
  (`/supprimer` puis `/confirm`) — écarté en brainstorming : l'essai à
  blanc n'aurait pas de lecture réelle avant confirmation automatique.
- Formatage riche (Markdown avancé, liens cliquables) des résultats —
  texte simple suffisant pour ce premier jet.
- Ré-exécution partielle d'un lot (relancer seulement les blocs en échec)
  — chaque déclenchement relance tout le lot depuis le début.

## Tests

- `server.py` : nouveau test couvrant chaque commande de la table (dispatch
  correct vers l'opération `wiki.py`), et un test dédié confirmant que
  `/supprimer` est refusée avec une erreur, jamais dispatchée.
- Plugin : tests sur l'extraction des blocs ```wiki (multi-blocs,
  argument multi-lignes, bloc sans argument, texte hors bloc ignoré) et
  sur le formatage par type de commande — même esprit que
  `capture-text.test.ts` existant.
