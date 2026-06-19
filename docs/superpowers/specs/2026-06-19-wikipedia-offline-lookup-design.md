# Support Wikipedia local/offline pour ingest

Date : 2026-06-19

## Contexte

`ingest.py` enrichit les pages de concepts et d'entites via `WikiLookup.lookup()`.
`WikiLookup` suit deja l'ordre `ZIM local -> cache SQLite -> API REST Wikipedia`, mais le mode offline n'est pas explicite et le repertoire ZIM par defaut est code en dur.

Le changement doit garder l'API actuelle pour les appelants existants :

- `WikiLookup(wiki_path)`
- `lookup(title, langs=None)`

Les scripts `ingest.py` et `build_wiki_cache.py` doivent beneficier de la configuration par variables d'environnement sans option CLI supplementaire dans ce premier increment.

## Objectifs

- Permettre de choisir le repertoire ZIM avec `WIKI_ZIM_DIR`.
- Permettre un mode strictement sans reseau avec `WIKI_LOOKUP_OFFLINE=1`.
- Permettre une configuration avancee des backends avec `WIKI_LOOKUP_BACKENDS`.
- Conserver le comportement par defaut actuel quand aucune variable n'est definie.
- Tester le comportement sans vrai fichier ZIM, sans reseau et sans LLM.

## Non-objectifs

- Ne pas modifier le pipeline `ingest.py`.
- Ne pas ajouter de telechargement automatique de fichiers ZIM.
- Ne pas resoudre les redirects, alias ou titres approximatifs dans cet increment.
- Ne pas ajouter d'index SQLite FTS des titres ZIM.

## Comportement

### Repertoire ZIM

`WikiLookup.__init__` continue d'accepter `zim_dir`.

Priorite de resolution :

1. argument explicite `zim_dir`;
2. variable `WIKI_ZIM_DIR`;
3. defaut existant `~/Secretarius/Wiki_LM/zim`.

### Backends

L'ordre par defaut reste :

```text
zim,cache,api
```

Si `WIKI_LOOKUP_OFFLINE` vaut `1`, `true`, `yes` ou `on` (insensible a la casse), `WikiLookup` force :

```text
zim,cache
```

Dans ce mode, l'API Wikipedia ne doit jamais etre appelee.

Si `WIKI_LOOKUP_OFFLINE` n'est pas actif et que `WIKI_LOOKUP_BACKENDS` est defini, sa valeur est parse comme une liste separee par des virgules, par exemple :

```text
WIKI_LOOKUP_BACKENDS=cache
WIKI_LOOKUP_BACKENDS=zim,cache
WIKI_LOOKUP_BACKENDS=cache,api
```

Les backends reconnus sont `zim`, `cache` et `api`.
Les valeurs inconnues sont ignorees.
Si la liste parse ne contient aucun backend reconnu, `WikiLookup` revient au defaut `zim,cache,api`.

### Lookup

Pour chaque langue demandee, `lookup()` parcourt les backends configures dans l'ordre.
Le premier resultat non vide est retourne.

Effets de bord conserves :

- un resultat ZIM est stocke dans le cache SQLite;
- un resultat API est stocke dans le cache SQLite;
- un resultat cache est retourne sans appel reseau.

## Tests

Les tests seront ajoutes dans `Wiki_LM/tests/test_wiki_lookup.py`.

Cas couverts :

- `WIKI_ZIM_DIR` est pris en compte quand `zim_dir` n'est pas fourni.
- `zim_dir` explicite garde la priorite sur `WIKI_ZIM_DIR`.
- `WIKI_LOOKUP_OFFLINE=1` empeche tout appel API.
- `WIKI_LOOKUP_BACKENDS=cache` consulte seulement le cache.
- le comportement par defaut reste `zim -> cache -> api`.
- une liste de backends invalide revient au defaut.

Les tests ZIM utiliseront `monkeypatch` sur `_zim_files` et `_zim_lookup` afin de ne pas dependre d'un vrai fichier `.zim` ni de `libzim`.

## Documentation

`Wiki_LM/README.md` documentera :

- ou placer les fichiers `.zim`;
- `WIKI_ZIM_DIR`;
- `WIKI_LOOKUP_OFFLINE`;
- `WIKI_LOOKUP_BACKENDS`;
- le fait que `ingest.py` utilise ces variables indirectement via `WikiLookup`.

## Risques

- Une variable d'environnement mal orthographiee pourrait laisser le fallback API actif.
  Le README devra montrer explicitement `WIKI_LOOKUP_OFFLINE=1` pour le mode sans reseau.
- L'ordre configurable des backends peut produire des effets inattendus si `cache` est place avant `zim`.
  C'est accepte pour le mode avance, car il peut etre utile de privilegier un cache prechauffe.
