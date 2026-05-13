---
name: secretarius-document-normalizer
description: Normalise les entrées documentaires en secretarius.document.v0.1, applique des règles de priorité (humain > existant > inféré), initialise et maintient indexing.state, puis prépare les données pour extract_expressions, expressions_to_embeddings et semantic_graph_search.
---

# Secretarius Document Normalizer

Utiliser ce skill quand l'utilisateur veut:
- normaliser une entrée partielle (`url`, `note`, document incomplet),
- enrichir un document selon `secretarius.document.v0.1`,
- préparer l'orchestration extraction -> embeddings -> recherche.

## Résultat attendu

Toujours produire un JSON valide avec:
- `schema = "secretarius.document.v0.1"`,
- `type` non vide,
- au moins un des champs: `source.url`, `content.text`, `content.content_ref`.

## Priorités de remplissage

Appliquer strictement cet ordre:
1. Valeurs humaines explicites (utilisateur).
2. Valeurs déjà présentes dans le document.
3. Valeurs inférées automatiquement.

Ne jamais écraser une valeur humaine non vide avec une valeur inférée.

## Normalisation d'entrée

### Cas URL brute
Entrée:
```json
{ "type": "url", "url": "https://exemple.org" }
```
Sortie minimale:
- `type = "url"`
- `source.url = <url>`
- `content.mode = "none"`
- `indexing.state = "new"`

### Cas note brute
Entrée:
```json
{ "type": "note", "note": "..." }
```
Sortie minimale:
- `type = "note"`
- `content.mode = "inline"`
- `content.text = <note>`
- `content.length_chars = len(content.text)`
- `indexing.state = "new"`

## Valeurs par défaut

Si absents:
- `user_fields.status = "draft"`
- `user_fields.tags = []`
- `user_fields.keywords = []`
- `source.authors = []`
- `indexing.errors = []`
- `indexing.pipeline_version = "v0.1"`

`content.mode`:
- `inline` si `content.text` existe,
- `ref` si `content.content_ref` existe,
- sinon `none`.

## Identifiants stables

- `doc_id`: conserver si présent, sinon générer un identifiant stable (hash de `canonical_url|url|content.hash|content.text[:512]`).
- `source.source_id`: conserver si présent; sinon hash de `canonical_url|url` quand disponible.
- `derived.chunks[].chunk_id`: hash de `doc_id|start|end|chunk_text[:256]`.

## Règles d'orchestration des 3 outils

1. `extract_expressions`
- Entrée: document normalisé + contenu résolu.
- Effets attendus:
  - remplit `derived.chunks`,
  - remplit `derived.expressions` (sans embeddings),
  - met `indexing.state` sur `extracting` puis `embedding` (ou `done`).

2. `expressions_to_embeddings`
- Entrée: `derived.expressions`.
- Effets attendus:
  - associe un `embedding_ref` à chaque expression (ou vecteur inline temporaire),
  - met `indexing.state` sur `embedding` puis `done`.

3. `semantic_graph_search`
- Entrée: embeddings + metadata + booléen `upsert`.
- Effets attendus:
  - `upsert=false`: recherche seule,
  - `upsert=true`: insertion + recherche,
  - met `indexing.state` sur `upserting` puis `done`.

## Machine d'état

États autorisés:
`new -> queued -> fetching -> extracting -> embedding -> upserting -> done`

En échec:
- `indexing.state = "error"`
- append dans `indexing.errors[]` avec:
  - `at` (ISO-8601),
  - `stage`,
  - `message`.

## Garde-fous

- Ne jamais supprimer un champ existant.
- Ne jamais vider `user_fields` s'il contient des données.
- Si ambiguïté, journaliser dans `indexing.errors[]` plutôt que perdre l'information.

## Référence

Voir `references/document-schema-v0.1.md` pour la structure cible résumée.
