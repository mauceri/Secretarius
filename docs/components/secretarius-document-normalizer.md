---
tags: [documentation, secretarius]
date: 2026-05-14
---

# Composant : secretarius-document-normalizer

## Rôle

Skill OpenClaw de normalisation des entrées documentaires vers le schéma
`secretarius.document.v0.1`. Orchestre le pipeline : extraction d'expressions ->
calcul d'embeddings -> indexation sémantique.

## Prérequis

- OpenClaw configuré
- Dépendances Wiki_LM installées

## Installation

Installé automatiquement par `install.sh` via `openclaw-config/workspace/skills/`.

```bash
ls ~/.openclaw/workspace/skills/secretarius-document-normalizer/SKILL.md
```

## Schéma `secretarius.document.v0.1`

Champ obligatoire : `schema = "secretarius.document.v0.1"`, `type` non vide,
et au moins un de : `source.url`, `content.text`, `content.content_ref`.

### Priorités de remplissage

1. Valeurs humaines explicites (ne jamais écraser)
2. Valeurs déjà présentes dans le document
3. Valeurs inférées automatiquement

### Cas d'usage typiques

**URL brute :**
```json
{ "type": "url", "url": "https://exemple.org" }
```
-> normalise vers `source.url`, `content.mode = "none"`, `indexing.state = "new"`

**Note brute :**
```json
{ "type": "note", "note": "Réflexion sur le Memex" }
```
-> normalise vers `content.mode = "inline"`, `content.text = ...`

## Pipeline d'orchestration

1. **`extract_expressions`** : découpe en chunks, extrait les expressions -> `derived.chunks` + `derived.expressions`
2. **`expressions_to_embeddings`** : calcule les embeddings -> `embedding_ref` par expression
3. **`semantic_graph_search`** : insère dans le graphe sémantique (upsert=true) ou recherche (upsert=false)

## Machine d'état

```
new -> queued -> fetching -> extracting -> embedding -> upserting -> done
```

En cas d'erreur : `indexing.state = "error"` + log dans `indexing.errors[]`.

## Notes d'architecture

Ce skill est l'interface entre le format documentaire brut et le pipeline vectoriel
de Secretarius (embeddings BGE-M3, graphe sémantique). Il impose une structure
stricte qui garantit la traçabilité et l'idempotence du pipeline.
