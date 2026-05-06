# Base de connaissance compactée — Spec de design

## Vue d'ensemble

Extraire la "substantifique moelle" de wiki_signets_05_2026 (1892 pages, 83-85 clusters) sous forme d'une base de connaissance évolutive stockée dans `~/Secretarius/Wiki_LM/knowledge_base/`. Cette base sert deux usages : fournir au LLM un contexte de proximité thématique lors de l'ingestion de nouveaux documents, et capitaliser la connaissance accumulée entre wikis archivés.

---

## Architecture

```
~/Secretarius/Wiki_LM/knowledge_base/
├── index.md                        # Carte thématique globale (lisible LLM + humain)
├── excluded.json                   # Clusters exclus avec raison (traçabilité)
├── axes/
│   ├── axis-0001.md                # Un axe thématique (= centroïde persistant)
│   ├── axis-0002.md
│   └── ...
├── embeddings/
│   ├── axes.npy                    # Matrice (K × dim) des centroïdes
│   └── axes_index.json             # {"ids": ["axis-0001", ...]}
└── tags/
    ├── tags_dict.json              # {canonical: [variants...]}
    └── tags_embeddings.npy         # Embeddings des tags canoniques (dim BGE-M3 = 1024)
```

`kb_update.py` est le seul outil qui écrit dans ce répertoire. Les outils de lecture (`ingest.py`, serveur) ne font que lire.

---

## Format des axes

Chaque `axis-NNNN.md` suit le frontmatter :

```yaml
---
title: Philosophie du langage
description: "Ce groupe traite des théories du langage, de la sémantique et de la pragmatique."
source_wikis: [wiki_signets_05_2026]
updated: 2026-05-06
members_count: 14
cohesion: 0.74          # similarité intra-cluster moyenne
tags: [linguistique, sémantique, pragmatique]
status: active          # active | deprecated | garbage
---

## Pages représentatives

- [[src-0042]] — Titre de la page
- [[src-0117]] — Titre de la page
- [[src-0389]] — Titre de la page
```

---

## Critères de sélection des clusters

Lors de l'extraction depuis un clustering archivé, un cluster est retenu comme axe si et seulement si :

| Critère | Seuil par défaut | Option CLI |
|---------|-----------------|------------|
| Taille minimum | >= 3 pages | `--min-size` |
| Cohésion minimum | >= theta/2 (theta parsé depuis le nom du répertoire de clustering) | `--min-cohesion` |
| Statut | != garbage | automatique |

Les clusters exclus sont enregistrés dans `excluded.json` :

```json
[
  {"cluster_id": "cluster-0023", "wiki": "wiki_signets_05_2026", "reason": "size < 3"},
  {"cluster_id": "cluster-0041", "wiki": "wiki_signets_05_2026", "reason": "cohesion < threshold"}
]
```

La détection des clusters poubelle s'appuie sur le cluster analysis skill (à venir). En attendant, marquage manuel via `status: garbage` dans le frontmatter du cluster source.

**Tags d'un axe** : agrégés depuis les tags des pages membres (top-10 les plus fréquents), puis normalisés via `tags_dict.json`.

---

## Composantes

### `tools/kb_update.py`

Prend un wiki archivé en entrée. Charge le clustering spécifié. Pour chaque cluster retenu :

1. Calcule son centroïde depuis les embeddings des pages membres
2. Cherche l'axe existant le plus similaire dans `axes.npy` (similarité cosinus)
3. Si similarité >= `fusion_threshold` (défaut 0.85) :
   - Met à jour le centroïde (moyenne pondérée par `members_count`)
   - Cumule `source_wikis`
   - Met à jour `tags`, `members_count`, `cohesion`, `updated`
4. Sinon : crée un nouvel axe `axis-NNNN.md` (NNNN = prochain entier disponible)

À la fin, régénère `index.md`, `axes.npy`, `axes_index.json` et `excluded.json`.

**CLI :**
```
python tools/kb_update.py \
  --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \
  --clustering clustering-embeddings-transfers-0.403 \
  --fusion-threshold 0.85 \
  --min-size 3 \
  --min-cohesion auto
```

**Gestion d'erreurs :**
- Clustering demandé absent → erreur claire, aucune écriture partielle
- `axes.npy` absent → créé depuis zéro (premier appel)
- Dimension des embeddings incompatible avec les axes existants → erreur explicite

### `tools/kb_query.py` (ou fonction dans `ingest.py`)

Étant donné le vecteur d'un document entrant :

```python
def kb_query(vec: np.ndarray, kb_dir: Path, top_k: int = 3) -> list[dict]:
    """Retourne les top_k axes les plus proches avec leur score."""
```

Sortie injectée dans le prompt LLM :
```
Ce document est proche de :
- Philosophie du langage (0.81)
- IA et cognition (0.74)
- Épistémologie (0.61)
```

### `tools/kb_tags.py`

Construit le dictionnaire de tags depuis les tags de tous les wikis archivés fournis :

1. Embed tous les tags connus (via le serveur d'embeddings BGE-M3)
2. Clustering par similarité cosinus (seuil configurable, défaut 0.90)
3. Forme canonique = tag le plus fréquent du groupe
4. Hapaxes (tags apparus une seule fois) : exclus par défaut (option `--keep-hapax`)
5. Stocke `tags_dict.json` + `tags_embeddings.npy`

---

## Flux de données

### Archivage d'un wiki

```
python tools/cluster.py --algo transfers --signal embeddings --theta auto
→ wiki/clusterings/clustering-embeddings-transfers-0.403/

python tools/kb_update.py \
  --wiki ~/Documents/Arbath/Wiki_LM/wiki_signets_05_2026 \
  --clustering clustering-embeddings-transfers-0.403

→ knowledge_base/axes/ mis à jour
→ knowledge_base/index.md régénéré
```

### Ingestion d'un nouveau document

```
ingest.py → embed(page) → kb_query(vecteur, top_k=3)
→ {"axes": [{"title": "...", "score": 0.81, "tags": [...]}, ...]}
→ injecté dans le prompt LLM de description/tagging
```

---

## Tests

### `tests/test_kb_update.py`

| Test | Vérifie |
|------|---------|
| `test_kb_update_first_run` | Premier appel sur corpus 2 groupes → 2 axes créés, `axes.npy` shape (2, dim) |
| `test_kb_update_fusion` | Deuxième appel avec clustering similaire → axes mis à jour (pas dupliqués), `source_wikis` cumulés |
| `test_kb_update_new_axis` | Cluster éloigné de tous les axes existants → nouvel axe créé |
| `test_kb_update_exclusion` | Cluster trop petit ou faible cohésion → dans `excluded.json`, absent de `axes/` |
| `test_kb_update_dim_mismatch` | Embeddings de dimension différente → erreur explicite |

### `tests/test_kb_query.py`

| Test | Vérifie |
|------|---------|
| `test_kb_query_top_k` | Vecteur proche d'un axe connu → top-1 correct, scores décroissants |
| `test_kb_query_empty_kb` | `axes.npy` absent → retourne liste vide sans exception |

### `tests/test_kb_tags.py`

| Test | Vérifie |
|------|---------|
| `test_kb_tags_synonyms` | Tags synonymes (embeddings proches) → même canonique |
| `test_kb_tags_hapax_excluded` | Hapaxes exclus par défaut |
| `test_kb_tags_keep_hapax` | Option `--keep-hapax` → hapaxes conservés |

---

## Fichiers impactés

| Fichier | Action | Rôle |
|---------|--------|------|
| `tools/kb_update.py` | Créer | Mise à jour de la base de connaissance |
| `tools/kb_query.py` | Créer | Requête de proximité à l'ingestion |
| `tools/kb_tags.py` | Créer | Construction du dictionnaire de tags |
| `tests/test_kb_update.py` | Créer | Tests kb_update |
| `tests/test_kb_query.py` | Créer | Tests kb_query |
| `tests/test_kb_tags.py` | Créer | Tests kb_tags |
| `tools/ingest.py` | Modifier | Appel de kb_query lors de l'ingestion |

`knowledge_base/` n'est pas versionné dans git (données dérivées volumineuses) — à ajouter dans `.gitignore`.

---

## Maintenance — détection de dérive

Un axe mis à jour de nombreuses fois par moyenne pondérée peut dériver de tout cluster concret. `kb_lint.py` (hors périmètre de ce plan — à spécifier séparément) détecte :

- **Dérive** : cohésion d'un axe inférieure à un seuil (cosine similarity centroïde vs pages représentatives)
- **Doublons** : deux axes dont la similarité mutuelle dépasse `fusion_threshold`
- **Axes orphelins** : axes non mis à jour depuis N archivages

Sortie : rapport Markdown + suggestion de recalcul complet (`kb_update.py --rebuild`).
