# Wiki Clustering — Design Spec

**Date :** 2026-04-28  
**Statut :** approuvé  
**Périmètre :** clustering des pages `src-` du wiki, multi-granularité, intégration Obsidian + serveur Flask

---

## Contexte

Le wiki contient 20 717 pages dont ~2 411 pages source (`src-`) avec embeddings BGE-M3 (1024-dim, L2-normalisés). Interroger le wiki en texte libre ne donne pas une vue d'ensemble de son contenu thématique. L'objectif est de produire des clusters de documents source navigables dans Obsidian et interrogeables via le serveur.

Cas d'usage couverts :
- **Navigation/découverte** — retrouver des sources voisines d'une source connue via le graphe Obsidian
- **Cartographie** — visualiser la densité thématique du wiki, identifier les zones vides ou surchargées
- **Requêtes enrichies** — possibilité future d'enrichir les réponses `/query` avec le cluster de rattachement

---

## Architecture

Trois nouveaux fichiers dans `tools/` :

### `similarity.py`

Calcule une matrice de similarité `(N×N, float32)` pour les N pages `src-` à partir d'un signal :

| Classe | Signal | Méthode |
|--------|--------|---------|
| `EmbeddingSimilarity` | Vecteurs BGE-M3 (déjà calculés) | Produit matriciel sur vecteurs L2-normalisés |
| `CoLinkSimilarity` | Liens `[[c-…]]` et `[[e-…]]` partagés | Jaccard sur ensembles de liens |
| `TagSimilarity` | Tags du frontmatter | Jaccard sur ensembles de tags |
| `CombinedSimilarity` | Combinaison de plusieurs signaux | Moyenne pondérée de matrices normalisées |

Interface commune :
```python
class BaseSimilarity:
    def compute(self, slugs: list[str]) -> np.ndarray: ...
```

### `cluster.py`

Prend une matrice de similarité, effectue le clustering, génère les fichiers wiki.

Étapes :
1. Charger les pages `src-` (slugs, frontmatter, résumés)
2. Calculer la matrice de similarité via le signal choisi
3. Lancer HDBSCAN (`min_cluster_size=param`) — algorithme interne, non exposé en façade
4. Pour chaque cluster : identifier le parangon (document à similarité moyenne maximale avec les membres)
5. Appeler le LLM pour titre (~5 mots) + description (2-3 phrases) à partir des résumés des 5 membres les plus centraux — un seul appel par cluster
6. Écrire les fichiers dans `wiki/clustering-<signal>-hdbscan-<param>/`
7. Écrire `unclustered.md` pour les points bruit (label HDBSCAN = -1)
8. Calculer la proximité inter-clusters : similarité cosinus entre centroïdes (moyenne des embeddings membres) — les 3 plus proches sont listés dans chaque fichier cluster
9. Écrire `index.md` avec statistiques du run

### Intégration `server.py`

Deux nouveaux endpoints, même modèle que `/embed` :

```
POST /cluster        {"signal": "embeddings", "param": 30}
GET  /cluster-status
```

Run en thread daemon. Réponse `/cluster-status` :
```json
{"running": false, "last": {"signal": "embeddings", "param": 30, "clusters": 74, "noise": 12}, "error": null}
```

---

## Format des fichiers de sortie

### Fichier cluster : `cluster-<signal>-hdbscan-<param>-<id>.md`

```markdown
---
category: cluster
signal: embeddings
algo: hdbscan
param: 30
members: 42
paragon: src-xxx
created: 2026-04-28
---

# [Titre généré par LLM]

[Description 2-3 phrases générée par LLM]

## Parangon

[[src-xxx]] — Titre du document le plus central

## Documents membres

- [[src-yyy]] — Titre
- [[src-zzz]] — Titre

## Clusters proches

- [[cluster-embeddings-hdbscan-30-007]] (similarité : 0.82)
- [[cluster-embeddings-hdbscan-30-012]] (similarité : 0.79)
```

### Fichier index : `clustering-<signal>-hdbscan-<param>/index.md`

Résumé du run : nombre de clusters, nombre de documents non assignés, liens vers tous les clusters.

### Fichier bruit : `clustering-<signal>-hdbscan-<param>/unclustered.md`

Liste des sources non assignées (label -1 HDBSCAN). Information utile : ces sources sont thématiquement isolées.

---

## Interface CLI

```bash
# Granularité unique
python tools/cluster.py --signal embeddings --param 30

# Plusieurs granularités d'un coup
python tools/cluster.py --signal embeddings --param 10,30,60

# Combinaison de signaux (poids égaux par défaut)
python tools/cluster.py --signal embeddings+colinks --param 30

# Sans appels LLM (géométrie seulement, pour tests rapides)
python tools/cluster.py --signal embeddings --param 30 --no-llm
```

`--algo` existe dans le code pour extension future mais n'est pas exposé en façade (HDBSCAN par défaut et seul algorithme implémenté).

---

## Dépendances

À ajouter à `requirements.txt` :
- `hdbscan>=0.8.33` ou `scikit-learn>=1.3` (HDBSCAN intégré depuis 1.3)

`numpy`, `sentence-transformers`, `python-frontmatter` déjà présents.

---

## Ce qui est hors périmètre (pour l'instant)

- Réorganisation du wiki en répertoires `Sources/`, `Concepts/`, `Entités/` — décision architecturale séparée
- Suppression des préfixes `src-`, `c-`, `e-` des slugs — dépend de la réorganisation
- Intégration des clusters dans les réponses `/query` — extension future
- Algorithmes autres que HDBSCAN (Louvain, K-means) — extension future
