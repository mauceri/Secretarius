# Algorithme des transferts — Spec de design

## Vue d'ensemble

Implémenter l'algorithme des transferts comme alternative à HDBSCAN pour le clustering des pages `src-` du wiki. Contrairement à HDBSCAN, cet algorithme produit une partition complète (sans notion de bruit), converge vers un optimum local stable, et supporte la mise à jour incrémentale sur un clustering existant.

Origine : algorithme classique de classification automatique sur sacs de mots (similarité cosinus TF-IDF). Adapté ici aux embeddings BGE-M3.

---

## Architecture

### Fichiers impactés

| Fichier | Action | Rôle |
|---------|--------|------|
| `tools/transfers.py` | Créer | Algorithme des transferts (pur, sans I/O wiki) |
| `tools/cluster.py` | Modifier | Intégration `algo="transfers"`, nouveaux paramètres CLI |
| `tools/server.py` | Modifier | Endpoint `GET /cluster-quality` |
| `tests/test_transfers.py` | Créer | Tests unitaires de l'algorithme |
| `tests/test_cluster.py` | Modifier | Test d'intégration `algo="transfers"` |

`transfers.py` est un module pur : pas d'I/O fichier, pas d'appel LLM. Toute la logique d'écriture des fichiers wiki reste dans `cluster.py`.

---

## Algorithme

### Algorithme 1 — Partition initiale

Ordre de traitement : aléatoire (évite les biais d'ordre).

```
Pour chaque document x (ordre aléatoire) :
  Pour chaque classe C existante :
    gain(C) = (1/|C|) × Σ_{y∈C} sim(x, y)
  C* = argmax gain(C)
  Si gain(C*) > θ ET (max_k est None OU nb_classes < max_k) :
    assigner x à C*
  Sinon si max_k atteint :
    assigner x à la poubelle (label -1)
  Sinon :
    créer nouvelle classe {x}
```

### Algorithme 2 — Amélioration (optimisation locale)

Répéter jusqu'à stabilité (aucun transfert lors d'une passe complète), ou jusqu'à `max_iter` itérations :

```
Pour chaque document x dans classe C_x :
  Si |C_x| > 1 :
    contrib = (1/(|C_x|-1)) × Σ_{y∈C_x, y≠x} sim(x, y)
  Sinon :
    contrib = 0
  Pour chaque classe C ≠ C_x :
    gain(C) = (1/|C|) × Σ_{y∈C} sim(x, y)
  Si max(gain) > contrib + min_gain_delta :
    déplacer x vers argmax(gain)
    mettre à jour les centroïdes de C_x et C_cible
```

**Terminaison :** deux critères d'arrêt combinés pour prévenir les oscillations :
- **`min_gain_delta`** : transfer uniquement si le gain dépasse la contribution courante d'au moins ε — élimine les transferts marginaux qui peuvent engendrer des cycles A→B→A
- **`max_iter`** : nombre maximum de passes complètes sur le corpus — filet de sécurité pour les oscillations persistantes à gain non nul

### Assignation forcée (optionnelle)

Après convergence de l'Algo 2, si `force_assign=True` :

```
Pour chaque document x en poubelle (label -1) :
  assigner x à la classe dont le centroïde est le plus similaire
```

### Complexité

**O(k × N × C)** où :
- N = nombre de documents
- C = nombre de clusters
- k = nombre d'itérations jusqu'à convergence

Avec centroïdes précomputés et mis à jour incrémentalement, chaque évaluation de gain est O(dim) (produit scalaire). Pour N=1900, C=50, k=10 : ~950 000 opérations.

---

## Estimation empirique de θ

θ est estimé à partir de la distribution des similarités par tirage aléatoire dans le triangle supérieur de la matrice de similarité :

```python
def estimate_theta(
    sim: np.ndarray,
    percentile: float = 75.0,
    sample_size: int = 50_000,
    rng: np.random.Generator | None = None,
) -> float:
```

- `percentile=75` : seuil par défaut (à calibrer selon les corpus)
- `sample_size=50_000` : statistiquement suffisant pour N < 20 000
- `rng` : graine fixable pour reproductibilité des tests

Les distributions de similarités diffèrent entre TF-IDF (origine de l'algorithme) et embeddings denses (usage ici) — l'estimation empirique absorbe cette différence.

---

## Interface publique — `transfers.py`

```python
QUALITY_THRESHOLD: float = 0.20  # ratio de transferts → reclustering recommandé
MIN_PAGES_FOR_CLUSTERING: int = 50  # seuil wiki nouveau

def estimate_theta(
    sim: np.ndarray,
    percentile: float = 75.0,
    sample_size: int = 50_000,
    rng: np.random.Generator | None = None,
) -> float: ...

def run_transfers(
    slugs: list[str],
    sim: np.ndarray,
    theta: float,
    max_k: int | None = None,
    force_assign: bool = False,
    dry_run: bool = False,
    initial_partition: dict[int, list[int]] | None = None,
    max_iter: int = 100,
    min_gain_delta: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> dict[int, list[int]] | dict:
    """
    Si dry_run=False : retourne {cluster_id: [indices]}, label -1 = poubelle.
    Si dry_run=True  : retourne {"proposed_transfers": int, "total": int,
                                  "ratio": float, "adequate": bool}.
    Si initial_partition fournie : Algo 1 uniquement sur les nouvelles pages,
    Algo 2 sur le corpus complet.
    """
```

---

## Intégration dans `cluster.py`

### Signature `run_clustering()`

```python
def run_clustering(
    wiki_dir: Path,
    embed_dir: Path,
    signal_str: str,
    param: int,
    llm: LLM | None = None,
    algo: str = "hdbscan",
    # Paramètres transfers uniquement :
    theta: float | None = None,
    max_k: int | None = None,
    force_assign: bool = False,
    incremental: bool = False,
    dry_run: bool = False,
) -> dict:
```

Quand `algo="transfers"` :
- Si `theta=None` : auto-estimation via `estimate_theta(sim)`
- Si `incremental=True` : charge la partition existante depuis les fichiers wiki, passe en `initial_partition`
- Si `dry_run=True` : retourne les stats sans écrire de fichiers

### CLI additions

```
--algo transfers
--theta FLOAT      # si absent : auto-estimé
--max-k INT
--force-assign
--incremental
--dry-run
```

### Endpoint serveur

**Existant** `POST /cluster` : reçoit les nouveaux champs `theta`, `max_k`, `force_assign`, `incremental`.

**Nouveau** `GET /cluster-quality` :
- Paramètres query : `signal`, `param` (identifient le clustering à évaluer)
- Charge le clustering le plus récent correspondant
- Lance `run_transfers(..., dry_run=True)` sur la matrice de similarité courante
- Retourne `{"proposed_transfers": int, "total": int, "ratio": float, "adequate": bool}`

---

## Mise à jour incrémentale

### Premier clustering (wiki nouveau)

`run_clustering()` retourne un dict avec clé `"error"` si `len(pages) < MIN_PAGES_FOR_CLUSTERING` (50 pages). Ce comportement est cohérent avec les autres cas d'erreur de `run_clustering` et facilite la gestion par les appelants CLI et serveur.

### Recalcul périodique

Critère d'inadéquation : un dry-run de l'Algo 2 sur la partition existante propose > `QUALITY_THRESHOLD` (20%) de transferts.

Workflow typique pour un skill LLM :

```
GET /cluster-quality?signal=embeddings&param=30
→ {"adequate": false, "ratio": 0.31}
→ POST /cluster {"algo": "transfers", "incremental": true, "signal": "embeddings", "param": 30}
```

### Partition initiale lors d'un recalcul

Avec `--incremental` / `incremental=True` :
1. Charger la partition depuis les fichiers `cluster-*.md` existants : parsé depuis la section `## Documents membres` du corps markdown (le champ `members` du frontmatter est un entier count, pas une liste)
2. Identifier les nouvelles pages (absentes de la partition)
3. Algo 1 uniquement sur les nouvelles pages
4. Algo 2 sur le corpus complet
5. Écriture des fichiers wiki mis à jour

### LLM et clustering existant

Les fichiers `cluster-*.md` contiennent titres et descriptions générés. Le LLM appelant le skill peut les lire comme contexte pour interpréter les résultats du recalcul ou prendre la décision de reclustering — aucun support technique supplémentaire nécessaire.

---

## Tests

### `tests/test_transfers.py`

Corpus synthétique : 10 documents, 2 groupes bien séparés (similarités intra ~0.8, inter ~0.2).

| Test | Vérifie |
|------|---------|
| `test_estimate_theta` | Valeur reproductible avec `rng` fixe, dans [0, 1] |
| `test_run_transfers_converges` | Algo 1+2 produit 2 clusters stables |
| `test_run_transfers_max_k` | `max_k=1` → surplus en poubelle (label -1) |
| `test_run_transfers_force_assign` | `force_assign=True` → aucun document en poubelle |
| `test_run_transfers_dry_run` | Retourne stats, partition non modifiée |
| `test_run_transfers_incremental` | `initial_partition` existante → nouvelles pages assignées via Algo 1 |
| `test_quality_threshold` | ratio > `QUALITY_THRESHOLD` → `adequate=False` |
| `test_run_transfers_no_oscillation` | corpus conçu pour oscillations (`min_gain_delta=0`) → `max_iter` stoppe l'exécution |
| `test_run_transfers_min_gain_delta` | `min_gain_delta` élevé → convergence en une seule passe |

### `tests/test_cluster.py`

`test_run_clustering_transfers` : corpus réel miniature (5 pages avec embeddings), vérifie la présence des fichiers wiki attendus dans `clusterings/`.

---

## Perspective future — Vecteurs de Salton et PCA

*(Hors périmètre de ce plan — à spécifier séparément)*

**Vecteurs de Salton** : après clustering, chaque document reçoit un vecteur de dimension C où la composante i est sa similarité au centroïde du cluster i. Représentation du document dans l'espace des clusters.

**PCA / axes d'inertie** : ACP sur ces vecteurs de Salton → axes principaux du corpus. Permet :
- Visualisation 2D/3D des documents dans l'espace thématique
- Détection de dimensions sémantiques transversales aux clusters
- Possible remplacement de Gram-Schmidt si les axes ne sont pas orthogonaux a priori

**Briques disponibles** : numpy (SVD), les centroïdes déjà calculés dans `run_transfers()`.
