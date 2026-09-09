---
tags: [documentation, LLM_Wiki, secretarius]
date: 2026-09-09
---

# Base de connaissance et clustering — synthèse

Document de synthèse sur une question récurrente : à quoi sert la base de
connaissance, comment elle se fabrique, quelle méthode de regroupement a été
retenue et pourquoi, et comment décider qu'il faut la rafraîchir.

## Ce qu'est la base de connaissance

La base de connaissance **est** un ensemble d'axes : 76 pages Markdown dans
`knowledge_base/axes/`, chacune décrivant une région thématique du corpus.
Un axe porte un titre, une description en prose, la liste de ses tags
dominants, son nombre de membres et sa cohésion moyenne.

Le reste du répertoire n'est que la machinerie pour les interroger :

| Chemin | Contenu |
|--------|---------|
| `knowledge_base/axes/` | les axes eux-mêmes (`axis-NNNN.md`) |
| `knowledge_base/embeddings/` | `axes.npy` + `axes_index.json` — vecteurs des axes |
| `knowledge_base/tags/` | `tags_dict.json` + `tags_embeddings.npy` |
| `knowledge_base/excluded.json` | axes écartés |
| `knowledge_base/index.md` | index lisible |

À ne pas confondre avec `$WIKI_PATH/embeddings/`, qui contient les
plongements **des pages du wiki** — un artefact distinct, sur lequel le
clustering travaille.

## Comment elle se fabrique, et comment elle sert

La chaîne complète compte trois maillons, et la circulation ne va pas dans
le même sens aux deux bouts.

1. **`embed.py`** calcule les plongements BGE-M3 de toutes les pages du wiki
   (service `wiki-lm-embed`, toutes les 6 h, incrémental).
2. **`cluster.py`** regroupe les pages **sources** (`src-` uniquement, pas
   les concepts ni les entités) et écrit une partition dans
   `wiki/clusterings/clustering-<signal>-<algo>-<param>/`.
3. **`kb_update.py`** transforme cette partition en axes : un axe par grappe
   retenue, avec titre et description produits par un appel LLM court
   (200 jetons par grappe), fusion des axes trop proches
   (`fusion_threshold`), exclusion des grappes trop petites (`min_size`).

Dans l'autre sens, à l'ingestion : `ingest.py` calcule le plongement du
nouveau document, appelle `kb_query` pour trouver les **trois axes les plus
proches**, et inscrit dans la page produite une ligne « Axes thématiques
proches : … ». La base sert donc à **situer** les documents entrants dans la
carte existante ; elle ne se met pas à jour de leur fait.

D'où le besoin périodique de reconstruire la carte : à mesure que le corpus
grossit, des thèmes nouveaux apparaissent sans axe pour les accueillir, et
les axes existants dérivent.

## Choix de la méthode : question tranchée en mai 2026

**HDBSCAN a été essayé puis écarté sur mesure.** Avec `param=5`, il produisait
46 grappes mais **76 % de bruit** — trois quarts des pages laissées non
classées. Ce résultat, consigné dans `docs/history/point-11-05-2026.md`, a
motivé l'écriture d'une alternative.

**L'algorithme des transferts** a été spécifié le 4 mai
(`Wiki_LM/docs/superpowers/specs/2026-05-04-transferts-design.md`) et
implémenté le 5 mai (`tools/transfers.py`, module pur sans I/O ni LLM). Il
vient de la classification automatique classique sur sacs de mots, transposée
ici aux plongements denses. Ses propriétés, opposées à celles d'HDBSCAN sur
les points qui posaient problème :

- **partition complète** — pas de notion de bruit ; toute page est classée
  (une « poubelle » existe mais reste optionnelle et réassignable via
  `force_assign`) ;
- **convergence vers un optimum local stable** — Algo 1 construit une
  partition initiale en ordre aléatoire, Algo 2 l'améliore par transferts
  successifs de type Gauss-Seidel jusqu'à stabilité ;
- **mise à jour incrémentale** — avec `initial_partition`, l'Algo 1 ne
  traite que les pages nouvelles, l'Algo 2 repasse sur le corpus complet ;
- **seuil auto-estimé** — `estimate_theta()` tire dans la distribution des
  similarités (75ᵉ percentile par défaut), ce qui absorbe l'écart entre
  distributions TF-IDF et plongements denses.

Deux garde-fous préviennent les oscillations : `min_gain_delta`, qui refuse
les transferts marginaux générateurs de cycles A→B→A, et `max_iter`.

Premier passage réel, mai 2026 : `clustering-embeddings-transfers-0.404`,
84 grappes, θ = 0,404.

## Comment savoir s'il faut rafraîchir

C'est le point le plus utile de la spec, et il évite de reconstruire à
l'aveugle. `run_transfers(..., dry_run=True)` exécute l'Algo 2 **sans rien
écrire** et retourne le nombre de transferts qu'il *voudrait* faire :

```
{"proposed_transfers": int, "total": int, "ratio": float, "adequate": bool}
```

Le ratio de transferts proposés mesure directement l'écart entre la carte
actuelle et celle que produirait le corpus d'aujourd'hui.
`QUALITY_THRESHOLD = 0.20` : au-delà de 20 %, un reclustering est
recommandé. Aucun appel LLM, donc quasi gratuit.

Exposé par le serveur : `GET /cluster-quality?signal=embeddings&param=75`.

**Limite à connaître** : cette mesure charge la partition existante
(`_load_existing_partition`, `cluster.py:257`) pour compter les transferts
qu'elle subirait. Elle suppose donc qu'un clustering existe déjà pour le
corpus évalué. Sur le wiki vivant, dont `clusterings/` est vide, elle n'a
rien à mesurer : il n'y a pas une carte à rafraîchir, il y a une **première**
carte à construire. La mesure reprend tout son sens ensuite, pour décider des
reconstructions suivantes.

## État constaté le 2026-09-09

- Les 76 axes datent des 11 et 12 mai et portent `source_wikis:
  wiki_signets_05_2026` : ils ont été calculés sur **l'archive des signets**,
  pas sur le wiki vivant. La carte décrit donc un corpus vieux de quatre mois
  et qui n'est pas celui qu'on interroge.
- `wiki/clusterings/` du wiki vivant est **vide** — d'où la plainte légitime
  de `/kbupdate`, qui n'a rien à consommer.
- La chaîne est en revanche compatible avec le wiki vivant : `cluster.py`
  écrit dans `wiki/clusterings/`, exactement là où `kb_update` lit, et la
  structure attendue (`clusterings/` + `sources/`) est celle du wiki courant.
  L'exemple d'usage archivé dans l'en-tête de `kb_update.py` n'est qu'un
  exemple.
- Volumétrie actuelle : 95 sources, 450 concepts, 296 entités. Le clustering
  ne porte que sur les 95 sources.

## Chantiers ouverts

- **`kb_lint.py`** — jamais écrit. Prévu pour détecter la dérive des
  centroïdes, les doublons d'axes et les axes orphelins. Complémentaire de la
  mesure `dry_run` ci-dessus, qui dit *qu'il y a* dérive sans dire *où*.
- **Assainir concepts et entités** — projet à part entière : utiliser le même
  clustering pour regrouper les 746 pages dérivées, fusionner les doublons,
  puis corriger les liens des pages sources en conséquence.
- **Ingérer la documentation du projet** dans le wiki, ou monter un wiki
  dédié à la documentation.
- **Précaution avant tout reclustering** : `kb_update` réécrit les axes
  existants. Conserver une copie de `knowledge_base/` hors du coffre permet
  de comparer l'ancienne carte à la nouvelle et de revenir en arrière.

## Références

- Spec : `Wiki_LM/docs/superpowers/specs/2026-05-04-transferts-design.md`
- Plans voisins : `Wiki_LM/docs/superpowers/plans/2026-05-01-wiki-clustering.md`,
  `2026-05-06-knowledge-base.md` (le plan d'exécution des transferts
  lui-même n'a pas été conservé sous ce nom)
- Historique et mesures : `docs/history/point-11-05-2026.md`
- Outils : `tools/transfers.py`, `tools/cluster.py`, `tools/kb_update.py`,
  `tools/kb_query.py`
