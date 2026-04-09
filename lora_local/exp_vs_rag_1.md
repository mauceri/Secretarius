# Expérience 1 — RAG classique vs indexation par expressions

**Date :** 2026-04-09  
**Corpus :** `corpus_synth_indexed_1000.jsonl` — 200 chunks, 1 chunk/document  
**Requêtes :** synthétiques générées par DeepSeek à partir du texte brut  
**Embeddings :** BGE-M3 (1024 dim, CPU, normalisés)  
**Vérité terrain :** requête i → chunk i (tâche 1-to-1, N=200)

## Conditions testées

| ID | Condition | Requête | Document |
|----|-----------|---------|----------|
| A | RAG classique | embedding texte brut | embedding chunk brut |
| B | Expressions jointes | embedding texte brut | embedding expressions jointes (1 vecteur) |
| C | Late interaction asymétrique | embedding texte brut | max-sim sur embeddings expressions individuels |
| D | Late interaction symétrique | expressions extraites par LoRA, embeddings individuels | max-sim sur embeddings expressions individuels |

D est la vraie late interaction : `score = Σ_i max_j cos(expr_query_i, expr_doc_j)`

## Résultats

| Condition | MRR | Recall@1 | Recall@5 | Recall@10 | n |
|-----------|-----|----------|----------|-----------|---|
| **A) RAG classique** | **0.7016** | **0.5200** | **0.9750** | 1.0000 | 200 |
| B) Expressions jointes | 0.6776 | 0.4650 | 0.9900 | 1.0000 | 200 |
| D) Late interaction symétrique | 0.6485 | 0.4398 | 0.9581 | 0.9895 | 191 |
| C) Late interaction asymétrique | 0.5918 | 0.3550 | 0.9550 | 1.0000 | 200 |

*n=191 pour D : 9 requêtes sans expressions extraites par LoRA.*

## Observations

- Le RAG classique gagne sur MRR et R@1 dans toutes les conditions.
- La late interaction symétrique (D) remonte bien par rapport à l'asymétrique (C) : +0.057 MRR, +0.085 R@1 — confirme l'importance d'utiliser des vecteurs multiples côté requête.
- Les expressions jointes (B) sont très proches du RAG classique et meilleures en R@5.

## Analyse

**Biais du benchmark (majeur) :**  
Les requêtes ont été générées par DeepSeek à partir du texte brut. BGE-M3 retrouve donc naturellement le même texte (A), ce qui favorise structurellement le RAG classique. Pour un test équitable, il faudrait des requêtes construites indépendamment du texte — par exemple générées à partir des expressions seules, ou issues d'utilisateurs réels.

**Pourquoi C < D :**  
La late interaction asymétrique utilise un seul vecteur requête : seule la meilleure expression du document contribue au score. La version symétrique somme les maxima sur toutes les expressions de la requête, ce qui est beaucoup plus riche.

**Pourquoi D < A malgré la symétrie :**  
1. Biais de benchmark (voir ci-dessus).
2. Les expressions LoRA sont des fragments verbatim courts (2-5 mots), peu alignés sémantiquement avec des questions naturelles en dense retrieval.
3. Les expressions de requête sont extraites d'une question, pas d'un texte — le modèle LoRA a été entraîné sur des textes, pas des questions.

## Suite — Expérience 2 (à faire)

Pour lever le biais principal, générer les requêtes **à partir des expressions** plutôt que du texte brut : demander à DeepSeek "quelle question ces expressions-clés permettent-elles de retrouver ?" Cela avantagerait structurellement D et B, et donnerait une image plus équilibrée.

Fichiers : `eval_rag_vs_expressions.py`, `eval_rag_queries.json`, `eval_rag_query_exprs.json`, `eval_rag_results.json`
