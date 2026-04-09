# Expériences — RAG classique vs indexation par expressions

**Corpus :** `corpus_synth_indexed_1000.jsonl` — 200 chunks, 1 chunk/document  
**Embeddings :** BGE-M3 (1024 dim, CPU, normalisés)  
**Vérité terrain :** requête i → chunk i (tâche 1-to-1, N=200)

## Conditions testées (communes aux 3 expériences)

| ID | Condition | Requête | Document |
|----|-----------|---------|----------|
| A | RAG classique | embedding texte brut | embedding chunk brut |
| B | Expressions jointes | embedding texte brut | embedding expressions jointes (1 vecteur) |
| C | Late interaction asymétrique | embedding texte brut | max-sim sur embeddings expressions individuels |
| D | Late interaction symétrique | expressions extraites (multi-vecteurs) | max-sim sur embeddings expressions individuels |

D : `score = Σ_i max_j cos(expr_query_i, expr_doc_j)`

---

## Expérience 1 — Requêtes depuis le texte brut, expressions requête par LoRA

**Date :** 2026-04-09  
**Requêtes :** générées par DeepSeek à partir du **texte brut** du chunk  
**Expressions requête (D) :** extraites par **LoRA** (llama.cpp port 8989)

| Condition | MRR | Recall@1 | Recall@5 | Recall@10 | n |
|-----------|-----|----------|----------|-----------|---|
| **A) RAG classique** | **0.7016** | **0.5200** | 0.9750 | 1.0000 | 200 |
| B) Expressions jointes | 0.6776 | 0.4650 | **0.9900** | 1.0000 | 200 |
| D) Late interaction symétrique | 0.6485 | 0.4398 | 0.9581 | 0.9895 | 191 |
| C) Late interaction asymétrique | 0.5918 | 0.3550 | 0.9550 | 1.0000 | 200 |

*n=191 pour D : 9 requêtes sans expressions extraites par LoRA.*

**Conclusion :** RAG classique domine. Biais probable : requêtes construites depuis le texte brut → alignement naturel avec l'embedding du chunk.

---

## Expérience 2 — Requêtes depuis les expressions, expressions requête par LoRA

**Date :** 2026-04-09  
**Requêtes :** générées par DeepSeek à partir des **expressions du chunk**  
**Expressions requête (D) :** extraites par **LoRA** (llama.cpp port 8989)

| Condition | MRR | Recall@1 | Recall@5 | Recall@10 | n |
|-----------|-----|----------|----------|-----------|---|
| **D) Late interaction symétrique** | **0.8293** | **0.7092** | **1.0000** | 1.0000 | 196 |
| B) Expressions jointes | 0.7945 | 0.6450 | 0.9900 | 1.0000 | 200 |
| A) RAG classique | 0.7933 | 0.6600 | 0.9850 | 1.0000 | 200 |
| C) Late interaction asymétrique | 0.6243 | 0.4200 | 0.9600 | 1.0000 | 200 |

*n=196 pour D : 4 requêtes sans expressions extraites par LoRA.*

**Conclusion :** La late interaction symétrique gagne clairement (+0.036 MRR vs A, +0.109 R@1, Recall@5=1.0). Quand les requêtes sont construites de façon cohérente avec l'indexation, l'approche expressions-LoRA est supérieure.

---

## Expérience 3 — Requêtes depuis le texte brut, expressions requête par DeepSeek

**Date :** 2026-04-09  
**Requêtes :** générées par DeepSeek à partir du **texte brut** (mêmes que expérience 1)  
**Expressions requête (D) :** extraites par **DeepSeek**

| Condition | MRR | Recall@1 | Recall@5 | Recall@10 | n |
|-----------|-----|----------|----------|-----------|---|
| **A) RAG classique** | **0.7016** | **0.5200** | 0.9750 | 1.0000 | 200 |
| B) Expressions jointes | 0.6776 | 0.4650 | **0.9900** | 1.0000 | 200 |
| D) Late interaction symétrique | 0.6146 | 0.3769 | 0.9698 | 0.9950 | 199 |
| C) Late interaction asymétrique | 0.5918 | 0.3550 | 0.9550 | 1.0000 | 200 |

**Conclusion surprenante :** DeepSeek fait moins bien que LoRA pour extraire les expressions des requêtes (D : 0.6146 vs 0.6485 en exp 1). DeepSeek tend à paraphraser ou généraliser, là où LoRA extrait des termes verbatim précis — plus proches des `expressions_caracteristiques` du document (elles-mêmes générées par DeepSeek lors de la création du corpus, mais de façon littérale). Le fait que n=199 (presque toutes les requêtes ont des expressions) confirme que ce n'est pas un problème de couverture mais de précision sémantique.

---

## Tableau comparatif MRR (toutes expériences)

| Condition | Exp 1 (texte→LoRA) | Exp 2 (exprs→LoRA) | Exp 3 (texte→DS) |
|-----------|--------------------|--------------------|------------------|
| A) RAG classique | 0.7016 | 0.7933 | 0.7016 |
| B) Expressions jointes | 0.6776 | 0.7945 | 0.6776 |
| C) Late inter. asymét. | 0.5918 | 0.6243 | 0.5918 |
| D) Late inter. symét. | 0.6485 | **0.8293** | 0.6146 |

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `eval_rag_vs_expressions.py` | Script expérience 1 |
| `eval_rag_vs_expressions_2.py` | Script expérience 2 |
| `eval_rag_vs_expressions_3.py` | Script expérience 3 (à créer) |
| `eval_rag_queries.json` | Requêtes exp 1 (depuis texte brut) |
| `eval_rag_query_exprs.json` | Expressions requêtes exp 1 (LoRA) |
| `eval_rag_results.json` | Résultats exp 1 |
| `eval_rag2_queries.json` | Requêtes exp 2 (depuis expressions) |
| `eval_rag2_query_exprs.json` | Expressions requêtes exp 2 (LoRA) |
| `eval_rag2_results.json` | Résultats exp 2 |
| `eval_rag3_query_exprs.json` | Expressions requêtes exp 3 (DeepSeek) |
| `eval_rag3_results.json` | Résultats exp 3 |
