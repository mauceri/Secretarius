# Design — Génération de page wiki par phi-4 local via passages centraux

Date : 2026-07-09

## Objectif

Remplacer, dans l'ingestion du wiki (`Wiki_LM/tools/ingest.py`), **l'appel LLM
externe (Euria/DeepSeek) qui génère la page source** par une version **locale
phi-4**. phi-4-mini ne pouvant absorber les ~12 000 caractères que l'ingestion
donne aujourd'hui au LLM, on insère en amont une **sélection de passages centraux**
qui comprime la source pour la faire tenir dans les 2048 tokens de phi-4.

C'est un **remplacement d'une étape existante**, pas un nouveau greffon. Le format
de la page produite reste **identique** à l'actuel, pour que tout l'aval (linkify
des `[[c-]]`/`[[e-]]`, `index.md`, recherche `/q`) continue de fonctionner sans
changement.

## Ce que fait l'ingestion aujourd'hui (état des lieux)

`ingest.py` (docstring lignes 3-10) :
1. Lit la source (arxiv / Wikipédia / HTML / PDF / fichier — lecteurs dédiés).
2. Tronque à 12 000 caractères (`_truncate`).
3. **Appelle le LLM (`from llm import LLM`) pour produire `src-<slug>.md`** : une page
   structurée (frontmatter YAML + corps + section « Concepts et entités mentionnés »
   avec des lignes `- entité: …` / `- concept: …`).
4. Extrait les concepts/entités → crée/enrichit leurs pages (`e-*`, `c-*`), linkifie.
5. Met à jour `index.md`, journalise dans `log.md`.

**Ce design ne touche QUE l'étape 3, et uniquement la génération de la page source.**

## Périmètre v1

- **Dans** : remplacer l'appel LLM de génération de `src-<slug>.md` (résumé + liste
  concepts/entités) par sélection de passages centraux → phi-4 nu → assemblage.
- **Hors** (différé) : l'enrichissement des **pages d'entités** (les appels LLM
  séparés, restent Euria en v1) ; le parsing RST/EDU ; les **longs documents
  techniques** (interprétabilité mécanique) ; les **verticaux métier**.

## Choix de conception et justifications

- **Sélection = PacSum** (Zheng & Lapata, ACL 2019) : centralité de phrase par
  **embeddings** (BGE-M3, déjà utilisé dans le projet) avec **biais de position**.
  Retenu contre LexRank/centroïde-MEAD, balayés par les plongements. La centralité
  fondée sur embeddings + position est « le meilleur des mondes » : robuste, pas de
  graphe O(n²) lourd requis, exploite l'encodeur déjà en place.
- **phi-4 produit du JSON contraint, pas du markdown.** phi-4-mini est faible sur
  l'adhérence au format (YAML/sections). Il émet donc `{"resume": str,
  "concepts": [str], "entites": [str]}` via `json_schema` (comme le routeur Tiron),
  et **le code assemble `src-<slug>.md` de façon déterministe**. On sépare le
  *contenu* (phi-4) du *format* (code fiable). Le titre, la date, l'URL, la catégorie
  viennent des **métadonnées** (déterministes), pas de phi-4.
- **Contexte figé à 2048.** On ne touche pas au `-c 2048` du serveur 8998 (il sert le
  routeur interactif). Budget : ~150 tk d'instructions + ~300 tk de sortie ⇒
  **~1600 tk (~6000 caractères) pour les passages sélectionnés**. La sélection
  comprime donc la source à ~6k caractères. Suffisant pour les captures moyennes ;
  c'est **la raison structurelle** de différer les longs documents techniques
  (impossible de comprimer 50 pages en 6k car. en une passe).
- **phi-4 nu via 8998, scale lora 0 par-requête** (mécanisme validé sur la FAQ :
  `"lora":[{"id":0,"scale":0}]` dans la requête `/v1/chat/completions`). Pas de
  second serveur, pas d'état global modifié, port prod intact.

## Architecture

```
source (lecture existante, inchangée)
  → nettoyage (clean_text, copié depuis chunk_data.py — dépôt séparé)
  → segmentation en phrases (nltk FR, copié depuis chunk_data.py)
  → embeddings BGE-M3 par phrase
  → centralité PacSum (similarité asymétrique + biais de position)
  → sélection des phrases top-score jusqu'au budget ~1600 tk, ordre d'origine préservé
  → phi-4 nu (json_schema) → {resume, concepts[], entites[]}
  → assemblage déterministe de src-<slug>.md (format actuel)
  → (linkify / index / log : machinerie existante, inchangée)
```

## Composants et interfaces

### 1. `Wiki_LM/tools/central_passages.py` (nouveau)

Responsabilité unique : d'un texte source, renvoyer les passages centraux tenant
dans un budget de tokens.

- `select_central_passages(text: str, budget_tokens: int = 1600, embed_fn=None) -> str`
  - segmente en phrases (nltk FR), embarque (BGE-M3), score par PacSum, sélectionne
    les phrases de plus haut score **dans l'ordre d'origine** jusqu'au budget, et
    renvoie le texte concaténé.
- PacSum : pour la phrase *i*, centralité =
  `λ1·Σ_{j<i} sim(i,j) + λ2·Σ_{j>i} sim(i,j)` après soustraction d'un seuil β
  (les similarités négatives sont ignorées), `sim` = cosinus BGE-M3. Hyperparamètres
  `λ1, λ2, β` avec les valeurs de départ du papier, surchargeables.
- `embed_fn` injectable (test avec un stub ; prod = encodeur BGE-M3 du projet).
- Texte vide / trop court → renvoie le texte nettoyé tel quel (pas de sélection).

### 2. `Wiki_LM/tools/page_llm_phi4.py` (nouveau)

Responsabilité unique : des passages centraux, produire le contenu structuré via
phi-4 nu.

- `generate_page_content(passages: str, base_url: str) -> dict`
  - POST `/v1/chat/completions` sur `base_url` (8998) avec
    `"lora":[{"id":0,"scale":0}]` (phi-4 nu) et `json_schema` imposant
    `{"resume": str, "concepts": [str], "entites": [str]}`.
  - Prompt : « À partir UNIQUEMENT des passages fournis, rédige un résumé fidèle et
    concis en français, et liste concepts et entités qui y figurent. N'invente rien. »
  - Erreur réseau / JSON invalide → exception explicite remontée à l'appelant.

### 3. `Wiki_LM/tools/ingest.py` (modifié, chirurgical)

- Remplacer l'appel LLM de génération de la page source par :
  `passages = select_central_passages(texte_source)` puis
  `contenu = generate_page_content(passages, PHI4_BASE)`, puis **assembler**
  `src-<slug>.md` à partir de `contenu` + métadonnées (titre/date/url/catégorie) au
  **format actuel** (frontmatter + corps + section « Concepts et entités mentionnés »).
- Réutiliser les fonctions existantes d'assemblage/linkify (`_linkify_concepts_section`,
  injection d'URL, etc.). Ne pas toucher aux étapes 4-5 ni à l'enrichissement des
  pages d'entités.

### 4. Harnais d'évaluation (nouveau, réutilise le juge existant)

- Réutilise le juge DeepSeek de `gen_corpus_qa` (`eval_qa.py`).
- Jeu de test : quelques sources brutes déjà ingérées (ex. « pêcheurs de jade »,
  « Maître Eckhart », un PDF, une note courte) + **leurs pages Euria actuelles comme
  référence**.
- Le juge note deux axes : **fidélité** (rien hors-source) et **couverture** (le
  contenu central de la page Euria est présent). Verdict = la page phi-4 égale ou
  approche la page Euria.

## Critère de succès (« résumé potable »)

Défini d'avance et mesurable — c'est le point qui manquait historiquement :

- **Fidélité** : la page phi-4 n'introduit aucun fait absent des passages (anti
  « Henri IV / Henry IV »).
- **Couverture** : le contenu central de la page de référence (Euria) s'y retrouve.

Seuils cibles chiffrés à figer avec le premier run (comme le seuil FAQ l'a été) ;
la référence Euria fournit l'étalon.

## Gestion d'erreurs

- Source vide / trop courte → passages = texte tel quel ; phi-4 rend une page minimale.
- phi-4 injoignable ou JSON invalide → l'ingestion signale l'échec pour cette source
  et passe à la suivante (ne bloque pas la file), sans écrire de page corrompue.
- Budget dépassé malgré sélection (source très longue) → tronquer les passages
  sélectionnés au budget ; noter le cas (candidat au périmètre « longs docs » différé).

## Tests

- `select_central_passages` : ordre d'origine préservé ; respect du budget ; texte
  court renvoyé tel quel ; PacSum favorise les phrases centrales (stub embeddings).
- `generate_page_content` : la requête contient bien `lora:[{id:0,scale:0}]` et le
  `json_schema` ; parse le JSON en `{resume, concepts, entites}` (mock HTTP).
- Assemblage `src-*.md` : format identique à l'actuel (frontmatter + section entités),
  déterministe à partir d'un `contenu` donné.
- Éval : le harnais produit les deux notes juge sur le jeu de test.

## Assets réutilisés (pour éviter de dupliquer)

- Segmentation/nettoyage : fonctions **copiées** depuis `~/indexation_wiki40b/chunk_data.py`
  (phrases FR, `clean_text`) — dépôt séparé, on vendorise plutôt qu'importer.
- Embeddings : BGE-M3 déjà en usage (routeur, recherche wiki).
- Inférence phi-4 nu : mécanisme lora-par-requête validé sur la FAQ.
- Juge DeepSeek : `gen_corpus_qa/eval_qa.py`.
- Lecture de source + assemblage/linkify : fonctions existantes d'`ingest.py`.

## Hors périmètre (rappel)

Enrichissement des pages d'entités (reste Euria) ; RST/EDU ; longs documents
techniques ; verticaux métier. Chacun fera l'objet de son propre cadrage.
