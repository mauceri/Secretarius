# Détection et réponse « question-Secretarius » pour Tiron — Design

## Contexte et motivation

Tiron (phi-4-mini + adaptateur de routage, déployé le 2026-07-06) sait router
les commandes explicites (`/q`, `/c`, `/chercher`…) vers les sous-agents wiki et
gog, à 93 %. Mais toute question générale sur Secretarius lui-même — « quel
modèle vous anime ? », « que sais-tu faire avec le wiki ? », « comment je
connecte Gmail ? » — est classée `out_of_scope` par l'adaptateur (`command:
null`) et reçoit un message d'échec.

Ce chantier donne à Tiron la latitude de **répondre** à ces questions, sans
encombrer le contexte de tous les tours de conversation.

**Leçon du chantier précédent** (adaptateur QA générique, verdict négatif le
2026-07-08, voir `gen_corpus_qa/RESULTATS_AB.md` et mémoire
[[project_qa_adapter_verdict]]) : phi-4-mini nu répond déjà à 0.82 (juge
DeepSeek) sur du QA-sur-document — un adaptateur LoRA de **réponse** n'apporte
rien. On applique donc ici le principe « mesurer le modèle nu avant
d'entraîner » : on n'entraîne un adaptateur que si un mécanisme plus léger
échoue mesurablement.

## Idée centrale : séparer *détecter* de *répondre*

Deux compétences distinctes, mesurées indépendamment :

1. **Détecter** qu'un message est une question-Secretarius (par opposition à une
   commande wiki/gog ou à un hors-sujet général). C'est un problème de
   **classification** — le maillon incertain.
2. **Répondre** à la question à partir d'un document Secretarius fourni en
   contexte. Compétence déjà présente dans phi-4-mini nu (mesurée à 0.82).

L'adaptateur, si un jour nécessaire, compilerait la **distinction** (routage),
jamais la **réponse**.

## Décisions tranchées (brainstorming 2026-07-08)

1. **Détection = extension du GogGate BGE-M3** (classifieur par centroïdes déjà
   en place dans `router_service/router.py`), avec un 4ᵉ centroïde
   `secretarius`. **Aucun réentraînement LoRA.** On mesure d'abord, avec le
   mécanisme le moins cher, et on n'escalade qu'en cas d'échec mesuré (voir
   « Échelle d'escalade » ci-dessous).

### Pourquoi le centroïde BGE-M3 plutôt qu'un classifieur entraîné

- **Il existe déjà** : le `GogGate` est déjà un classifieur par centroïdes
  BGE-M3 en production (portail de confiance gog). Ajouter `secretarius` = une
  ligne de plus dans sa matrice de centroïdes, pas un nouveau composant à
  entraîner, servir et maintenir (modification chirurgicale).
- **Coût quasi nul** : un centroïde se calcule en secondes de CPU (moyennes
  d'embeddings), sans données étiquetées volumineuses ni GPU ni contention avec
  la prod — à opposer aux 12h GPU du chantier QA précédent.
- **BGE-M3 est taillé pour ça** : modèle d'embedding multilingue (1024 dim)
  entraîné pour la similarité sémantique ; la séparation par plus proche
  centroïde est son cas d'usage naturel, et il est déjà chargé en mémoire.
- **Compromis assumé** : le centroïde est le classifieur le plus simple — il
  suppose des classes séparables par une frontière linéaire en cosinus. Un
  modèle entraîné apprendrait une frontière plus fine. Si la distinction
  commande/question est subtile, le centroïde peut ne pas suffire — d'où le
  risque n°1 et le « mesurer d'abord ».

### Échelle d'escalade (du moins cher au plus cher)

Le repli n'est pas binaire. En cas d'échec mesuré, on monte d'un cran à la fois :

1. **Centroïde BGE-M3** (coût ~nul) — ce qu'on teste dans ce chantier.
2. **Classifieur léger sur embeddings BGE-M3** (régression logistique ou petit
   MLP entraîné sur les embeddings déjà calculés) — pas de GPU, quelques
   secondes CPU, quelques centaines d'exemples ; apprend une vraie frontière de
   décision au lieu de comparer à une moyenne. Repli intermédiaire privilégié.
3. **Réentraînement de l'adaptateur LoRA routeur** (~12h GPU, services prod à
   l'arrêt) — dernier recours, seulement si les deux échelons précédents
   échouent.

Chaque montée d'échelon est un chantier distinct, décidé au vu de la mesure.
2. **Réponse = appel déterministe** à phi-4 **nu** : llama-server 8998 avec le
   scale de l'adaptateur routeur mis à **0**, et le document Secretarius
   (~617 tokens) injecté en contexte. Réutilise le hot-swap par requête validé
   le 2026-07-01 et le patron `set_lora_scale` de `gen_corpus_qa/eval_qa.py`.
3. **Périmètre : validation locale hors-ligne uniquement.** On produit et mesure
   la détection + la réponse, et on écrit un verdict. On **ne touche pas** au
   `router_service` en production ni à `derisk-deleg`. Le câblage réel (insérer
   la branche dans `route_message` + dispatch derisk-deleg + déploiement + test
   Telegram) est un **chantier suivant**, décidé seulement si la validation
   locale est concluante.

## Architecture

```
Message entrant
   │
   ▼
[GogGate.classify(message)]  ← BGE-M3, 4 centroïdes (wiki/gog/secretarius/null)
   │
   ├─ "wiki"        → routage inchangé (adaptateur extrait /c, /q, …)
   ├─ "gog"         → routage inchangé (+ gog_confident)
   ├─ "null"        → out_of_scope (message d'échec, inchangé)
   └─ "secretarius" → repondre_secretarius(message)
                         │
                         ▼
                    POST 8998  (adaptateur scale=0 → phi-4 nu)
                    system = instruction QA ancrée
                    user   = "Document:\n<secretarius.md>\n\nQuestion: <message>"
                         │
                         ▼
                    réponse en langage naturel
```

**En validation locale (ce chantier), ce flux est simulé hors-ligne** par un
script de mesure : `classify()` sur un jeu de test étiqueté, puis
`repondre_secretarius()` sur les cas classés `secretarius`, puis compilation du
verdict. Le flux réel n'est pas branché ici.

**Règle de priorité des classes :** en cas de doute, une **vraie commande**
l'emporte sur `secretarius`, pour ne jamais casser le routage wiki/gog existant
(93 %). Concrètement, `secretarius` ne gagne que s'il est argmax **et** au-dessus
d'un seuil dédié (à l'image du `SEUIL_GOG` déjà en place).

## Composants

### A. Document Secretarius unifié

Fichier `gen_corpus_qa/documents/secretarius.md` : concaténation des trois
documents seed existants (`config-materiel-logiciel.md`, `capacites-wiki.md`,
`capacites-gog.md`), ~617 tokens au total. Source de vérité factuelle, éditée à
la main, jamais compilée dans des poids — donc toujours à jour.

### B. Classifieur étendu (`GogGate`)

Extension de la classe `GogGate` (`router_service/router.py`) : ajout d'un
centroïde `secretarius` calculé à partir d'exemples de questions-Secretarius
tirés du corpus QA (`gen_corpus_qa/corpus_qa.jsonl`, champ `question`, plusieurs
registres). Nouvelle méthode :

```
classify(message: str) -> "wiki" | "gog" | "secretarius" | "null"
```

qui coexiste avec le `gog_confident()` existant (non modifié). La matrice de
centroïdes passe de 3 à 4 lignes.

### C. Jeu de test de détection (étiqueté)

Construit à partir de matériaux existants, sans nouvelle génération :
- **questions-Secretarius** : depuis `gen_corpus_qa/corpus_qa.jsonl` (questions
  `factuelle`/`reformulation` sur les 3 docs) — étiquette `secretarius`.
- **commandes wiki/gog** : depuis `gen_corpus/corpus.jsonl` — étiquettes `wiki`
  et `gog`.
- **hors-sujet général** : depuis `gen_corpus/corpus.jsonl`, variantes
  `aide_generale` et `conversation_libre` — étiquette `null`.

Les exemples servant à **calculer** le centroïde `secretarius` sont disjoints de
ceux du jeu de **test** (pas de fuite).

### D. Fonction de réponse

```
repondre_secretarius(question: str) -> str
```

POST au llama-server 8998, scale de l'adaptateur routeur à 0 (phi-4 nu),
document Secretarius injecté, question posée. Réutilise `set_lora_scale` de
`gen_corpus_qa/eval_qa.py`.

### E. Harnais de mesure + verdict

Script produisant `RESULTATS.md` avec :
1. **Matrice de confusion** du classifieur sur le jeu de test complet — le point
   critique : combien de vraies commandes wiki/gog sont détournées vers
   `secretarius`, et combien de questions-Secretarius sont manquées.
2. **Qualité des réponses** phi-4 nu sur un échantillon de questions-
   Secretarius : notation par juge DeepSeek (patron `judge_score` de
   `eval_qa.py`), complétée par une inspection manuelle de l'échantillon comme
   garde-fou (le juge peut être indulgent).

## Critère de succès (point de décision)

- **Détection** : rappel correct de `secretarius` **sans** dégrader le routage
  existant — quasi zéro vraie commande wiki/gog détournée (seuil indicatif :
  < ~2-3 % de commandes wiki/gog reclassées `secretarius`).
- **Réponse** : réponses correctes et ancrées sur l'échantillon (référence : le
  0.82 déjà mesuré en QA).
- **Si la détection échoue nettement** → stop, on documente, et on monte d'un
  cran sur l'échelle d'escalade (classifieur léger sur embeddings, puis en
  dernier recours réentraînement de l'adaptateur routeur) — chaque cran étant un
  chantier distinct. Si les deux critères passent → feu vert pour le chantier
  d'intégration OpenClaw.

## Tests

- Unitaire `classify()` : messages étiquetés à la main (une commande wiki, une
  gog, une question-Secretarius, un out_of_scope) → assert la classe attendue.
- Unitaire `repondre_secretarius()` : vérifie le format de l'appel (scale 0,
  document injecté) avec un llama-server mocké ; l'appel réel est exercé par le
  harnais de mesure (E), pas par le test unitaire.
- Harnais de mesure (E) : pas un test unitaire — produit la matrice de confusion
  et les notes de réponse.

## Risques identifiés

1. **Frontière commande/question** (principal) : « comment interroger le
   wiki ? » (→ secretarius) vs « /q pêcheurs de jade » (→ commande wiki). Si
   BGE-M3 ne les sépare pas, la détection échoue — révélé tôt par la mesure (C+E).
2. **Représentativité du centroïde** : les exemples viennent du corpus QA
   (généré par DeepSeek) ; un vrai message Telegram peut être formulé autrement.
   Atténué en tirant les exemples de plusieurs registres.
3. **Contamination du `null`** : les questions-Secretarius étaient jusqu'ici
   dans `out_of_scope` ; vérifier que le centroïde `null` ne les capture pas à
   la place de `secretarius`.

## Hors périmètre (chantiers suivants)

- Intégration OpenClaw : insertion de la branche `secretarius` dans
  `route_message`, dispatch dans `derisk-deleg`, déploiement, test Telegram E2E.
- Montées sur l'échelle d'escalade (classifieur léger sur embeddings, puis
  réentraînement de l'adaptateur routeur) si le centroïde échoue.
- Extension du document Secretarius à d'autres sujets (comptabilité, légal…).

## Artefacts réutilisés (chantier QA précédent)

- `gen_corpus_qa/documents/*.md` — les 3 documents seed (base du document
  unifié).
- `gen_corpus_qa/corpus_qa.jsonl` — source des questions-Secretarius (centroïde
  + test).
- `gen_corpus_qa/eval_qa.py` — `set_lora_scale`, juge DeepSeek, réutilisés.
- `router_service/router.py` — `GogGate` à étendre.
- `gen_corpus/corpus.jsonl` — source des commandes wiki/gog et des out_of_scope.
