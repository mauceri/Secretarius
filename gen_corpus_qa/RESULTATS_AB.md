# Verdict A/B — Adaptateur QA-sur-document générique

**Date :** 2026-07-08
**Chantier :** `docs/superpowers/specs/2026-07-07-adaptateur-qa-document-design.md`
**Statut : CLOS — verdict négatif, pas d'intégration.**

## Résultat

Test A/B (juge DeepSeek, échelle 0..1), phi-4-mini nu vs phi-4-mini + adaptateur
LoRA « QA-sur-document », sur un sous-ensemble de 10 exemples du jeu d'éval tenu
à l'écart (`corpus_qa_eval.jsonl`, 150 ex.), backend PEFT/CPU.

| Condition | Note moyenne (10 ex.) |
|-----------|-----------------------|
| phi-4-mini **nu** | **0.820** |
| phi-4-mini **+ adaptateur** | 0.760 |
| **DELTA** | **-0.060 (-7,3 %)** |

L'adaptateur **dégrade** la performance au lieu de l'améliorer.

## Interprétation

1. **phi-4-mini nu est déjà bon en QA ancré** (0.82 selon le juge DeepSeek, qui
   est aussi le teacher ayant généré les réponses de référence). Autrement dit,
   sur ce type de tâche, phi-4 nu fait déjà presque aussi bien que DeepSeek —
   il n'y a quasiment pas de marge à récupérer par fine-tuning.
2. Le critère de succès de la spec (« l'adaptateur bat *nettement* le modèle nu,
   en particulier sur l'ancrage et le refus ») n'est pas atteint : l'écart est
   négatif et significatif.
3. Conformément à la spec (« si l'écart est marginal, on documente et on
   s'arrête »), le chantier s'arrête ici. L'intégration OpenClaw n'est pas
   justifiée.

## Cause racine (hypothèse)

Le principe S2L (Skill-to-LoRA) fonctionne quand le modèle nu **échoue** sur la
tâche et que l'adaptateur compile une compétence absente — c'est le cas du
routeur (sortie JSON contrainte, 93 %). Ici, la compétence « répondre à partir
d'un document » est **déjà présente** dans phi-4-mini-instruct : il n'y a rien à
compiler, et l'entraînement ne fait que perturber un comportement déjà correct
(léger surapprentissage du style du corpus).

**Leçon transférable :** un adaptateur S2L n'a d'intérêt que si le modèle nu
échoue mesurablement sur la tâche. Toujours mesurer le modèle nu **avant**
d'entraîner. Voir le pivot de conception qui en découle (objectif 1 : compiler
la *distinction* gog/wiki/secretarius, PAS la réponse).

## Incidents techniques rencontrés (Task 7)

- **OOM HIP** au 1er essai : `--max_len 2048` + packing gonflaient chaque batch à
  2048 tokens alors que le corpus réel plafonne à 419 tokens (p99=407). Corrigé
  par `--max_len 512`.
- **Crash HIP « no kernel image available » déterministe** aux essais suivants,
  avec les services de prod actifs (`slm-llama_cpp` + `tiron-router`). Cause =
  contention GPU (l'iGPU partage 30 Go et un plafond ROCm ~15,3 Gio ; erreur HIP
  asynchrone signalée sur `fixed_cross_entropy` mais fautif réel ailleurs).
  Résolu en arrêtant temporairement les services de prod — l'entraînement passe
  alors sans erreur. Services redémarrés après entraînement.
- **Durée d'entraînement : 12h46** (vs 8h50 estimé) — ~174 s/step, 306 steps,
  6 epochs, 1350 ex. iGPU memory-bandwidth-bound.
- **Éval GPU également bloquée** par la contention/erreur HIP → A/B fait en CPU
  (d'où le sous-ensemble de 10 ex. plutôt que les 150).

## Artefacts réutilisables (ne pas supprimer)

- `gen_corpus_qa/documents/{config-materiel-logiciel,capacites-wiki,capacites-gog}.md`
  — les 3 documents seed ; `config-materiel-logiciel.md` servira de document
  injecté dans la branche « question secretarius » de l'objectif 1.
- `gen_corpus_qa/corpus_qa.jsonl` (1500 ex.), `corpus_qa_train.jsonl` (1350),
  `corpus_qa_eval.jsonl` (150) — corpus QA généré par DeepSeek.
- `gen_corpus_qa/GEPAPrompt.txt` — prompt de génération optimisé (+ règle de
  fidélité syntaxique des commandes ajoutée manuellement).
- Checkpoint : `/home/mauceric/lora_slm/checkpoints/qa-document/checkpoint-306`
  (adaptateur produit ; conservé pour référence, non intégré).
- Pipeline `gen_corpus_qa/` complet (Tasks 1-6, 14 tests) — réutilisable pour
  l'extension du corpus routeur de l'objectif 1.

## Suite

L'objectif 1 (latitude de Tiron) est redéfini : compiler dans l'adaptateur
**routeur** la *distinction* gog / wiki / question-secretarius, et répondre aux
questions secretarius avec phi-4 **nu** + document config en contexte (chargé
seulement après routage). À brainstormer.
