# Adaptateur LoRA « QA-sur-document » générique pour Tiron — Design

## Contexte et motivation

Tiron (phi-4-mini + adaptateur de routage, déployé le 2026-07-06) ne sait
aujourd'hui que router des commandes vers des sous-agents. Toute question qui
n'est pas une commande reçoit un message d'échec (« Je n'ai pas identifié de
commande… »), y compris les questions générales sur Secretarius lui-même
(« quel modèle vous anime ? », « que sais-tu faire ? »).

Ce chantier donne à Tiron un peu de latitude en compilant une **compétence**
(pas des faits) dans un adaptateur LoRA, selon le principe Skill-to-LoRA (S2L,
arXiv 2606.16769). Il sert aussi de brique préparatoire à l'objectif 2 (QA/
résumé sur le wiki) : l'adaptateur produit ici est réutilisable tel quel.

Recherche préalable (session 2026-07-06, mémoire
[[reference_context_economy_research]]) : T2L et LatentSkill (hyperréseaux)
mis en réserve (prérequis lourds, dépôts vides ou coûteux) ; S2L retenu car
c'est déjà notre pratique (pipeline `gen_corpus/`).

## Idée centrale : séparer les faits du comportement

- **Les faits** (nom du modèle, version, fonctionnalités, capacités wiki/gog)
  restent dans un **document texte** édité à la main, fourni en contexte à
  l'inférence — donc toujours à jour, jamais périmé, aucun réentraînement si
  l'architecture change.
- **Le comportement** (répondre à une question en s'appuyant *strictement* sur
  le document fourni : concis, en français, ancré sans invention, avec **refus
  propre** si la réponse n'y est pas) est compilé dans l'adaptateur.

C'est plus fin qu'un S2L « pur » qui retirerait *tout* du contexte : ici on ne
compile que la compétence de lecture-réponse, pas le contenu. Le coût de
contexte devient petit **et borné** (un seul document, pas le mur des schémas
d'outils du tool-calling natif) et stable dans le temps.

## Décisions tranchées (brainstorming 2026-07-07)

1. **Un seul adaptateur générique** « QA-sur-document », pas un adaptateur par
   domaine. Les domaines servent à assurer la **diversité** du corpus
   d'entraînement (éviter le surapprentissage du vocabulaire d'un seul
   document).
2. **Portée : générique** (répondre à partir de n'importe quel document
   fourni), pas spécialisé Secretarius. Réutilisable pour le wiki (objectif 2).
   Le document Secretarius devient un simple paramètre d'entrée.
3. **3 domaines seed pour démarrer** : configuration matériel/logiciel,
   capacités wiki, capacités gog. Extensible plus tard (comptabilité, légal,
   clientèle…) — non implémenté dans ce chantier (YAGNI).
4. **Teacher = DeepSeek** (accord explicite de l'utilisateur donné ;
   `DEEPSEEK_API_KEY` requis). Juge de l'éval A/B = DeepSeek aussi.
5. **Périmètre du chantier : s'arrête à la validation A/B.** Produire le
   pipeline de données + l'adaptateur + le protocole d'évaluation nu-vs-adaptateur.
   L'intégration OpenClaw (routage document + dispatch + Telegram) est un
   chantier suivant, décidé seulement si l'adaptateur bat le modèle nu.

## Architecture de l'adaptateur (inférence)

Format ChatML d'un exemple :

- `system` : instruction de lecture-réponse ancrée (répondre uniquement à
  partir du document, concis, français, refuser si absent).
- `user` : `Document:\n<document>\n\nQuestion: <question>`
- `assistant` : la réponse ancrée (ou le refus propre si hors-document).

Le document reste en contexte ; l'adaptateur compile la compétence de réponse.

## Pipeline de données (généralisation de `gen_corpus/`)

Nouveau répertoire `gen_corpus_qa/` (copie adaptée de `gen_corpus/`, pour ne
pas toucher au pipeline routeur qui est en production).

| Fichier routeur (`gen_corpus/`) | Équivalent QA (`gen_corpus_qa/`) | Rôle |
|---|---|---|
| `intentions.json` | `domaines.json` | 3 domaines, chacun pointant un document seed |
| (documents absents) | `documents/` | les 3 documents seed (config HW/SW, wiki, gog) |
| `seed.json` | `seed.json` | exemples amorces de triplets (document, question, réponse), **dont des exemples négatifs** (question hors-document → refus) |
| `promptGenGEPA.py` | réutilisé | optimise le prompt générateur (document, type-question, registre) → (question, réponse ancrée) |
| `GEPAPrompt.txt` | régénéré | prompt optimisé pour la génération QA |
| `generate_corpus.py` | adapté | produit le corpus de triplets ; teacher = **DeepSeek** |
| `to_lora_format.py` | adapté | ChatML avec le nouveau `SYSTEM_PROMPT` QA |

Taille cible du corpus : ~1500-2000 exemples (comme le routeur), incluant une
part explicite d'exemples négatifs (refus hors-document) pour ancrer ce
comportement, essentiel sur un SLM faible.

## Entraînement

Réutilise `lora_slm/lora_train.py` avec les hyperparamètres déjà éprouvés sur
l'adaptateur routeur : R16, α32, LR 2e-4, ~6 epochs. Conversion en GGUF LoRA
via `convert_lora_to_gguf.py` (build-rocm). Checkpoints hors dépôt (convention
existante `/home/mauceric/lora_slm/checkpoints/`). Brique déjà maîtrisée, pas
de nouveauté.

## Validation A/B (maillon le plus incertain)

Évaluer du QA ancré est intrinsèquement plus mou que l'exactitude JSON du
routeur. Protocole :

1. **Jeu de test tenu à l'écart** : triplets (document, question, réponse-
   référence) non vus à l'entraînement, incluant des questions hors-document.
2. **LLM-juge (DeepSeek)** notant chaque réponse candidate sur : (a) exactitude
   factuelle vs document, (b) ancrage (aucune information hors-document), (c)
   concision et langue (français), et (d) comportement de refus correct sur les
   questions hors-document.
3. **Comparaison nu vs adaptateur** : phi-4-mini nu (avec le même prompt et le
   même document en contexte) contre phi-4-mini + adaptateur, sur les mêmes
   questions.
4. **Inspection manuelle** d'un échantillon pour valider le jugement automatique.

**Critère de succès** : l'adaptateur bat nettement le modèle nu, en particulier
sur l'ancrage et le refus hors-document. Si l'écart est marginal, on documente
et on s'arrête (l'intégration ne se justifie pas) — c'est le point de décision
prévu.

### Deux environnements de test de l'adaptateur

L'adaptateur doit être testable par deux voies, utiles à la fois pour le
confort d'itération et comme contrôle croisé (une divergence entre les deux
signalerait un problème de conversion GGUF — panne déjà rencontrée, cf. mémoire
[[project_lora_slm_session_20260630]]) :

1. **Jupyter (transformers + peft)** : charge le checkpoint PEFT **directement**
   (sans conversion GGUF), phi-4-mini de base + adaptateur monté en Python.
   Permet d'itérer et d'inspecter les réponses avant même l'étape de conversion,
   et sert de référence « fidèle » du comportement de l'adaptateur.
2. **Serveur llama.cpp (GGUF + `--lora`)** : le GGUF converti servi par
   `build-rocm/bin/llama-server`, interrogé par HTTP — c'est l'environnement de
   production réel (ROCm, mesure de vitesse représentative). Réutilise le patron
   de `gen_corpus/eval_adapter.py` (paramètre `--base-url`).

Le harnais d'évaluation A/B doit pouvoir cibler l'une ou l'autre voie, afin que
le même jeu de test puisse être passé dans les deux et que les résultats soient
comparés.

## Ce qui est hors périmètre (chantiers suivants, non traités ici)

- Intégration OpenClaw : détection « question générale » côté routeur, sélection
  du document à injecter, dispatch dans `derisk-deleg`, test Telegram.
- Extension à d'autres domaines (comptabilité, légal, clientèle).
- Application au wiki (objectif 2) : sélection de la/les page(s) pertinente(s)
  par recherche BGE-M3 — l'adaptateur produit ici sera réutilisé, mais la
  chaîne de recherche est un chantier distinct.

## Risques identifiés

- **Valeur ajoutée non garantie** : phi-4-mini-instruct sait déjà faire du QA
  basique sur un document fourni. La valeur de l'adaptateur (ancrage strict,
  refus, concision) est une hypothèse à mesurer — d'où le A/B tôt comme point
  de contrôle avant tout investissement ultérieur.
- **Qualité du jugement automatique** : le LLM-juge peut être indulgent ou
  incohérent ; l'inspection manuelle d'un échantillon est un garde-fou
  nécessaire.
- **Diversité insuffisante** : avec seulement 3 domaines, risque de
  surapprentissage du style des documents seed ; à surveiller dans l'éval
  (tester sur un document d'un style non vu si possible).
