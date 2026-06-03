# Routage par intention — Design (Spec 1)

**Date :** 2026-06-02
**Branche :** v0.2.0-dev
**Statut :** validé, prêt pour le plan d'implémentation

## Contexte et motivation

L'architecture cible de v0.2.0 fait de Tiron un **orchestrateur léger** qui ne porte aucun
outil métier : il détecte l'intention d'une demande et la transmet telle quelle à un agent
spécialisé (wikilm, gog, superpowers), chacun portant son propre MCP, son propre modèle et
son propre prompt.

La raison est concrète : le Tiron actuel charge un prompt de **11 286 tokens** (29 outils + 67
commandes + bootstrap). Sur l'iGPU de sanroque (build llama.cpp CPU-only, ~32 tok/s), prefiller
ce prompt prend ~6 minutes — inutilisable. En réduisant Tiron à un routeur, son prompt devient
minuscule.

Mais avant de câbler quoi que ce soit dans OpenClaw (avec ses redémarrages de 3 minutes, ses
stalls Telegram et un wifi instable), on isole **l'unique inconnu réel** : sait-on détecter le
bon agent de façon fiable ? Ce spec construit un **harnais d'évaluation autonome**, hors
OpenClaw, qui répond à cette question et produit en même temps le **corpus réutilisable comme
dataset LoRA**.

## Principe de routage

Le routeur fait **uniquement de la détection d'intention**. Il ne sait rien des arguments :
pour « Pouvez-vous capturer cette url https://exemple.fr ? », il décide `wikilm` et transmet le
message **intact**. C'est l'agent wikilm qui extraira l'URL. On passe la patate chaude.

Conséquence : pour de la pure classification parmi 3-4 agents, un LLM de 3,8 B (Phi-4-mini) est
probablement surdimensionné, et son prefill CPU le rend lent même avec un prompt court. Le
projet dispose déjà de **BGE-M3** (embeddings 1024-dim, `Wiki_LM`), bien plus adapté à une
classification d'intention. Le harnais compare donc empiriquement deux routeurs sur le même
corpus, sans présupposer le gagnant.

## Catalogue d'agents

Liste déclarative `agents.json`. Chaque entrée : `name` + `description` (une ligne). La
description sert à la fois de contexte au routeur LLM et de base aux prototypes d'embeddings.

```json
{
  "agents": [
    {"name": "wikilm",      "description": "Capture, recherche et ingestion de connaissances : URLs à mémoriser, notes, tags, questions sur la base documentaire."},
    {"name": "gog",         "description": "Google Workspace : email (lire, chercher, envoyer), agenda, fichiers Drive."},
    {"name": "superpowers", "description": "Rédaction de textes longs, brainstorming, conception de plans et de specifications."},
    {"name": "clarify",     "description": "Intention floue, ambiguë ou hors-sujet : demander une précision à l'utilisateur."}
  ]
}
```

`clarify` est une cible de routage à part entière : quand aucune intention claire ne ressort,
le bon comportement est de demander une précision (et, hors périmètre de ce spec, le LLM pourra
enrober cette demande de façon naturelle).

## Corpus

Fichier `corpus.jsonl`, une ligne JSON par cas :

```json
{"message": "Capture cette page https://fr.wikipedia.org/wiki/Henri_IV", "agent": "wikilm"}
{"message": "Quels sont mes rendez-vous demain ?", "agent": "gog"}
{"message": "Rédige-moi une note de synthèse sur la sobriété énergétique", "agent": "superpowers"}
{"message": "Envoie un mail à Paul pour annuler la réunion", "agent": "gog"}
{"message": "Qu'est-ce que tu en penses ?", "agent": "clarify"}
```

Catégories couvertes (chaque cas étiqueté avec l'agent attendu) :
- **Intention claire** par agent (le gros du corpus)
- **Ambigu / hors-sujet** → `clarify`
- **Multi-intention** (« cherche X dans le wiki et envoie-le par mail ») : étiqueté avec
  l'agent **prioritaire** par convention (le premier acte à poser), et listé à part comme cas
  à surveiller — le multi-intention n'est pas résolu en Spec 1, seulement mesuré.

Volume initial visé : ~60-100 cas, dont ~15-25 par agent réel et ~15 cas `clarify`/limites.

La **génération du corpus est un composant de première classe** (`corpus_gen.py`, ci-dessous),
pas un geste manuel ponctuel : chaque nouvel agent introduit plus tard exige sa tranche de
corpus, et l'outil doit accompagner l'utilisateur de façon répétable. Le corpus étant aussi le
futur dataset LoRA, sa qualité compte.

## Composants

Tous dans `Wiki_LM/routing/` (réutilise le venv `Wiki_LM/.venv` et BGE-M3 déjà présents).

### `agents.json`
Le catalogue ci-dessus. Source unique de vérité pour les deux routeurs.

### `corpus.jsonl`
Le corpus étiqueté.

### `corpus_gen.py`
Génération assistée du corpus, **itérative et few-shot**, réutilisable à chaque nouvel agent.
Utilise un LLM *capable* en cloud (DeepSeek ou Euria) — c'est de l'outillage hors-ligne, pas le
runtime confidentiel, donc le cloud est acceptable ici.

Workflow pour un agent (`--agent <nom>`) :
1. **Amorçage.** Si `corpus.jsonl` contient déjà ≥ 5 exemples validés pour cet agent, ils
   servent d'**ancres few-shot** dans le prompt de génération (et les rejets connus, s'il y en
   a, de négatifs « évite ce genre de cas »). Sinon, génération zéro-shot à partir de la seule
   description de l'agent dans `agents.json`.
2. **Génération.** Le LLM produit N messages candidats. Le prompt exige explicitement :
   diversité de registre, de longueur et de formulation ; cas avec et sans arguments ; et
   quelques **cas-frontière** proches d'autres agents du catalogue (pour durcir le routeur).
3. **Revue.** Les candidats sont écrits dans `candidates_<agent>.jsonl` avec le label proposé.
   L'utilisateur édite, supprime, corrige, ajoute des cas à la main.
4. **Validation.** `corpus_gen.py --commit --agent <nom>` valide le format des candidats revus
   et les ajoute à `corpus.jsonl`.

Boucle d'amélioration : si le tour 1 déçoit, on garde 5-10 cas corrigés et on relance — le tour
suivant s'ancre dessus et se recadre. Tout exemple validé, qu'il vienne de la génération ou
plus tard de la récolte d'usage réel, enrichit le pool d'ancrage des générations futures.

**Caveat assumé.** Un corpus généré par LLM mesure d'abord l'accord routeur↔générateur, pas le
comportement réel des utilisateurs. La curation humaine atténue ce biais ; la récolte d'usage
réel (suite possible) reste la vérité terrain. Ne pas surinterpréter des scores élevés obtenus
sur un corpus purement synthétique.

### `router_base.py`
Interface commune : une classe `Router` avec une méthode
`route(message: str) -> RouteResult`, où `RouteResult = {agent: str, confidence: float}`.
Les deux routeurs en héritent pour être interchangeables dans le runner.

### `router_embed.py`
Routeur par embeddings BGE-M3.
- Construction des prototypes : pour chaque agent, moyenne L2-normalisée des embeddings des
  messages du corpus étiquetés pour cet agent (les prototypes sont donc dérivés du corpus
  d'entraînement, évalués sur un split de test — voir Validation).
- `route()` : embedde le message, cosinus avec chaque prototype, retourne l'agent au cosinus
  max et le score comme `confidence`. Si `confidence < seuil` → agent `clarify`.
- Le seuil est un paramètre (défaut à fixer empiriquement, ex. 0,55).

### `router_llm.py`
Routeur par LLM local.
- Construit le prompt : description du rôle + catalogue d'agents + instruction de répondre
  **uniquement** par `{"agent": "<nom>"}`.
- POST sur `http://127.0.0.1:8998/v1/chat/completions` (llama.cpp, Phi-4-mini), `temperature` basse.
- Parse le JSON de sortie ; en cas de sortie non parsable ou d'agent inconnu → `clarify`.
- `confidence` non disponible nativement → 1.0 si parse OK, 0.0 sinon (champ présent pour
  l'interface commune).

### `eval_routing.py`
Runner d'évaluation.
- Charge `agents.json` et `corpus.jsonl`.
- **Split train/test unique** (ex. 70/30 stratifié par agent, graine fixe pour la
  reproductibilité). Les **deux** routeurs sont évalués sur le **même test set**, pour une
  comparaison équitable :
  - `router_embed` construit ses prototypes sur le **train**, prédit sur le **test**.
  - `router_llm` est zero-shot (ignore le train), prédit sur le **même test**.
- Passe le routeur choisi (argument CLI `--router embed|llm`) sur le test set.
- Produit : exactitude globale, exactitude par agent, **matrice de confusion**, et la **liste
  des messages mal routés** (message, attendu, prédit, confidence).

## Flux de données

```
agents.json ─┬─> corpus_gen.py ──> candidates_<agent>.jsonl ──[revue humaine]──┐
             │        ↑ (few-shot : exemples déjà validés de l'agent)          │
             │        └──────────────────────────────────────────────┐        │
             │                                                  corpus.jsonl <──┘ (--commit)
             │                                                        │
             ├─> router_embed | router_llm ──> prédiction par message │
             │              ↑                          │               │
corpus(test) ┴──[split]─────┴──────────> comparaison ──┴─> exactitude + matrice + erreurs
```

## Validation et critère de succès

- **Test unitaire de l'interface** : un routeur factice déterministe (mappe par mot-clé)
  vérifie que `eval_routing.py` calcule correctement exactitude et matrice sur un mini-corpus
  connu. Cela teste le harnais indépendamment des modèles.
- **Métrique cible** (à confirmer par l'utilisateur au vu des premiers chiffres), mesurée sur
  le test set commun : exactitude ≥ 90 % sur les cas d'intention claire, et rappel correct de
  la classe `clarify` sur les cas ambigus/hors-sujet (ne pas router à tort un cas flou vers un
  agent réel).
- **Décision** : si `router_embed` atteint la cible, c'est le routeur retenu (léger, CPU,
  quasi instantané). Sinon on évalue `router_llm` ; s'il faut davantage, le corpus sert de
  base à un LoRA Phi-4 (spec ultérieur).

## Hors périmètre (explicitement pas dans ce spec)

- Aucun câblage OpenClaw, aucun `sessions_spawn`, aucun Telegram.
- Aucun agent spécialiste réel : les agents ne sont que des noms dans le catalogue.
- Aucune extraction d'arguments (la patate chaude est passée intacte).
- Aucun entraînement LoRA (le corpus est *préparé* pour, pas entraîné ici).
- Aucune gestion du multi-intention (mesurée, pas résolue).
- Le rebuild ROCm de llama.cpp (problème séparé, n'affecte `router_llm` que pour la vitesse,
  pas la justesse — la justesse se mesure quelle que soit la vitesse).

## Suites possibles (specs ultérieurs, pour mémoire, non engagés)

- Spec 2 : câblage OpenClaw du routeur retenu + un premier spécialiste réel (wikilm).
- Spec 3+ : gog, superpowers. Chaque nouvel agent réutilise `corpus_gen.py` pour sa tranche.
- Récolte d'usage réel (approche C) : alimenter `corpus.jsonl` depuis les vrais messages
  Telegram étiquetés par la décision du routeur + correction humaine — vérité terrain pour
  affiner routeur et générations futures.
- LoRA Phi-4 routage si le zero-shot/embeddings ne suffit pas (le corpus est le dataset).
- Rebuild ROCm de llama.cpp pour l'iGPU gfx1035 (Radeon 680M).
