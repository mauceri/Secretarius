# Pipeline d'expérience — évaluation des mécanismes de routage — Design (Spec 2)

**Date :** 2026-06-03
**Branche :** v0.2.0-dev
**Statut :** validé en brainstorming, à relire avant le plan
**Dépend de :** harnais de routage (Spec 1, `Wiki_LM/routing/`, déjà implémenté)

## Contexte et motivation

Le harnais de routage (Spec 1) fournit deux routeurs et un évaluateur, mais le corpus graine
(20 cas) est trop petit pour conclure : l'évaluation embeddings y donne ~37 %. On veut savoir,
empiriquement et à moindre coût, **deux choses** :

1. Le routage par embeddings est-il viable, et **combien de corpus** faut-il pour atteindre un
   seuil d'acceptabilité ? (courbe d'apprentissage)
2. Entre le **prototype-cosinus** actuel et une **tête de classification** sur BGE-M3 gelé,
   lequel est meilleur à corpus égal ? (comparaison de mécanismes)

Le tout sans curation humaine (trop lente pour itérer), mais sans tomber dans le piège du
corpus auto-référentiel : on dissocie **générateur** et **critique** (deux modèles distincts).

## Principes

**Générateur ≠ critique.** DeepSeek **génère** les exemples ; Mistral/Euria les **critique**
(garde/rejette). La vérité-terrain n'est plus l'idée d'un seul modèle mais l'**accord entre
deux modèles** — substitut partiel automatisé à la curation humaine.

**Le corpus reste synthétique.** Les chiffres absolus sont un **plafond optimiste**, pas la
précision réelle sur de vrais utilisateurs. Ce qui transfère, c'est la **comparaison relative**
entre mécanismes et la **forme de la courbe**. Le rapport le dit explicitement.

**Coût mesuré, pas estimé.** Chaque appel LLM renvoie son `usage` (tokens). On cumule par
modèle ; le rapport affiche tokens bruts + coût calculé via une table de prix configurable.

**Génération une fois, sous-échantillonnage ensuite.** Pour une courbe d'apprentissage propre
ET un coût borné : on génère **un pool une seule fois**, on fige un **test-set commun**, et la
courbe fait varier la taille du **train** par sous-échantillonnage. Pas de régénération par
point de courbe.

## Composants

Tous dans `Wiki_LM/routing/` (réutilisent l'existant : `router_base`, `router_embed`,
`eval_routing`, `corpus_gen`). `PY = Wiki_LM/.venv/bin/python`.

### `llm_clients.py`
Deux clients minces, OpenAI-compatibles, renvoyant `(text, usage)` où
`usage = {"prompt_tokens": int, "completion_tokens": int}` :
- `deepseek_generate(prompt) -> (text, usage)` — base_url `https://api.deepseek.com`, modèle
  `deepseek-chat`, clé `DEEPSEEK_API_KEY`.
- `mistral_critique(prompt) -> (text, usage)` — base_url
  `https://api.infomaniak.com/2/ai/${EURIA_PRODUCT_ID}/openai/v1`, modèle
  `mistralai/Mistral-Small-4-119B-2603`, clé `EURIA_API_KEY`.

Ces fonctions sont les seules à toucher le réseau. Tout le reste les reçoit par injection, donc
testable hors-ligne.

### `cost.py`
`CostTracker` : accumule les tokens par nom de modèle, calcule le coût via une table de prix
`{model: {"input": $/Mtok, "output": $/Mtok}}`. Méthodes : `add(model, usage)`,
`tokens(model) -> (in, out)`, `cost(model) -> float`, `summary() -> str`. La table de prix est
fournie par l'opérateur ; prix inconnu → 0, mais les **tokens bruts sont toujours rapportés**
(c'est la donnée que l'utilisateur veut mesurer pour Mistral/Infomaniak).

### `critique.py`
Le critique Mistral. Pour un candidat `{message, agent}` (étiquette proposée) :
- `build_critique_prompt(candidate, agents)` : demande « Ce message relève-t-il clairement de
  l'agent <agent> (description…) et d'aucun autre ? Réponds par GARDER ou REJETER. »
- `parse_verdict(text) -> bool` : True si GARDER, False sinon (rejet par défaut si ambigu).
- `critique_candidates(candidates, agents, critique_fn) -> (kept, usage_total)` : applique le
  critique à chaque candidat, retourne ceux gardés + l'usage cumulé. `critique_fn` injectable.

Pas de ré-étiquetage : un candidat jugé ambigu ou mal classé est **rejeté**, pas déplacé (pour
empêcher le critique de devenir un second générateur).

### `router_clf.py`
`ClfRouter(Router)` — tête de classification sur BGE-M3 **gelé**.
- `from_corpus(train, encode_fn, threshold=0.55, exclude=("clarify",))` : encode les messages
  d'entraînement (hors clarify), entraîne une `sklearn.linear_model.LogisticRegression` sur
  (embeddings → agent). Stocke le classifieur et la liste des classes.
- `route(message)` : encode, `predict_proba` ; si proba max < seuil → `clarify`, sinon la classe
  argmax. `RouteResult(agent, confidence=proba_max)`.
- Même convention que `EmbedRouter` : clarify exclu de l'entraînement, atteint par le seuil.
  `encode_fn` injectable (tests sans GPU).

### `experiment.py`
Orchestrateur du protocole. CLI :
`--max-per-agent M --clarify K --sizes "3,6,9,12" --threshold 0.55 --min-accuracy 0.9 --seed 42`.

Étapes :
1. **Génération du pool (une fois).** Pour chacun des 3 agents réels : `corpus_gen` (DeepSeek)
   produit M candidats → `critique` (Mistral) garde les approuvés. Génère aussi K exemples
   `clarify` (ambigus/hors-sujet), critiqués pour confirmer l'ambiguïté. Le pool et l'usage
   tokens sont sauvegardés (`experiment_pool.jsonl`, `experiment_usage.json`).
2. **Test-set figé.** Split stratifié du pool des agents réels → test fixe + pool d'entraînement.
   Les exemples `clarify` vont **entièrement dans le test** (jamais entraînés par ces deux
   mécanismes). Le test-set est identique pour tous les points de courbe et tous les mécanismes.
3. **Courbe d'apprentissage.** Pour chaque taille n de `--sizes` : sous-échantillonne n
   exemples/agent du pool d'entraînement (stratifié, graine fixe) ; pour chaque mécanisme
   ∈ {`EmbedRouter`, `ClfRouter`} : construit sur le sous-échantillon, évalue sur le **test-set
   figé** via `eval_routing.evaluate`. Enregistre exactitude globale, par agent, et rappel
   `clarify`. Le critique rejetant une part variable des candidats, le pool d'entraînement par
   agent est ≤ M : toute taille n excédant le minimum disponible parmi les agents est **plafonnée
   à ce minimum**, et le rapport signale la taille effective utilisée.
4. **Rapport.** Écrit `experiment_report.md` : tableau (taille × mécanisme → exactitude), la
   ligne du seuil `--min-accuracy` marquée (atteint/non atteint par mécanisme), rappel `clarify`,
   et la **synthèse de coût** (tokens + coût par modèle). Avertissement explicite « corpus
   synthétique, chiffres = plafond optimiste ».

## Flux de données

```
agents.json ─> corpus_gen (DeepSeek) ─> candidats ─> critique (Mistral) ─> pool approuvé
                     │ usage                              │ usage              │
                     └──────────> CostTracker <───────────┘        experiment_pool.jsonl
                                                                            │
                              [split stratifié] ── test-set figé ───────────┤
                                                   train pool ──[sous-éch. n]┤
                                                                            │
        pour n dans sizes × mécanisme ∈ {Embed, Clf} :                      │
            construit(sous-éch.) ─> evaluate(test figé) ─> exactitude ──────┤
                                                                            ▼
                                                      experiment_report.md (+ coût)
```

## Métriques et critère

- **Exactitude** globale et par agent, sur le test figé, à chaque (taille, mécanisme).
- **Rappel clarify** : proportion des cas ambigus correctement renvoyés vers `clarify`
  (ne pas router à tort un cas flou vers un agent réel).
- **Seuil d'acceptabilité** : `--min-accuracy` (défaut 0,9), fourni par l'opérateur. Le rapport
  marque, pour chaque mécanisme, la plus petite taille de corpus qui l'atteint (ou « jamais »).
- **Coût** : tokens entrée/sortie et coût par modèle (DeepSeek, Mistral), cumulés sur toute
  l'expérience (la génération domine ; l'éval embeddings/clf est locale et gratuite).

## Validation (tests, TDD)

Dépendances injectées partout → tests hors-ligne, sans réseau ni GPU :
- `cost.py` : accumulation + calcul de coût sur usage connu → valeurs exactes.
- `critique.py` : `parse_verdict` (GARDER/REJETER/ambigu), `critique_candidates` avec
  `critique_fn` factice (garde un sur deux) → filtrage correct + usage cumulé.
- `router_clf.py` : `ClfRouter` avec encodeur factice 2-D déterministe → classification correcte
  des cas nets, seuil → clarify.
- `experiment.py` : logique de sous-échantillonnage (taille respectée, stratifié, déterministe)
  et assemblage du rapport, avec générateur/critique/encodeur factices → pas de réseau.
- Un **test de fumée** final (manuel, hors CI) lance une mini-expérience réelle
  (`--max-per-agent 6 --sizes "3,6"`) : vérifie que DeepSeek + Mistral répondent, que le pipeline
  produit un rapport et une synthèse de coût non nulle. Coût attendu : quelques centimes.

## Hors périmètre

- Aucun affinage de BGE-M3 (adaptateur/LoRA) ni routeur LLM Phi-4 dans la boucle : on ajoute ces
  mécanismes **seulement après** lecture de la courbe (suite possible).
- Aucune curation humaine (remplacée par le critique Mistral).
- Aucune récolte d'usage réel (vérité terrain, suite possible).
- Aucun câblage OpenClaw.
- GPT-5 comme critique : réservé à plus tard (nécessiterait OAuth via OpenClaw ou une clé API
  OpenAI ; voir notes de session). Mistral/Euria suffit pour ce premier protocole.

## Suites possibles (non engagées)

- Ajouter `ClfRouter`-clarify-comme-classe, le routeur LLM Phi-4, puis l'affinage (tête → LoRA
  BGE-M3) comme mécanismes supplémentaires dans la même courbe.
- Critique GPT-5 (via clé API ou OAuth/OpenClaw) si la qualité du critique Mistral plafonne.
- Récolte d'usage réel Telegram comme test-set de vérité terrain, à comparer au test synthétique.
