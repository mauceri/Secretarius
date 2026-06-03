# Pipeline d'expérience routage — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un pipeline autonome qui génère un corpus de routage synthétique (DeepSeek génère, Mistral critique), puis mesure comment l'exactitude de routage évolue avec la taille du corpus pour deux mécanismes CPU (prototype-cosinus, tête de classification BGE-M3 gelé), en suivant le coût réel en tokens, et produit un rapport.

**Architecture:** Modules Python plats dans `Wiki_LM/routing/`, réutilisant le harnais existant (`router_base`, `router_embed`, `eval_routing`, `corpus_gen`). Toutes les dépendances modèle (génération, critique, encodeur) sont injectées : la logique est testable hors-ligne, sans réseau ni GPU. Seul `llm_clients.py` touche le réseau, et seul le test de fumée final l'exerce.

**Tech Stack:** Python 3.11, scikit-learn 1.8.0 (LogisticRegression), sentence_transformers (BGE-M3), numpy, openai (DeepSeek + Euria/Mistral), pytest.

---

## Structure des fichiers

| Fichier | Responsabilité |
|---------|----------------|
| `Wiki_LM/routing/cost.py` | `CostTracker` : tokens + coût par modèle |
| `Wiki_LM/routing/critique.py` | Critique Mistral : garde/rejette un candidat |
| `Wiki_LM/routing/router_clf.py` | `ClfRouter` : tête de classification sur BGE-M3 gelé |
| `Wiki_LM/routing/llm_clients.py` | Clients DeepSeek + Mistral, renvoient `(text, usage)` |
| `Wiki_LM/routing/experiment.py` | Orchestrateur : pool, courbe d'apprentissage, rapport |
| `Wiki_LM/routing/tests/test_*.py` | Tests unitaires par composant |

Convention : modules plats (`from router_base import Router`). Tests depuis `Wiki_LM/routing/`.
`PY = /home/mauceric/Secretarius/Wiki_LM/.venv/bin/python`.

Rappel des interfaces existantes (Spec 1) réutilisées :
- `router_base.Router` (ABC, `route(message)->RouteResult`), `RouteResult(agent, confidence)`, `load_agents`, `load_corpus`
- `router_embed.EmbedRouter.from_corpus(train, threshold=, encode_fn=, exclude=("clarify",))`, `router_embed._default_encode`
- `eval_routing.evaluate(router, test_set)->EvalReport(accuracy, per_agent, confusion, misroutes)`, `eval_routing.stratified_split(corpus, test_frac, seed)`
- `corpus_gen.build_generation_prompt(agent, all_agents, examples, negatives, n)`, `corpus_gen.parse_candidates(text, agent_name)`

---

### Task 1 : Suivi du coût (CostTracker)

**Files:**
- Create: `Wiki_LM/routing/cost.py`
- Test: `Wiki_LM/routing/tests/test_cost.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_cost.py` :

```python
from cost import CostTracker


def test_add_accumulates_tokens():
    c = CostTracker(prices={"m": {"input": 0.0, "output": 0.0}})
    c.add("m", {"prompt_tokens": 100, "completion_tokens": 20})
    c.add("m", {"prompt_tokens": 50, "completion_tokens": 5})
    assert c.tokens("m") == (150, 25)


def test_cost_uses_price_table():
    c = CostTracker(prices={"m": {"input": 1.0, "output": 2.0}})
    c.add("m", {"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
    # 1M input * 1.0 + 0.5M output * 2.0 = 1.0 + 1.0
    assert abs(c.cost("m") - 2.0) < 1e-9


def test_unknown_model_zero_cost_but_tokens_tracked():
    c = CostTracker(prices={})
    c.add("inconnu", {"prompt_tokens": 10, "completion_tokens": 3})
    assert c.tokens("inconnu") == (10, 3)
    assert c.cost("inconnu") == 0.0


def test_summary_contains_total():
    c = CostTracker(prices={"m": {"input": 1.0, "output": 1.0}})
    c.add("m", {"prompt_tokens": 1_000_000, "completion_tokens": 0})
    s = c.summary()
    assert "m" in s
    assert "TOTAL" in s
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_cost.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'cost'`

- [ ] **Step 3 : Implémenter cost.py**

Créer `Wiki_LM/routing/cost.py` :

```python
"""Suivi du coût LLM : tokens cumulés par modèle + conversion en coût."""
from __future__ import annotations

from collections import defaultdict

# Prix par défaut, en $/million de tokens. Mistral/Euria à 0 : à mesurer
# (les tokens bruts sont toujours rapportés, c'est la donnée recherchée).
PRICES = {
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    "mistralai/Mistral-Small-4-119B-2603": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    def __init__(self, prices: dict | None = None):
        self.prices = dict(PRICES) if prices is None else dict(prices)
        self._in: dict = defaultdict(int)
        self._out: dict = defaultdict(int)

    def add(self, model: str, usage: dict) -> None:
        self._in[model] += usage.get("prompt_tokens", 0)
        self._out[model] += usage.get("completion_tokens", 0)

    def tokens(self, model: str) -> tuple[int, int]:
        return (self._in[model], self._out[model])

    def cost(self, model: str) -> float:
        p = self.prices.get(model, {"input": 0.0, "output": 0.0})
        return self._in[model] / 1e6 * p["input"] + self._out[model] / 1e6 * p["output"]

    def summary(self) -> str:
        lines: list[str] = []
        total = 0.0
        for model in sorted(set(self._in) | set(self._out)):
            i, o = self._in[model], self._out[model]
            c = self.cost(model)
            total += c
            lines.append(f"  {model}: {i} in + {o} out tokens → {c:.4f} $")
        lines.append(f"  TOTAL: {total:.4f} $")
        return "\n".join(lines)
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_cost.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/cost.py Wiki_LM/routing/tests/test_cost.py
git commit -m "feat(routing): suivi du coût LLM (CostTracker, tokens + prix par modèle)"
```

---

### Task 2 : Critique Mistral (garde/rejette)

**Files:**
- Create: `Wiki_LM/routing/critique.py`
- Test: `Wiki_LM/routing/tests/test_critique.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_critique.py` :

```python
from critique import build_critique_prompt, parse_verdict, critique_candidates


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


def test_build_critique_prompt_mentions_target_and_message():
    prompt = build_critique_prompt({"message": "envoie un mail", "agent": "gog"}, _AGENTS)
    assert "gog" in prompt
    assert "envoie un mail" in prompt
    assert "GARDER" in prompt and "REJETER" in prompt


def test_parse_verdict_keep():
    assert parse_verdict("GARDER") is True
    assert parse_verdict("  garder  ") is True
    assert parse_verdict("Verdict : GARDER") is True


def test_parse_verdict_reject():
    assert parse_verdict("REJETER") is False
    assert parse_verdict("je ne sais pas") is False
    assert parse_verdict("GARDER ou REJETER ? REJETER") is False


def test_critique_candidates_filters_and_sums_usage():
    candidates = [
        {"message": "bon", "agent": "gog"},
        {"message": "mauvais", "agent": "gog"},
        {"message": "bon2", "agent": "gog"},
    ]

    def _fake_critique(prompt):
        # garde si le message "bon" apparaît dans le prompt
        verdict = "GARDER" if ("bon" in prompt) else "REJETER"
        return verdict, {"prompt_tokens": 10, "completion_tokens": 1}

    kept, usage = critique_candidates(candidates, _AGENTS, _fake_critique)
    assert [c["message"] for c in kept] == ["bon", "bon2"]
    assert usage == {"prompt_tokens": 30, "completion_tokens": 3}
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_critique.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'critique'`

- [ ] **Step 3 : Implémenter critique.py**

Créer `Wiki_LM/routing/critique.py` :

```python
"""Critique d'un candidat de corpus par un modèle tiers (Mistral/Euria).

Le critique est DISTINCT du générateur (anti-circularité) : il confirme ou rejette
chaque exemple. Pas de ré-étiquetage — un candidat ambigu ou mal classé est rejeté.
"""
from __future__ import annotations


def build_critique_prompt(candidate: dict, agents: list[dict]) -> str:
    target = candidate["agent"]
    desc = next((a["description"] for a in agents if a["name"] == target), "")
    lines = [
        f'Un message a été étiqueté comme relevant de l\'agent "{target}".',
        f'Rôle de "{target}" : {desc}',
        "Autres agents :",
    ]
    for a in agents:
        if a["name"] != target:
            lines.append(f'- {a["name"]} : {a["description"]}')
    lines += [
        f'Message : "{candidate["message"]}"',
        "Ce message relève-t-il clairement et sans ambiguïté de cet agent, et d'aucun autre ?",
        "Réponds par un seul mot : GARDER ou REJETER.",
    ]
    return "\n".join(lines)


def parse_verdict(text: str) -> bool:
    """True (garder) seulement si GARDER présent ET REJETER absent."""
    up = text.upper()
    return "GARDER" in up and "REJETER" not in up


def critique_candidates(candidates: list[dict], agents: list[dict], critique_fn):
    """Applique le critique à chaque candidat. Retourne (gardés, usage_cumulé).

    critique_fn(prompt) -> (text, usage) ; usage = {prompt_tokens, completion_tokens}.
    """
    kept: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    for c in candidates:
        text, usage = critique_fn(build_critique_prompt(c, agents))
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
        if parse_verdict(text):
            kept.append(c)
    return kept, usage_total
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_critique.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/critique.py Wiki_LM/routing/tests/test_critique.py
git commit -m "feat(routing): critique Mistral garde/rejette (anti-circularité)"
```

---

### Task 3 : Routeur tête de classification (BGE-M3 gelé)

**Files:**
- Create: `Wiki_LM/routing/router_clf.py`
- Test: `Wiki_LM/routing/tests/test_router_clf.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_router_clf.py` :

```python
import numpy as np

from router_clf import ClfRouter


def _fake_encode(texts):
    """Encodeur 2-D déterministe : gog→[1,0], wikilm→[0,1], autre→[0.5,0.5]."""
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low:
            vecs.append([1.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0])
        else:
            vecs.append([0.5, 0.5])
    return np.array(vecs, dtype=np.float32)


def _train():
    return [
        {"message": "envoie un mail", "agent": "gog"},
        {"message": "lis mon mail", "agent": "gog"},
        {"message": "mail urgent", "agent": "gog"},
        {"message": "capture cette url", "agent": "wikilm"},
        {"message": "ajoute au wiki", "agent": "wikilm"},
        {"message": "url wiki à garder", "agent": "wikilm"},
        {"message": "bla bla flou", "agent": "clarify"},
    ]


def test_excludes_clarify_from_classes():
    router = ClfRouter.from_corpus(_train(), threshold=0.55, encode_fn=_fake_encode)
    assert set(router.clf.classes_) == {"gog", "wikilm"}


def test_routes_clear_message():
    router = ClfRouter.from_corpus(_train(), threshold=0.55, encode_fn=_fake_encode)
    res = router.route("envoie un nouveau mail")
    assert res.agent == "gog"
    assert res.confidence > 0.55


def test_ambiguous_below_threshold_is_clarify():
    # message → [0.5, 0.5], proba symétrique 0.5/0.5 < 0.55 → clarify
    router = ClfRouter.from_corpus(_train(), threshold=0.55, encode_fn=_fake_encode)
    res = router.route("quelque chose de totalement flou")
    assert res.agent == "clarify"
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_clf.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'router_clf'`

- [ ] **Step 3 : Implémenter router_clf.py**

Créer `Wiki_LM/routing/router_clf.py` :

```python
"""Routeur par tête de classification sur embeddings BGE-M3 gelés.

Le modèle d'embedding reste gelé ; on entraîne seulement une régression logistique
(rapide, CPU) sur (embedding → agent). Même convention que EmbedRouter : clarify
exclu de l'entraînement, atteint via le seuil de confiance.
"""
from __future__ import annotations

import numpy as np

from router_base import Router, RouteResult
from router_embed import _default_encode


class ClfRouter(Router):
    def __init__(self, clf, threshold: float = 0.55, encode_fn=_default_encode):
        self.clf = clf
        self.threshold = threshold
        self.encode_fn = encode_fn

    @classmethod
    def from_corpus(cls, train: list[dict], threshold: float = 0.55,
                    encode_fn=_default_encode, exclude=("clarify",)):
        from sklearn.linear_model import LogisticRegression
        msgs: list[str] = []
        labels: list[str] = []
        for row in train:
            if row["agent"] in exclude:
                continue
            msgs.append(row["message"])
            labels.append(row["agent"])
        X = encode_fn(msgs)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, labels)
        return cls(clf, threshold, encode_fn)

    def route(self, message: str) -> RouteResult:
        vec = self.encode_fn([message])  # forme (1, d)
        proba = self.clf.predict_proba(vec)[0]
        best = int(np.argmax(proba))
        score = float(proba[best])
        if score < self.threshold:
            return RouteResult("clarify", score)
        return RouteResult(str(self.clf.classes_[best]), score)
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_clf.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/router_clf.py Wiki_LM/routing/tests/test_router_clf.py
git commit -m "feat(routing): routeur tête de classification (LogisticRegression sur BGE-M3 gelé)"
```

---

### Task 4 : Clients LLM (DeepSeek + Mistral)

**Files:**
- Create: `Wiki_LM/routing/llm_clients.py`
- Test: `Wiki_LM/routing/tests/test_llm_clients.py`

- [ ] **Step 1 : Écrire le test qui échoue**

Seul `_extract` (transformation d'une réponse OpenAI-like en `(text, usage)`) est testé
hors-ligne ; les appels réseau sont couverts par le test de fumée (Task 6).

Créer `Wiki_LM/routing/tests/test_llm_clients.py` :

```python
from types import SimpleNamespace

from llm_clients import _extract


def test_extract_text_and_usage():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="bonjour"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )
    text, usage = _extract(resp)
    assert text == "bonjour"
    assert usage == {"prompt_tokens": 12, "completion_tokens": 3}
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_llm_clients.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'llm_clients'`

- [ ] **Step 3 : Implémenter llm_clients.py**

Créer `Wiki_LM/routing/llm_clients.py` :

```python
"""Clients LLM minces, OpenAI-compatibles, renvoyant (text, usage).

Seul module qui touche le réseau. usage = {prompt_tokens, completion_tokens}.
"""
from __future__ import annotations

import os


def _extract(resp) -> tuple[str, dict]:
    text = resp.choices[0].message.content
    u = resp.usage
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
    return text, usage


def deepseek_generate(prompt: str, temperature: float = 0.9) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return _extract(resp)


def mistral_critique(prompt: str, temperature: float = 0.0) -> tuple[str, dict]:
    from openai import OpenAI
    pid = os.environ["EURIA_PRODUCT_ID"]
    client = OpenAI(base_url=f"https://api.infomaniak.com/2/ai/{pid}/openai/v1",
                    api_key=os.environ["EURIA_API_KEY"])
    resp = client.chat.completions.create(
        model="mistralai/Mistral-Small-4-119B-2603",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=8,
    )
    return _extract(resp)
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_llm_clients.py -v`
Expected: PASS (1 test)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/llm_clients.py Wiki_LM/routing/tests/test_llm_clients.py
git commit -m "feat(routing): clients DeepSeek + Mistral (renvoient text + usage tokens)"
```

---

### Task 5 : Orchestrateur — logique (sous-échantillonnage, courbe, rapport)

**Files:**
- Create: `Wiki_LM/routing/experiment.py`
- Test: `Wiki_LM/routing/tests/test_experiment.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_experiment.py` :

```python
import numpy as np

from experiment import subsample, clamp_sizes, run_curve, format_experiment_report


def _fake_encode(texts):
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low:
            vecs.append([1.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0])
        else:
            vecs.append([0.5, 0.5])
    return np.array(vecs, dtype=np.float32)


def _pool():
    return (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(5)]
        + [{"message": f"url wiki {i}", "agent": "wikilm"} for i in range(5)]
    )


def test_subsample_is_stratified_and_deterministic():
    sub = subsample(_pool(), n=2, seed=1)
    counts = {}
    for r in sub:
        counts[r["agent"]] = counts.get(r["agent"], 0) + 1
    assert counts == {"gog": 2, "wikilm": 2}
    sub2 = subsample(_pool(), n=2, seed=1)
    assert [r["message"] for r in sub] == [r["message"] for r in sub2]


def test_clamp_sizes_caps_to_min_available():
    pool = (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(3)]
        + [{"message": f"url {i}", "agent": "wikilm"} for i in range(6)]
    )
    clamped, cap = clamp_sizes([2, 4, 8], pool)
    assert cap == 3
    assert clamped == [2, 3]  # 4 et 8 plafonnés à 3, dédupliqués, triés


def test_run_curve_one_entry_per_size_and_mechanism():
    pool = (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(4)]
        + [{"message": f"url wiki {i}", "agent": "wikilm"} for i in range(4)]
    )
    test_set = [
        {"message": "mail test", "agent": "gog"},
        {"message": "url wiki test", "agent": "wikilm"},
        {"message": "totalement flou", "agent": "clarify"},
    ]
    results = run_curve(pool, test_set, sizes=[2, 3], threshold=0.55,
                        encode_fn=_fake_encode, seed=1)
    # 2 tailles × 2 mécanismes = 4 entrées
    assert len(results) == 4
    mechs = {r["mechanism"] for r in results}
    assert mechs == {"prototype", "clf"}
    for r in results:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert "clarify_recall" in r
        assert "per_agent" in r


def test_format_report_has_caveat_table_and_cost():
    results = [
        {"size": 3, "mechanism": "prototype", "accuracy": 0.8, "clarify_recall": 1.0},
        {"size": 3, "mechanism": "clf", "accuracy": 0.95, "clarify_recall": 1.0},
    ]
    report = format_experiment_report(results, "  modele: 10 in + 2 out tokens → 0.0001 $",
                                      min_accuracy=0.9, agent_names=["gog", "wikilm"], cap=3)
    assert "PLAFOND OPTIMISTE" in report or "plafond" in report.lower()
    assert "prototype" in report and "clf" in report
    assert "Coût" in report or "coût" in report.lower()
    assert "0.0001" in report
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_experiment.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'experiment'`

- [ ] **Step 3 : Implémenter experiment.py (logique seule pour cette tâche)**

Créer `Wiki_LM/routing/experiment.py` :

```python
"""Pipeline d'expérience : génération+critique du corpus, courbe d'apprentissage, rapport.

Cette première moitié contient la logique pure (testable hors-ligne). La génération
réelle (build_pool) et le câblage CLI (main) sont ajoutés ensuite.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict


def subsample(train_pool: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sous-échantillonne n exemples par agent (stratifié, déterministe)."""
    by_agent: dict = defaultdict(list)
    for row in train_pool:
        by_agent[row["agent"]].append(row)
    rng = random.Random(seed)
    out: list[dict] = []
    for rows in by_agent.values():
        rows = rows[:]
        rng.shuffle(rows)
        out.extend(rows[:n])
    return out


def clamp_sizes(sizes: list[int], train_pool: list[dict]):
    """Plafonne chaque taille au minimum d'exemples disponibles parmi les agents.

    Retourne (tailles_plafonnées_triées_uniques, cap).
    """
    counts = Counter(r["agent"] for r in train_pool)
    cap = min(counts.values()) if counts else 0
    clamped = sorted({min(s, cap) for s in sizes if s > 0})
    return clamped, cap


def run_curve(train_pool: list[dict], test_set: list[dict], sizes: list[int],
              threshold: float, encode_fn, seed: int = 42) -> list[dict]:
    """Pour chaque (taille, mécanisme) : construit sur un sous-échantillon, évalue sur test_set."""
    from router_embed import EmbedRouter
    from router_clf import ClfRouter
    from eval_routing import evaluate

    mechanisms = {"prototype": EmbedRouter, "clf": ClfRouter}
    results: list[dict] = []
    for n in sizes:
        sub = subsample(train_pool, n, seed)
        for name, cls in mechanisms.items():
            router = cls.from_corpus(sub, threshold=threshold, encode_fn=encode_fn)
            report = evaluate(router, test_set)
            results.append({
                "size": n,
                "mechanism": name,
                "accuracy": report.accuracy,
                "per_agent": dict(report.per_agent),
                "clarify_recall": report.per_agent.get("clarify", 0.0),
            })
    return results


def format_experiment_report(results: list[dict], cost_summary: str,
                             min_accuracy: float, agent_names: list[str], cap: int) -> str:
    lines = [
        "# Rapport d'expérience — routage par intention",
        "",
        "⚠ Corpus SYNTHÉTIQUE (généré par DeepSeek, critiqué par Mistral).",
        "Les exactitudes sont un PLAFOND OPTIMISTE, pas la précision réelle sur de vrais",
        "utilisateurs. Ce qui transfère : la comparaison relative et la forme de la courbe.",
        "",
        f"Taille max disponible par agent (après rejets du critique) : {cap}",
        f"Seuil d'acceptabilité : {min_accuracy:.0%}",
        "",
        "| Taille/agent | Mécanisme | Exactitude | Rappel clarify | ≥ seuil |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        ok = "✓" if r["accuracy"] >= min_accuracy else ""
        lines.append(
            f"| {r['size']} | {r['mechanism']} | {r['accuracy']:.1%} "
            f"| {r['clarify_recall']:.1%} | {ok} |"
        )
    lines += ["", "## Coût réel mesuré", cost_summary]
    return "\n".join(lines)
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_experiment.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/experiment.py Wiki_LM/routing/tests/test_experiment.py
git commit -m "feat(routing): orchestrateur — sous-échantillonnage, courbe, rapport"
```

---

### Task 6 : Orchestrateur — génération du pool, CLI, test de fumée réel

**Files:**
- Modify: `Wiki_LM/routing/experiment.py` (ajouter `build_pool` et `main`)
- Test: `Wiki_LM/routing/tests/test_experiment_pool.py`
- Test: exécution réelle d'une mini-expérience (DeepSeek + Mistral + BGE-M3)

- [ ] **Step 1 : Écrire le test de build_pool qui échoue**

Créer `Wiki_LM/routing/tests/test_experiment_pool.py` :

```python
from cost import CostTracker
from experiment import build_pool


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


def _fake_generate(prompt):
    # Détecte l'agent cible via la phrase distinctive 'vers l'agent "X"'
    # (un simple 'gog in prompt' échouerait : gog figure aussi comme repoussoir
    # dans les prompts des autres agents).
    target = "clarify"
    for agent in ("gog", "wikilm", "clarify"):
        if f'vers l\'agent "{agent}"' in prompt:
            target = agent
            break
    text = (
        f'{{"message": "exemple A pour {target}", "agent": "{target}"}}\n'
        f'{{"message": "exemple B pour {target}", "agent": "{target}"}}\n'
    )
    return text, {"prompt_tokens": 50, "completion_tokens": 20}


def _fake_critique(prompt):
    # Garde A, rejette B
    verdict = "GARDER" if "exemple A" in prompt else "REJETER"
    return verdict, {"prompt_tokens": 10, "completion_tokens": 1}


def test_build_pool_generates_critiques_and_tracks_cost():
    cost = CostTracker(prices={})
    pool, clarify_pool = build_pool(
        _AGENTS, max_per_agent=2, clarify_k=2,
        generate_fn=_fake_generate, critique_fn=_fake_critique, cost=cost,
    )
    # 2 agents réels × 1 gardé (A) chacun = 2 ; clarify séparé = 1
    assert len(pool) == 2
    assert {r["agent"] for r in pool} == {"gog", "wikilm"}
    assert all("exemple A" in r["message"] for r in pool)
    assert len(clarify_pool) == 1
    assert clarify_pool[0]["agent"] == "clarify"
    # Coût suivi pour les deux modèles
    assert cost.tokens("deepseek-chat")[0] > 0
    assert cost.tokens("mistralai/Mistral-Small-4-119B-2603")[0] > 0
```

- [ ] **Step 2 : Lancer le test, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_experiment_pool.py -v`
Expected: FAIL avec `ImportError: cannot import name 'build_pool'`

- [ ] **Step 3 : Ajouter build_pool et main à experiment.py**

Ajouter à la fin de `Wiki_LM/routing/experiment.py` (après `format_experiment_report`) :

```python
_DEEPSEEK_MODEL = "deepseek-chat"
_MISTRAL_MODEL = "mistralai/Mistral-Small-4-119B-2603"


def build_pool(agents, max_per_agent, clarify_k, generate_fn, critique_fn, cost):
    """Génère (DeepSeek) puis critique (Mistral) le pool. Retourne (pool_réel, pool_clarify).

    pool_réel : exemples approuvés des agents réels. pool_clarify : exemples clarify approuvés
    (destinés au test uniquement). cost est un CostTracker mis à jour au fil des appels.
    """
    from corpus_gen import build_generation_prompt, parse_candidates
    from critique import critique_candidates

    def _gen_and_critique(agent, n):
        prompt = build_generation_prompt(agent, agents, examples=[], negatives=[], n=n)
        text, usage = generate_fn(prompt)
        cost.add(_DEEPSEEK_MODEL, usage)
        cands = parse_candidates(text, agent["name"])
        kept, cusage = critique_candidates(cands, agents, critique_fn)
        cost.add(_MISTRAL_MODEL, cusage)
        return kept

    pool: list[dict] = []
    for agent in agents:
        if agent["name"] == "clarify":
            continue
        pool.extend(_gen_and_critique(agent, max_per_agent))

    clarify_pool: list[dict] = []
    clarify_agent = next((a for a in agents if a["name"] == "clarify"), None)
    if clarify_agent is not None and clarify_k > 0:
        clarify_pool.extend(_gen_and_critique(clarify_agent, clarify_k))

    return pool, clarify_pool


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from router_base import load_agents
    from eval_routing import stratified_split
    from router_embed import _default_encode
    from cost import CostTracker
    from llm_clients import deepseek_generate, mistral_critique

    parser = argparse.ArgumentParser(description="Expérience de routage (génère, critique, courbe)")
    parser.add_argument("--agents", default="agents.json")
    parser.add_argument("--max-per-agent", type=int, default=20)
    parser.add_argument("--clarify", type=int, default=15)
    parser.add_argument("--sizes", default="3,6,9,12")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="experiment_report.md")
    args = parser.parse_args()

    agents = load_agents(args.agents)
    cost = CostTracker()

    print("[experiment] Génération + critique du pool (DeepSeek → Mistral)...")
    pool, clarify_pool = build_pool(agents, args.max_per_agent, args.clarify,
                                    deepseek_generate, mistral_critique, cost)
    Path("experiment_pool.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool + clarify_pool),
        encoding="utf-8",
    )

    train_pool, test_real = stratified_split(pool, args.test_frac, args.seed)
    test_set = test_real + clarify_pool

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    clamped, cap = clamp_sizes(sizes, train_pool)

    print(f"[experiment] Courbe sur tailles {clamped} (cap={cap}), test={len(test_set)} cas...")
    results = run_curve(train_pool, test_set, clamped, args.threshold, _default_encode, args.seed)

    report = format_experiment_report(
        results, cost.summary(), args.min_accuracy, [a["name"] for a in agents], cap
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le test, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_experiment_pool.py -v`
Expected: PASS (1 test)

- [ ] **Step 5 : Vérifier que toute la suite passe**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/ -v`
Expected: PASS (toute la suite Spec 1 + Spec 2 : 23 + 4 + 4 + 3 + 1 + 4 + 1 = 40 tests)

- [ ] **Step 6 : Test de fumée — mini-expérience réelle**

Charge les clés depuis gateway.systemd.env, lance une expérience minuscule (coût attendu : quelques centimes). Charge aussi BGE-M3 (1-2 min au premier appel).

Run:
```bash
cd ~/Secretarius/Wiki_LM/routing
set -a; source ~/.openclaw/gateway.systemd.env; set +a
../.venv/bin/python experiment.py --max-per-agent 6 --clarify 4 --sizes "3,6" --test-frac 0.4
```
Expected : affiche le rapport (tableau taille × mécanisme, rappel clarify, synthèse de coût avec des tokens DeepSeek ET Mistral non nuls), écrit `experiment_report.md`, sans exception. Copie la sortie complète dans le rapport final.

Si DeepSeek ou Mistral renvoie une erreur réseau/auth, signale-le en BLOCKED avec la trace (ne pas masquer).

- [ ] **Step 7 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/experiment.py Wiki_LM/routing/tests/test_experiment_pool.py
git commit -m "feat(routing): build_pool + CLI experiment + validation de bout en bout"
```

---

## Notes d'exécution

- **Ordre** : T1 → T2 → T3 → T4 → T5 → T6. T1-T4 sont indépendants entre eux. T5 dépend de T3 (run_curve utilise EmbedRouter + ClfRouter). T6 dépend de T1, T2, T4, T5.
- **Aucun réseau dans les tests unitaires** : génération, critique et encodeur sont injectés via fakes. Seul le test de fumée (T6 Step 6) appelle DeepSeek + Mistral réellement et charge BGE-M3.
- **Le `experiment_pool.jsonl` produit** est l'artefact réutilisable (corpus synthétique critiqué) ; il peut servir de point de départ à une future récolte d'usage réel ou à un affinage.
- **Fichiers générés non versionnés** : `experiment_pool.jsonl`, `experiment_report.md` sont des sorties d'exécution — ne pas les committer (les ajouter à `.gitignore` si souhaité, hors périmètre de ce plan).
