# Routage par intention — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un harnais autonome (hors OpenClaw) qui évalue le routage par intention d'un message vers le bon agent, en comparant un routeur par embeddings (BGE-M3) et un routeur par LLM local (Phi-4-mini), avec un outil de génération de corpus réutilisable.

**Architecture:** Modules Python plats dans `Wiki_LM/routing/`, réutilisant le venv `Wiki_LM/.venv` et BGE-M3 déjà présents. Tous les composants dépendants d'un modèle (encodeur, LLM, HTTP) reçoivent leur dépendance par injection, ce qui rend la logique testable en TDD sans réseau ni GPU. `eval_routing.py` est agnostique du routeur via une interface commune `Router`.

**Tech Stack:** Python 3.11, sentence_transformers (BGE-M3), numpy, openai (DeepSeek pour la génération), pytest. llama.cpp Phi-4-mini exposé sur `http://127.0.0.1:8998/v1`.

---

## Structure des fichiers

| Fichier | Responsabilité |
|---------|----------------|
| `Wiki_LM/routing/agents.json` | Catalogue déclaratif des agents (nom + description) |
| `Wiki_LM/routing/corpus.jsonl` | Corpus étiqueté `{message, agent}` (graine initiale) |
| `Wiki_LM/routing/router_base.py` | `RouteResult`, ABC `Router`, `load_agents`, `load_corpus` |
| `Wiki_LM/routing/eval_routing.py` | Split stratifié, `evaluate`, rapport (exactitude/matrice/erreurs) |
| `Wiki_LM/routing/router_embed.py` | `EmbedRouter` : prototypes BGE-M3, cosinus + seuil → clarify |
| `Wiki_LM/routing/router_llm.py` | `LlmRouter` : prompt catalogue, POST :8998, parse JSON |
| `Wiki_LM/routing/corpus_gen.py` | Génération assistée few-shot du corpus (LLM cloud) |
| `Wiki_LM/routing/tests/conftest.py` | Ajoute `routing/` au `sys.path` pour les tests |
| `Wiki_LM/routing/tests/test_*.py` | Tests unitaires par composant |
| `Wiki_LM/routing/README.md` | Mode d'emploi du harnais |

Convention d'imports : modules plats (`from router_base import Router`), comme `Wiki_LM/tools/`. Les tests s'exécutent avec `cd Wiki_LM/routing && ../.venv/bin/python -m pytest`.

`PY` ci-dessous désigne `/home/mauceric/Secretarius/Wiki_LM/.venv/bin/python`.

---

### Task 1 : Socle — catalogue, interface commune, helpers

**Files:**
- Create: `Wiki_LM/routing/agents.json`
- Create: `Wiki_LM/routing/router_base.py`
- Create: `Wiki_LM/routing/tests/conftest.py`
- Test: `Wiki_LM/routing/tests/test_router_base.py`

- [ ] **Step 1 : Créer le catalogue d'agents**

Créer `Wiki_LM/routing/agents.json` :

```json
{
  "agents": [
    {"name": "wikilm", "description": "Capture, recherche et ingestion de connaissances : URLs à mémoriser, notes, tags, questions sur la base documentaire."},
    {"name": "gog", "description": "Google Workspace : email (lire, chercher, envoyer), agenda, fichiers Drive."},
    {"name": "superpowers", "description": "Rédaction de textes longs, brainstorming, conception de plans et de spécifications."},
    {"name": "clarify", "description": "Intention floue, ambiguë ou hors-sujet : demander une précision à l'utilisateur."}
  ]
}
```

- [ ] **Step 2 : Créer le conftest des tests**

Créer `Wiki_LM/routing/tests/conftest.py` :

```python
import sys
from pathlib import Path

# Rend les modules plats de routing/ importables depuis les tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_router_base.py` :

```python
import json

from router_base import RouteResult, Router, load_agents, load_corpus


def test_route_result_fields():
    r = RouteResult(agent="gog", confidence=0.9)
    assert r.agent == "gog"
    assert r.confidence == 0.9


class _KeywordRouter(Router):
    def route(self, message):
        agent = "gog" if "mail" in message.lower() else "clarify"
        return RouteResult(agent, 1.0)


def test_router_is_abstract():
    # Instancier l'ABC directement doit échouer
    import pytest
    with pytest.raises(TypeError):
        Router()


def test_concrete_router_routes():
    assert _KeywordRouter().route("Envoie un mail").agent == "gog"
    assert _KeywordRouter().route("Bonjour").agent == "clarify"


def test_load_agents(tmp_path):
    p = tmp_path / "agents.json"
    p.write_text('{"agents":[{"name":"gog","description":"mail"}]}', encoding="utf-8")
    agents = load_agents(p)
    assert agents[0]["name"] == "gog"
    assert agents[0]["description"] == "mail"


def test_load_corpus_skips_blank_lines(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"message":"salut","agent":"gog"}\n\n{"message":"yo","agent":"wikilm"}\n',
        encoding="utf-8",
    )
    rows = load_corpus(p)
    assert len(rows) == 2
    assert rows[1]["agent"] == "wikilm"
```

- [ ] **Step 4 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_base.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'router_base'`

- [ ] **Step 5 : Implémenter router_base.py**

Créer `Wiki_LM/routing/router_base.py` :

```python
"""Interface commune des routeurs d'intention + helpers de chargement."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RouteResult:
    """Décision de routage : agent choisi + confiance dans [0, 1]."""
    agent: str
    confidence: float


class Router(ABC):
    """Tout routeur expose route(message) -> RouteResult."""

    @abstractmethod
    def route(self, message: str) -> RouteResult:
        ...


def load_agents(path: str | Path) -> list[dict]:
    """Lit agents.json, retourne la liste [{name, description}, ...]."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["agents"]


def load_corpus(path: str | Path) -> list[dict]:
    """Lit corpus.jsonl, retourne [{message, agent}, ...] (lignes vides ignorées)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
```

- [ ] **Step 6 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_base.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/agents.json Wiki_LM/routing/router_base.py \
        Wiki_LM/routing/tests/conftest.py Wiki_LM/routing/tests/test_router_base.py
git commit -m "feat(routing): socle — catalogue agents, interface Router, helpers"
```

---

### Task 2 : Évaluation — split stratifié, métriques, rapport

**Files:**
- Create: `Wiki_LM/routing/eval_routing.py`
- Test: `Wiki_LM/routing/tests/test_eval_routing.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_eval_routing.py` :

```python
from router_base import Router, RouteResult
from eval_routing import stratified_split, evaluate


def _corpus():
    return [
        {"message": "m1", "agent": "gog"},
        {"message": "m2", "agent": "gog"},
        {"message": "m3", "agent": "gog"},
        {"message": "m4", "agent": "gog"},
        {"message": "w1", "agent": "wikilm"},
        {"message": "w2", "agent": "wikilm"},
        {"message": "w3", "agent": "wikilm"},
        {"message": "w4", "agent": "wikilm"},
    ]


def test_split_is_stratified_and_deterministic():
    train, test = stratified_split(_corpus(), test_frac=0.5, seed=1)
    # 4 par agent, 50% → 2 test par agent
    test_agents = sorted(r["agent"] for r in test)
    assert test_agents == ["gog", "gog", "wikilm", "wikilm"]
    # déterministe : même graine → même split
    train2, test2 = stratified_split(_corpus(), test_frac=0.5, seed=1)
    assert [r["message"] for r in test] == [r["message"] for r in test2]
    # pas de fuite train/test
    assert set(r["message"] for r in train).isdisjoint(r["message"] for r in test)


class _AlwaysGog(Router):
    def route(self, message):
        return RouteResult("gog", 0.42)


def test_evaluate_metrics():
    test_set = [
        {"message": "a", "agent": "gog"},
        {"message": "b", "agent": "gog"},
        {"message": "c", "agent": "wikilm"},
        {"message": "d", "agent": "wikilm"},
    ]
    report = evaluate(_AlwaysGog(), test_set)
    # 2/4 corrects
    assert report.accuracy == 0.5
    assert report.per_agent["gog"] == 1.0
    assert report.per_agent["wikilm"] == 0.0
    # matrice : (gog,gog)=2, (wikilm,gog)=2
    assert report.confusion[("gog", "gog")] == 2
    assert report.confusion[("wikilm", "gog")] == 2
    # erreurs : les 2 wikilm mal routés, avec confidence
    assert len(report.misroutes) == 2
    assert report.misroutes[0]["expected"] == "wikilm"
    assert report.misroutes[0]["predicted"] == "gog"
    assert report.misroutes[0]["confidence"] == 0.42


def test_evaluate_empty_test_set():
    report = evaluate(_AlwaysGog(), [])
    assert report.accuracy == 0.0
    assert report.misroutes == []
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_eval_routing.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'eval_routing'`

- [ ] **Step 3 : Implémenter eval_routing.py**

Créer `Wiki_LM/routing/eval_routing.py` :

```python
"""Évaluation d'un routeur sur le corpus : split stratifié, métriques, rapport."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass

from router_base import Router, load_agents, load_corpus


def stratified_split(corpus: list[dict], test_frac: float = 0.3, seed: int = 42):
    """Découpe (train, test) en gardant la proportion par agent. Graine fixe = reproductible."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in corpus:
        by_agent[row["agent"]].append(row)
    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for rows in by_agent.values():
        rows = rows[:]
        rng.shuffle(rows)
        # Au moins 1 cas de test par agent dès qu'il y a >1 exemple
        n_test = max(1, round(len(rows) * test_frac)) if len(rows) > 1 else 0
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    return train, test


@dataclass
class EvalReport:
    accuracy: float
    per_agent: dict          # agent -> exactitude
    confusion: dict          # (attendu, prédit) -> compte
    misroutes: list          # [{message, expected, predicted, confidence}]


def evaluate(router: Router, test_set: list[dict]) -> EvalReport:
    confusion: dict = defaultdict(int)
    total: dict = defaultdict(int)
    correct_by: dict = defaultdict(int)
    misroutes: list = []
    correct = 0
    for row in test_set:
        expected = row["agent"]
        result = router.route(row["message"])
        predicted = result.agent
        confusion[(expected, predicted)] += 1
        total[expected] += 1
        if predicted == expected:
            correct += 1
            correct_by[expected] += 1
        else:
            misroutes.append({
                "message": row["message"],
                "expected": expected,
                "predicted": predicted,
                "confidence": round(result.confidence, 3),
            })
    accuracy = correct / len(test_set) if test_set else 0.0
    per_agent = {a: correct_by[a] / total[a] for a in total}
    return EvalReport(accuracy, per_agent, dict(confusion), misroutes)


def format_report(report: EvalReport, agent_names: list[str]) -> str:
    lines = [f"Exactitude globale : {report.accuracy:.1%}", "", "Par agent :"]
    for a in agent_names:
        if a in report.per_agent:
            lines.append(f"  {a:14s} {report.per_agent[a]:.1%}")
    lines += ["", "Matrice de confusion (attendu → prédit) :"]
    for expected in agent_names:
        for predicted in agent_names:
            count = report.confusion.get((expected, predicted), 0)
            if count:
                lines.append(f"  {expected:12s} → {predicted:12s} : {count}")
    if report.misroutes:
        lines += ["", f"Erreurs ({len(report.misroutes)}) :"]
        for m in report.misroutes:
            lines.append(
                f"  [{m['expected']} → {m['predicted']} c={m['confidence']}] {m['message']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue un routeur sur le corpus")
    parser.add_argument("--router", choices=["embed", "llm"], required=True)
    parser.add_argument("--agents", default="agents.json")
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    agents = load_agents(args.agents)
    corpus = load_corpus(args.corpus)
    train, test = stratified_split(corpus, args.test_frac, args.seed)

    if args.router == "embed":
        from router_embed import EmbedRouter
        router = EmbedRouter.from_corpus(train, threshold=args.threshold)
    else:
        from router_llm import LlmRouter
        router = LlmRouter(agents)

    report = evaluate(router, test)
    print(format_report(report, [a["name"] for a in agents]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_eval_routing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/eval_routing.py Wiki_LM/routing/tests/test_eval_routing.py
git commit -m "feat(routing): évaluation — split stratifié, métriques, matrice de confusion"
```

---

### Task 3 : Routeur par embeddings (BGE-M3)

**Files:**
- Create: `Wiki_LM/routing/router_embed.py`
- Test: `Wiki_LM/routing/tests/test_router_embed.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

On injecte un encodeur factice 2-D pour rendre le test déterministe et sans GPU.
Convention : `gog → [1,0]`, `wikilm → [0,1]`. Vecteurs normalisés.

Créer `Wiki_LM/routing/tests/test_router_embed.py` :

```python
import numpy as np

from router_embed import EmbedRouter


def _fake_encode(texts):
    """Encodeur 2-D déterministe basé sur des mots-clés."""
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low or "gog" in low:
            vecs.append([1.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0])
        else:
            # ambigu : à 45°, cosinus 0.707 avec chaque prototype
            vecs.append([0.7071, 0.7071])
    return np.array(vecs, dtype=np.float32)


def _train():
    return [
        {"message": "envoie un mail", "agent": "gog"},
        {"message": "cherche mon mail", "agent": "gog"},
        {"message": "capture cette url", "agent": "wikilm"},
        {"message": "ajoute au wiki", "agent": "wikilm"},
        {"message": "n'importe quoi", "agent": "clarify"},
    ]


def test_prototypes_exclude_clarify():
    router = EmbedRouter.from_corpus(_train(), threshold=0.5, encode_fn=_fake_encode)
    assert set(router.prototypes.keys()) == {"gog", "wikilm"}


def test_routes_clear_message():
    router = EmbedRouter.from_corpus(_train(), threshold=0.5, encode_fn=_fake_encode)
    res = router.route("envoie un mail à Paul")
    assert res.agent == "gog"
    assert res.confidence > 0.9


def test_below_threshold_is_clarify():
    # message ambigu (45°, cos 0.707) avec seuil élevé → clarify
    router = EmbedRouter.from_corpus(_train(), threshold=0.8, encode_fn=_fake_encode)
    res = router.route("quelque chose de flou")
    assert res.agent == "clarify"


def test_no_prototypes_is_clarify():
    router = EmbedRouter({}, threshold=0.5, encode_fn=_fake_encode)
    assert router.route("peu importe").agent == "clarify"
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_embed.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'router_embed'`

- [ ] **Step 3 : Implémenter router_embed.py**

Créer `Wiki_LM/routing/router_embed.py` :

```python
"""Routeur par embeddings BGE-M3 : un prototype par agent, cosinus + seuil → clarify."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from router_base import Router, RouteResult

_MODEL_NAME = "BAAI/bge-m3"
_model = None


def _default_encode(texts: list[str]) -> np.ndarray:
    """Encode via BGE-M3 (chargé paresseusement), vecteurs L2-normalisés float32."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model.encode(texts, normalize_embeddings=True).astype(np.float32)


class EmbedRouter(Router):
    """Route vers l'agent au prototype le plus proche (cosinus) ; sous le seuil → clarify."""

    def __init__(self, prototypes: dict, threshold: float = 0.55, encode_fn=_default_encode):
        self.prototypes = prototypes
        self.threshold = threshold
        self.encode_fn = encode_fn
        self._agents = list(prototypes.keys())
        self._matrix = (
            np.vstack([prototypes[a] for a in self._agents])
            if prototypes else np.zeros((0, 0), dtype=np.float32)
        )

    @classmethod
    def from_corpus(cls, train: list[dict], threshold: float = 0.55,
                    encode_fn=_default_encode, exclude=("clarify",)):
        """Construit un prototype par agent (moyenne L2-normalisée), hors agents exclus."""
        msgs_by_agent: dict = defaultdict(list)
        for row in train:
            if row["agent"] in exclude:
                continue
            msgs_by_agent[row["agent"]].append(row["message"])
        prototypes: dict = {}
        for agent, msgs in msgs_by_agent.items():
            vecs = encode_fn(msgs)
            proto = vecs.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            prototypes[agent] = proto.astype(np.float32)
        return cls(prototypes, threshold, encode_fn)

    def route(self, message: str) -> RouteResult:
        if not self._agents:
            return RouteResult("clarify", 0.0)
        vec = self.encode_fn([message])[0]
        sims = self._matrix @ vec  # produit scalaire = cosinus (vecteurs normalisés)
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < self.threshold:
            return RouteResult("clarify", score)
        return RouteResult(self._agents[best], score)
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_embed.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/router_embed.py Wiki_LM/routing/tests/test_router_embed.py
git commit -m "feat(routing): routeur par embeddings BGE-M3 (prototypes + seuil clarify)"
```

---

### Task 4 : Routeur par LLM local (Phi-4-mini)

**Files:**
- Create: `Wiki_LM/routing/router_llm.py`
- Test: `Wiki_LM/routing/tests/test_router_llm.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

On injecte un `post_fn` factice qui renvoie une réponse OpenAI-like, sans réseau.

Créer `Wiki_LM/routing/tests/test_router_llm.py` :

```python
from router_llm import LlmRouter, build_prompt


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
    {"name": "clarify", "description": "intention floue"},
]


def _fake_post_returning(content):
    def _post(url, payload):
        return {"choices": [{"message": {"content": content}}]}
    return _post


def test_build_prompt_lists_agents():
    prompt = build_prompt(_AGENTS)
    assert "gog" in prompt and "wikilm" in prompt and "clarify" in prompt
    assert "JSON" in prompt


def test_parses_clean_json():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('{"agent": "gog"}'))
    res = router.route("envoie un mail")
    assert res.agent == "gog"
    assert res.confidence == 1.0


def test_parses_json_embedded_in_prose():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('Voici: {"agent": "wikilm"} voilà'))
    assert router.route("capture url").agent == "wikilm"


def test_unknown_agent_is_clarify():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('{"agent": "inexistant"}'))
    assert router.route("xxx").agent == "clarify"


def test_garbage_output_is_clarify():
    router = LlmRouter(_AGENTS, post_fn=_fake_post_returning('je ne sais pas'))
    assert router.route("xxx").agent == "clarify"


def test_post_exception_is_clarify():
    def _boom(url, payload):
        raise RuntimeError("connexion refusée")
    router = LlmRouter(_AGENTS, post_fn=_boom)
    res = router.route("xxx")
    assert res.agent == "clarify"
    assert res.confidence == 0.0
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_llm.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'router_llm'`

- [ ] **Step 3 : Implémenter router_llm.py**

Créer `Wiki_LM/routing/router_llm.py` :

```python
"""Routeur par LLM local (llama.cpp Phi-4-mini sur :8998)."""
from __future__ import annotations

import json
import re
import urllib.request

from router_base import Router, RouteResult

_ENDPOINT = "http://127.0.0.1:8998/v1/chat/completions"
_MODEL = "phi-4-mini-instruct"


def _default_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def build_prompt(agents: list[dict]) -> str:
    lines = [
        "Tu es un routeur. Choisis l'agent le plus adapté à la demande de l'utilisateur.",
        "Agents disponibles :",
    ]
    for a in agents:
        lines.append(f'- {a["name"]} : {a["description"]}')
    lines.append('Réponds UNIQUEMENT par un objet JSON {"agent": "<nom>"} sans aucun autre texte.')
    return "\n".join(lines)


def _parse_agent(content: str) -> str | None:
    m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    agent = obj.get("agent")
    return agent if isinstance(agent, str) else None


class LlmRouter(Router):
    def __init__(self, agents: list[dict], endpoint: str = _ENDPOINT,
                 model: str = _MODEL, post_fn=_default_post):
        self.system_prompt = build_prompt(agents)
        self.valid = {a["name"] for a in agents}
        self.endpoint = endpoint
        self.model = model
        self.post_fn = post_fn

    def route(self, message: str) -> RouteResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
        }
        try:
            data = self.post_fn(self.endpoint, payload)
            content = data["choices"][0]["message"]["content"]
            agent = _parse_agent(content)
        except Exception:
            return RouteResult("clarify", 0.0)
        if agent in self.valid:
            return RouteResult(agent, 1.0)
        return RouteResult("clarify", 0.0)
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_router_llm.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/router_llm.py Wiki_LM/routing/tests/test_router_llm.py
git commit -m "feat(routing): routeur par LLM local (Phi-4-mini :8998, parse JSON, repli clarify)"
```

---

### Task 5 : Génération assistée du corpus (few-shot)

**Files:**
- Create: `Wiki_LM/routing/corpus_gen.py`
- Test: `Wiki_LM/routing/tests/test_corpus_gen.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `Wiki_LM/routing/tests/test_corpus_gen.py` :

```python
import json

from corpus_gen import (
    build_generation_prompt,
    parse_candidates,
    existing_examples,
    commit_candidates,
)


_AGENTS = [
    {"name": "gog", "description": "email et agenda"},
    {"name": "wikilm", "description": "base de connaissances"},
]


def test_prompt_zero_shot_when_no_examples():
    prompt = build_generation_prompt(_AGENTS[0], _AGENTS, examples=[], negatives=[], n=5)
    assert "gog" in prompt
    assert "wikilm" in prompt  # les autres agents sont listés comme à éviter
    assert "5" in prompt


def test_prompt_includes_fewshot_examples():
    prompt = build_generation_prompt(
        _AGENTS[0], _AGENTS,
        examples=["envoie un mail à Paul"], negatives=["capture cette url"], n=3,
    )
    assert "envoie un mail à Paul" in prompt
    assert "capture cette url" in prompt


def test_parse_candidates_keeps_valid_skips_garbage():
    text = (
        '{"message": "envoie un mail", "agent": "gog"}\n'
        "blabla pas du json\n"
        '- {"message": "cherche mon agenda", "agent": "gog"}\n'
        '{"message": "", "agent": "gog"}\n'  # message vide ignoré
    )
    cands = parse_candidates(text, "gog")
    assert len(cands) == 2
    assert all(c["agent"] == "gog" for c in cands)
    assert cands[1]["message"] == "cherche mon agenda"


def test_existing_examples_filters_by_agent():
    corpus = [
        {"message": "m1", "agent": "gog"},
        {"message": "w1", "agent": "wikilm"},
        {"message": "m2", "agent": "gog"},
    ]
    assert existing_examples(corpus, "gog") == ["m1", "m2"]


def test_commit_appends_valid_skips_malformed(tmp_path):
    candidates = tmp_path / "candidates_gog.jsonl"
    candidates.write_text(
        '{"message": "bon", "agent": "gog"}\n'
        "pas du json\n"
        '{"message": "aussi bon", "agent": "gog"}\n',
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"message": "déjà là", "agent": "wikilm"}\n', encoding="utf-8")

    added = commit_candidates(candidates, corpus)
    assert added == 2

    rows = [json.loads(l) for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3  # 1 existant + 2 ajoutés
    assert rows[-1]["message"] == "aussi bon"
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_corpus_gen.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'corpus_gen'`

- [ ] **Step 3 : Implémenter corpus_gen.py**

Créer `Wiki_LM/routing/corpus_gen.py` :

```python
"""Génération assistée du corpus de routage, itérative et few-shot, via LLM cloud.

Workflow :
  1) python corpus_gen.py --agent gog --n 20
       → génère candidates_gog.jsonl (few-shot si ≥5 exemples validés existent)
  2) revue humaine du fichier candidates_gog.jsonl (éditer/supprimer/ajouter)
  3) python corpus_gen.py --commit --agent gog
       → valide et ajoute les lignes à corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from router_base import load_agents, load_corpus

_FEWSHOT_MIN = 5


def build_generation_prompt(agent: dict, all_agents: list[dict],
                            examples: list[str], negatives: list[str], n: int) -> str:
    lines = [
        f'Génère {n} messages d\'utilisateur variés, en français, qui doivent être '
        f'routés vers l\'agent "{agent["name"]}".',
        f'Rôle de cet agent : {agent["description"]}',
        "Autres agents (NE génère PAS de messages relevant d\'eux) :",
    ]
    for a in all_agents:
        if a["name"] != agent["name"]:
            lines.append(f'- {a["name"]} : {a["description"]}')
    lines += [
        "Contraintes : registres et longueurs variés ; certains avec arguments "
        "(URL, noms, dates), d\'autres sans ; quelques cas-frontière proches d\'un "
        "autre agent mais qui restent du ressort de celui-ci.",
    ]
    if examples:
        lines.append("Exemples déjà validés pour cet agent (inspire-t\'en, varie) :")
        lines += [f"- {e}" for e in examples]
    if negatives:
        lines.append("Évite ce genre de cas (rejetés) :")
        lines += [f"- {e}" for e in negatives]
    lines.append(
        'Réponds par UNE ligne JSON par message, au format : '
        f'{{"message": "...", "agent": "{agent["name"]}"}}'
    )
    return "\n".join(lines)


def parse_candidates(text: str, agent_name: str) -> list[dict]:
    """Extrait les lignes JSON valides ; force le label = agent_name."""
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if isinstance(msg, str) and msg.strip():
            out.append({"message": msg.strip(), "agent": agent_name})
    return out


def existing_examples(corpus: list[dict], agent_name: str) -> list[str]:
    return [r["message"] for r in corpus if r["agent"] == agent_name]


def commit_candidates(candidates_path: str | Path, corpus_path: str | Path) -> int:
    """Ajoute les candidats bien formés à corpus.jsonl. Retourne le nombre ajouté."""
    valid: list[dict] = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg, agent = obj.get("message"), obj.get("agent")
            if isinstance(msg, str) and isinstance(agent, str) and msg.strip():
                valid.append({"message": msg.strip(), "agent": agent})
    with open(corpus_path, "a", encoding="utf-8") as f:
        for obj in valid:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(valid)


def _default_llm(prompt: str) -> str:
    """Appel DeepSeek (outillage hors-ligne, cloud acceptable ici)."""
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return resp.choices[0].message.content


def main(llm=_default_llm) -> None:
    parser = argparse.ArgumentParser(description="Génération assistée du corpus de routage")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--agents", default="agents.json")
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--commit", action="store_true",
                        help="Valide candidates_<agent>.jsonl et l'ajoute au corpus")
    args = parser.parse_args()

    candidates_path = f"candidates_{args.agent}.jsonl"

    if args.commit:
        added = commit_candidates(candidates_path, args.corpus)
        print(f"[corpus_gen] {added} cas ajoutés à {args.corpus}")
        return

    all_agents = load_agents(args.agents)
    agent = next((a for a in all_agents if a["name"] == args.agent), None)
    if agent is None:
        raise SystemExit(f"Agent inconnu : {args.agent}")

    corpus = load_corpus(args.corpus) if Path(args.corpus).exists() else []
    examples = existing_examples(corpus, args.agent)
    examples = examples if len(examples) >= _FEWSHOT_MIN else []

    prompt = build_generation_prompt(agent, all_agents, examples, negatives=[], n=args.n)
    text = llm(prompt)
    cands = parse_candidates(text, args.agent)
    with open(candidates_path, "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    mode = "few-shot" if examples else "zéro-shot"
    print(f"[corpus_gen] {len(cands)} candidats ({mode}) écrits dans {candidates_path}")
    print("Relisez/éditez ce fichier, puis : "
          f"python corpus_gen.py --commit --agent {args.agent}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/test_corpus_gen.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/corpus_gen.py Wiki_LM/routing/tests/test_corpus_gen.py
git commit -m "feat(routing): génération de corpus few-shot itérative (corpus_gen)"
```

---

### Task 6 : Corpus graine, README, et test de fumée d'intégration

**Files:**
- Create: `Wiki_LM/routing/corpus.jsonl`
- Create: `Wiki_LM/routing/README.md`
- Test: exécution réelle de `eval_routing --router embed` (BGE-M3)

- [ ] **Step 1 : Créer un corpus graine écrit à la main**

Créer `Wiki_LM/routing/corpus.jsonl` (≥5 cas par agent réel pour que le split stratifié 70/30 produise au moins un cas de test par agent ; + cas clarify) :

```jsonl
{"message": "Capture cette page https://fr.wikipedia.org/wiki/Henri_IV", "agent": "wikilm"}
{"message": "Mémorise ce lien pour plus tard : https://exemple.fr/article", "agent": "wikilm"}
{"message": "Qu'est-ce que la base de connaissances dit sur la sobriété énergétique ?", "agent": "wikilm"}
{"message": "Ajoute une note avec le tag #histoire sur la Saint-Barthélemy", "agent": "wikilm"}
{"message": "Recherche dans le wiki les pages sur le protestantisme", "agent": "wikilm"}
{"message": "Quels sont mes rendez-vous demain ?", "agent": "gog"}
{"message": "Envoie un mail à Paul pour annuler la réunion de jeudi", "agent": "gog"}
{"message": "Cherche les mails non lus de cette semaine", "agent": "gog"}
{"message": "Ajoute un événement agenda vendredi 14h dentiste", "agent": "gog"}
{"message": "Télécharge le fichier budget.xlsx depuis mon Drive", "agent": "gog"}
{"message": "Rédige-moi une note de synthèse sur la sobriété énergétique", "agent": "superpowers"}
{"message": "Aide-moi à réfléchir à l'architecture d'un assistant personnel", "agent": "superpowers"}
{"message": "Écris un plan détaillé pour un article sur la confidentialité", "agent": "superpowers"}
{"message": "Brainstorm avec moi sur les noms possibles pour ce projet", "agent": "superpowers"}
{"message": "Rédige une spécification pour un module de cache", "agent": "superpowers"}
{"message": "Qu'est-ce que tu en penses ?", "agent": "clarify"}
{"message": "Aide-moi", "agent": "clarify"}
{"message": "Bonjour, ça va ?", "agent": "clarify"}
{"message": "Fais le nécessaire", "agent": "clarify"}
{"message": "C'est compliqué tout ça", "agent": "clarify"}
```

- [ ] **Step 2 : Vérifier que toute la suite de tests passe**

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python -m pytest tests/ -v`
Expected: PASS (tous les tests des tâches 1-5, soit 23 tests)

- [ ] **Step 3 : Test de fumée — évaluation embeddings réelle (BGE-M3)**

Ceci charge réellement BGE-M3 (premier appel : téléchargement/chargement du modèle, peut prendre 1-2 min).

Run: `cd ~/Secretarius/Wiki_LM/routing && ../.venv/bin/python eval_routing.py --router embed --test-frac 0.4`
Expected: affiche un rapport « Exactitude globale : XX% », la ventilation par agent et la matrice de confusion, sans exception. (La valeur exacte importe peu ici — on valide que la chaîne complète tourne de bout en bout.)

- [ ] **Step 4 : Créer le README**

Créer `Wiki_LM/routing/README.md` :

```markdown
# Harnais de routage par intention

Évalue, hors OpenClaw, la capacité à router un message vers le bon agent
(wikilm, gog, superpowers, clarify). Compare deux routeurs sur le même corpus.

## Prérequis
- venv : `../.venv` (sentence_transformers, numpy, openai, pytest)
- Routeur LLM : service `slm-llama-cpp` (Phi-4-mini) sur http://127.0.0.1:8998
- Génération de corpus : `DEEPSEEK_API_KEY` dans l'environnement

## Évaluer
```bash
cd ~/Secretarius/Wiki_LM/routing
../.venv/bin/python eval_routing.py --router embed     # routeur embeddings BGE-M3
../.venv/bin/python eval_routing.py --router llm       # routeur Phi-4-mini local
```

## Enrichir le corpus (par agent)
```bash
../.venv/bin/python corpus_gen.py --agent gog --n 20   # génère candidates_gog.jsonl
# relire/éditer candidates_gog.jsonl
../.venv/bin/python corpus_gen.py --commit --agent gog # ajoute au corpus.jsonl
```

## Tests
```bash
../.venv/bin/python -m pytest tests/ -v
```

## Fichiers
- `agents.json` — catalogue des agents
- `corpus.jsonl` — corpus étiqueté (= futur dataset LoRA)
- `router_embed.py` / `router_llm.py` — les deux routeurs
- `eval_routing.py` — évaluation (exactitude, matrice, erreurs)
- `corpus_gen.py` — génération assistée few-shot
```

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius
git add Wiki_LM/routing/corpus.jsonl Wiki_LM/routing/README.md
git commit -m "feat(routing): corpus graine, README, validation de bout en bout"
```

---

## Notes d'exécution

- **Ordre des tâches** : 1 → 2 → 3 → 4 → 5 → 6. Les tâches 3, 4, 5 sont indépendantes entre elles (toutes dépendent de 1 ; eval de la tâche 2 ne dépend que de l'interface). La tâche 6 dépend de tout.
- **Pas de réseau requis pour les tests** : tous les tests unitaires injectent leurs dépendances (encodeur, post HTTP, LLM). Seul le test de fumée (Task 6, Step 3) charge BGE-M3 réellement.
- **Le routeur LLM réel** (`eval_routing.py --router llm`) nécessite le service `slm-llama-cpp` démarré ; il n'est pas exercé par les tests automatisés (volontairement, pour ne pas dépendre du GPU/CPU lent). Son évaluation est une étape manuelle post-implémentation.
