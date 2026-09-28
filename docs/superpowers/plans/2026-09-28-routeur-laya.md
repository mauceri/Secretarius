# Remplacement du routeur Tiron par Laya — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer le classifieur génératif phi-4-mini+LoRA (port 8998) et le
garde-fou séparé `GogGate` (BGE-M3, 3 centroïdes) du routeur Tiron par un
modèle Laya fine-tuné (décision calibrée, une seule passe), sans changer le
comportement des commandes explicites ni de la FAQ.

**Architecture:** `router_service/server.py::route_message()` garde sa
structure à trois voies (commande explicite / FAQ / texte libre). Seule la
voie « texte libre » change de moteur : `call_adapter()` (HTTP génératif vers
llama.cpp) est remplacé par `LayaClassifier.classify()` (modèle Laya chargé
en process). `GogGate` disparaît ; sa fonction d'embedding BGE-M3 est
extraite en `embed_bge_m3()` pour que `FaqIndex` continue de fonctionner
sans changement. `args` devient une heuristique pure (texte brut du
message), plus jamais générée.

**Tech Stack:** Python 3, `laya` (Apache 2.0, pip), `transformers`/`torch`
(déjà présents pour BGE-M3), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-routeur-laya-design.md`

## Global Constraints

- `args` n'est jamais généré par un modèle — texte après la commande pour un
  appel explicite, message entier brut pour une commande inférée.
- `GogGate` et sa logique à 3 centroïdes sont supprimés, pas conservés en
  parallèle.
- La FAQ (`router_service/faq.py`) n'est pas modifiée — seule sa source
  d'embedding change de propriétaire (`embed_bge_m3()` au lieu de
  `GogGate._embed`).
- Le seuil de confiance de départ pour les commandes gog est `0.50`
  (reprend `SEUIL_GOG` existant).
- Aucune collecte de données neuve — seul le corpus existant
  (`gen_corpus/corpus_lora_train.jsonl`) est reformaté.
- Le service phi-4-mini/llama.cpp (port 8998) n'est décommissionné qu'après
  validation du remplacement en production — étape explicite, jamais
  silencieuse.
- Santiago (prod) est hors périmètre de ce plan : seul sanroque (dev) est
  couvert. La bascule prod suit une fois sanroque validé.
- Toutes les commandes de ce plan s'exécutent depuis la racine du worktree
  courant (jamais de `cd` vers un autre checkout — le harnais le bloque).

---

## File Structure

- `gen_corpus/__init__.py` (nouveau, vide) — fait de `gen_corpus` un
  package Python régulier, pour que `from gen_corpus.to_laya_format import
  ...` (Task 3) résolve de façon fiable.
- `gen_corpus/to_laya_format.py` (nouveau) — reformate le corpus ChatML
  existant vers le schéma Laya (`state`/`questions`/`answers`), avec split
  train/val.
- `gen_corpus/test_to_laya_format.py` (nouveau) — tests du script de
  reformatage.
- `router_service/router.py` (modifié) — `GogGate` et sa logique de
  centroïdes supprimées ; `embed_bge_m3()` ajoutée en remplacement.
- `router_service/laya_classifier.py` (nouveau) — charge un checkpoint Laya
  et expose `classify(message: str) -> tuple[str | None, float]`.
- `router_service/eval_laya.py` (nouveau) — évalue un `LayaClassifier` sur
  un split de validation, rapporte l'exactitude.
- `router_service/server.py` (modifié) — `call_adapter()` et l'usage de
  `_gate`/`GogGate` remplacés par `LayaClassifier` ; `FaqIndex` reçoit
  `embed_bge_m3` au lieu de `_gate._embed`.
- `router_service/test_router.py` (modifié en deux temps : Task 2 puis
  Task 5) — tests `GogGate` retirés, tests `embed_bge_m3` et
  `LayaClassifier`/nouvelle logique gog ajoutés, tests existants de
  `call_adapter` renommés vers la nouvelle fonction.

---

### Task 1 : Reformater le corpus vers le schéma Laya

**Files:**
- Create: `gen_corpus/__init__.py` (fichier vide)
- Create: `gen_corpus/to_laya_format.py`
- Test: `gen_corpus/test_to_laya_format.py`

**Interfaces:**
- Consumes: `gen_corpus/corpus_lora_train.jsonl` (format ChatML existant,
  chaque ligne `{"messages": [{"role": "system", ...}, {"role": "user",
  "content": <texte>}, {"role": "assistant", "content": <JSON string
  {"command": "/x" ou null, "args": "..."}>}]}`).
- Produces: `COMMAND_CRITERIA: dict[str, str]` (nom de commande → description
  courte, y compris `"aucune"`) et la fonction
  `reformat(input_path: Path, train_path: Path, val_path: Path, val_ratio:
  float = 0.15, seed: int = 0) -> tuple[int, int]` (retourne le nombre
  d'exemples écrits train/val). Format de sortie, une ligne JSON par
  exemple :
  ```json
  {"state": "<message user>", "questions": {"command": {"type": "choice",
  "instructions": "Quelle commande Tiron ce message déclenche-t-il ?",
  "criteria": {"/c": "...", ..., "aucune": "..."}}}, "answers": {"command":
  "/c"}}
  ```
  Ces noms (`COMMAND_CRITERIA`, `reformat`) sont réutilisés tels quels par
  la Task 3 (le classifieur construit son schéma `questions.command` à
  partir du même `COMMAND_CRITERIA`, pour ne jamais diverger entre
  entraînement et inférence).

- [ ] **Step 1: Write the failing test**

```python
# gen_corpus/test_to_laya_format.py
import json
from pathlib import Path

from gen_corpus.to_laya_format import COMMAND_CRITERIA, reformat


def test_command_criteria_covers_all_router_commands():
    from router_service.router import WIKI_CMDS, GOG_CMDS
    known = WIKI_CMDS | GOG_CMDS | {"aucune"}
    assert set(COMMAND_CRITERIA) == known


def test_reformat_converts_one_example_with_command(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "ignoré"},
            {"role": "user", "content": "cherche transformers dans le wiki"},
            {"role": "assistant", "content": json.dumps(
                {"command": "/r", "args": "transformers"})},
        ]
    }) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    n_train, n_val = reformat(src, train_path, val_path, val_ratio=0.0)

    assert n_train == 1
    assert n_val == 0
    row = json.loads(train_path.read_text(encoding="utf-8").strip())
    assert row["state"] == "cherche transformers dans le wiki"
    assert row["questions"]["command"]["type"] == "choice"
    assert row["questions"]["command"]["criteria"] == COMMAND_CRITERIA
    assert row["answers"]["command"] == "/r"


def test_reformat_converts_null_command_to_aucune(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "ignoré"},
            {"role": "user", "content": "il fait beau aujourd'hui"},
            {"role": "assistant", "content": json.dumps(
                {"command": None, "args": ""})},
        ]
    }) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    reformat(src, train_path, val_path, val_ratio=0.0)

    row = json.loads(train_path.read_text(encoding="utf-8").strip())
    assert row["answers"]["command"] == "aucune"


def test_reformat_splits_train_and_val_deterministically(tmp_path):
    src = tmp_path / "corpus.jsonl"
    lines = []
    for i in range(20):
        lines.append(json.dumps({
            "messages": [
                {"role": "system", "content": "ignoré"},
                {"role": "user", "content": f"message {i}"},
                {"role": "assistant", "content": json.dumps(
                    {"command": "/q", "args": f"message {i}"})},
            ]
        }))
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    n_train, n_val = reformat(src, train_path, val_path, val_ratio=0.2, seed=0)

    assert n_train == 16
    assert n_val == 4
    # déterministe : un deuxième run avec la même seed donne la même coupe
    train_path2 = tmp_path / "train2.jsonl"
    val_path2 = tmp_path / "val2.jsonl"
    reformat(src, train_path2, val_path2, val_ratio=0.2, seed=0)
    assert train_path.read_text() == train_path2.read_text()
    assert val_path.read_text() == val_path2.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest gen_corpus/test_to_laya_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gen_corpus.to_laya_format'`

- [ ] **Step 3: Write minimal implementation**

Create `gen_corpus/__init__.py` — empty file (zero bytes).

```python
# gen_corpus/to_laya_format.py
"""Reformate gen_corpus/corpus_lora_train.jsonl (ChatML) vers le schéma Laya
(state/questions/answers) — cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md.
Aucune collecte de données neuve : seule la mise en forme change, et
`args` du corpus d'origine n'est pas repris (le routeur n'en génère plus)."""
from __future__ import annotations

import json
import random
from pathlib import Path

COMMAND_CRITERIA: dict[str, str] = {
    "/c": "capturer une note ou une URL dans le wiki",
    "/q": "poser une question au wiki, réponse synthétisée",
    "/ingest": "lancer l'ingestion des sources en attente",
    "/source": "déléguer une recherche web à Scout",
    "/wikistatus": "connaître l'état de l'ingestion du wiki",
    "/r": "rechercher par mots-clés dans le wiki, sans synthèse",
    "/tags": "lister les tags du wiki",
    "/kbupdate": "mettre à jour la base de connaissances du wiki",
    "/supprimer": "supprimer une page du wiki",
    "/relire": "obtenir la prochaine page du wiki à relire",
    "/verifie": "marquer une page du wiki comme vérifiée",
    "/chercher": "rechercher dans les emails Gmail",
    "/connecter": "démarrer la connexion au compte Google",
    "/inbox": "lister les nouveaux emails",
    "/drive": "rechercher dans Google Drive",
    "/repondre": "répondre à un email",
    "/lire": "lire le contenu d'un email",
    "aucune": "aucune commande ne correspond, message hors sujet",
}

_INSTRUCTIONS = "Quelle commande Tiron ce message déclenche-t-il ?"


def _iter_examples(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = row["messages"]
        user_text = msgs[-2]["content"]
        assistant = json.loads(msgs[-1]["content"])
        command = assistant.get("command") or "aucune"
        yield user_text, command


def _to_laya_row(user_text: str, command: str) -> dict:
    return {
        "state": user_text,
        "questions": {
            "command": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": COMMAND_CRITERIA,
            }
        },
        "answers": {"command": command},
    }


def reformat(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[int, int]:
    rows = [_to_laya_row(text, cmd) for text, cmd in _iter_examples(Path(input_path))]
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = round(len(rows) * val_ratio)
    val_indices = set(indices[:n_val])

    train_lines = []
    val_lines = []
    for i, row in enumerate(rows):
        line = json.dumps(row, ensure_ascii=False)
        if i in val_indices:
            val_lines.append(line)
        else:
            train_lines.append(line)

    Path(train_path).write_text(
        ("\n".join(train_lines) + "\n") if train_lines else "", encoding="utf-8"
    )
    Path(val_path).write_text(
        ("\n".join(val_lines) + "\n") if val_lines else "", encoding="utf-8"
    )
    return len(train_lines), len(val_lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    n_train, n_val = reformat(
        root / "corpus_lora_train.jsonl",
        root / "laya_train.jsonl",
        root / "laya_val.jsonl",
    )
    print(f"train={n_train} val={n_val}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest gen_corpus/test_to_laya_format.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Générer les fichiers réels et vérifier les comptes**

Run: `python3 gen_corpus/to_laya_format.py`
Expected: imprime `train=<N> val=<M>` avec N+M proche de 2509 (taille du
corpus source) ; inspecter `gen_corpus/laya_train.jsonl` (première ligne)
à l'œil pour confirmer le format.

- [ ] **Step 6: Commit**

```bash
git add gen_corpus/__init__.py gen_corpus/to_laya_format.py gen_corpus/test_to_laya_format.py
git commit -m "feat(router): script de reformatage du corpus vers le schéma Laya

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2 : Extraire embed_bge_m3, supprimer GogGate

**Files:**
- Modify: `router_service/router.py`
- Modify: `router_service/test_router.py:1-16` (uniquement les 16
  premières lignes — le reste du fichier, tests de `server.py`, n'est PAS
  touché par cette tâche ; il sera modifié à la Task 5)

**Interfaces:**
- Consumes: rien de nouveau (utilise `transformers.AutoModel`/`AutoTokenizer`
  déjà présents).
- Produces: `embed_bge_m3(texts: list[str]) -> torch.Tensor` (embeddings
  normalisés L2, forme `[len(texts), D]`) — c'est cette fonction que la
  Task 5 branche dans `FaqIndex` à la place de `_gate._embed`.

- [ ] **Step 1: Write the failing test**

Remplacer UNIQUEMENT les 16 premières lignes actuelles de
`router_service/test_router.py` (les tests `GogGate`) par le texte
ci-dessous. Ne touchez à rien d'autre dans ce fichier — tout ce qui suit
la ligne 16 (les tests HTTP/`call_adapter`) reste identique pour l'instant,
la Task 5 s'en chargera :

```python
# router_service/test_router.py (remplace les 16 premières lignes actuelles,
# tout le reste du fichier reste inchangé à cette étape)
from router_service.router import embed_bge_m3, WIKI_CMDS, GOG_CMDS


def test_embed_bge_m3_returns_normalized_vectors():
    import torch
    vecs = embed_bge_m3(["bonjour", "au revoir"])
    assert vecs.shape[0] == 2
    norms = torch.linalg.norm(vecs, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_command_sets_disjoint():
    assert WIKI_CMDS.isdisjoint(GOG_CMDS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest router_service/test_router.py -k embed_bge_m3 -v`
Expected: FAIL — `ImportError: cannot import name 'embed_bge_m3'`

- [ ] **Step 3: Write minimal implementation**

Remplacer tout le contenu de `router_service/router.py` par :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensembles de commandes du routeur Tiron, et embedding BGE-M3 partagé
(utilisé par la FAQ — cf. router_service/faq.py). Le garde-fou de confiance
gog par centroïdes (GogGate) a été retiré le 2026-09-28 : la confiance vient
désormais du score calibré du classifieur Laya (router_service/laya_classifier.py),
cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md."""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WIKI_CMDS = {"/c", "/q", "/ingest", "/source", "/wikistatus", "/r", "/tags", "/kbupdate", "/supprimer", "/relire", "/verifie"}
GOG_CMDS = {"/chercher", "/connecter", "/inbox", "/drive", "/repondre", "/lire"}

_tok = None
_mdl = None


def _load_bge_m3():
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        _mdl = AutoModel.from_pretrained("BAAI/bge-m3").eval()
    return _tok, _mdl


def embed_bge_m3(texts: list[str]) -> torch.Tensor:
    """Embeddings BGE-M3 normalisés L2 (CLS pooling) — utilisé par la FAQ."""
    tok, mdl = _load_bge_m3()
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state[:, 0]
    return F.normalize(out, p=2, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest router_service/test_router.py -k "embed_bge_m3 or command_sets_disjoint" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add router_service/router.py router_service/test_router.py
git commit -m "refactor(router): retire GogGate, extrait embed_bge_m3 partagé

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3 : LayaClassifier

**Files:**
- Create: `router_service/laya_classifier.py`
- Test: `router_service/test_laya_classifier.py`

**Interfaces:**
- Consumes: `COMMAND_CRITERIA` (Task 1, `gen_corpus/to_laya_format.py`) —
  même dictionnaire utilisé pour entraîner et pour interroger, afin que le
  schéma ne diverge jamais entre les deux.
- Produces: classe `LayaClassifier` avec constructeur
  `LayaClassifier(checkpoint: str)` et méthode
  `classify(message: str) -> tuple[str | None, float]` — retourne
  `(None, 0.0)` si le choix gagnant est `"aucune"`, sinon
  `(commande, probabilité)`. C'est cette méthode que la Task 5 appelle en
  remplacement de `call_adapter()`.

**API confirmée** (README `github.com/NandhaKishorM/laya`, vérifié le
2026-09-28 lors de la revue de cette tâche) : l'appel Python direct est
`agent.predict(state, questions)` — pas d'aller-retour HTTP. La réponse a
la forme `result["answers"][nom_question]["choice"]` (label gagnant) et
`result["answers"][nom_question]["confidence"]` (confiance calibrée,
1 − entropie normalisée). Le code ci-dessous utilise cette forme.

- [ ] **Step 1: Write the failing test**

```python
# router_service/test_laya_classifier.py
from router_service.laya_classifier import LayaClassifier


def test_classify_returns_top_command_and_confidence(monkeypatch):
    clf = LayaClassifier.__new__(LayaClassifier)  # évite de charger un vrai modèle
    clf.checkpoint = "test"

    def fake_query(self, message):
        return {"answers": {"command": {"choice": "/q", "confidence": 0.91}}}

    monkeypatch.setattr(LayaClassifier, "_query", fake_query)

    command, confidence = clf.classify("qu'est-ce que le SPLADE ?")

    assert command == "/q"
    assert confidence == 0.91


def test_classify_returns_none_when_top_choice_is_aucune(monkeypatch):
    clf = LayaClassifier.__new__(LayaClassifier)
    clf.checkpoint = "test"

    def fake_query(self, message):
        return {"answers": {"command": {"choice": "aucune", "confidence": 0.80}}}

    monkeypatch.setattr(LayaClassifier, "_query", fake_query)

    command, confidence = clf.classify("il fait beau aujourd'hui")

    assert command is None
    assert confidence == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest router_service/test_laya_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router_service.laya_classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# router_service/laya_classifier.py
"""Classifieur de commandes Tiron basé sur Laya (décision calibrée, une
seule passe) — remplace l'appel génératif à phi-4-mini+LoRA. Cf.
docs/superpowers/specs/2026-09-28-routeur-laya-design.md."""
from __future__ import annotations

from gen_corpus.to_laya_format import COMMAND_CRITERIA

_INSTRUCTIONS = "Quelle commande Tiron ce message déclenche-t-il ?"


class LayaClassifier:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._agent = self._load(checkpoint)

    def _load(self, checkpoint: str):
        import laya
        return laya.load(checkpoint)

    def _query(self, message: str) -> dict:
        questions = {
            "command": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": COMMAND_CRITERIA,
            }
        }
        return self._agent.predict(message, questions)

    def classify(self, message: str) -> tuple[str | None, float]:
        result = self._query(message)
        answer = result["answers"]["command"]
        value = answer["choice"]
        if value == "aucune":
            return None, 0.0
        return value, answer["confidence"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest router_service/test_laya_classifier.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add router_service/laya_classifier.py router_service/test_laya_classifier.py
git commit -m "feat(router): wrapper LayaClassifier (classification calibrée)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4 : Script d'évaluation hors-ligne

**Files:**
- Create: `router_service/eval_laya.py`
- Test: `router_service/test_eval_laya.py`

**Interfaces:**
- Consumes: `LayaClassifier.classify()` (Task 3, mockée dans les tests via
  un objet factice exposant la même méthode).
- Produces: `evaluate(classifier, val_path: Path) -> float` (exactitude,
  0.0-1.0) et un point d'entrée CLI qui l'imprime — c'est ce script que la
  Task 6 (étape manuelle) exécute contre le vrai checkpoint entraîné.

- [ ] **Step 1: Write the failing test**

```python
# router_service/test_eval_laya.py
import json
from pathlib import Path

from router_service.eval_laya import evaluate


class _FakeClassifier:
    def __init__(self, answers):
        self._answers = answers  # dict: message -> (command, prob)

    def classify(self, message):
        return self._answers[message]


def test_evaluate_computes_accuracy(tmp_path):
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("\n".join([
        json.dumps({"state": "a", "answers": {"command": "/q"}}),
        json.dumps({"state": "b", "answers": {"command": "/r"}}),
        json.dumps({"state": "c", "answers": {"command": "aucune"}}),
    ]) + "\n", encoding="utf-8")
    fake = _FakeClassifier({
        "a": ("/q", 0.9),
        "b": ("/q", 0.6),   # faux
        "c": (None, 0.0),
    })

    accuracy = evaluate(fake, val_path)

    assert accuracy == 2 / 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest router_service/test_eval_laya.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router_service.eval_laya'`

- [ ] **Step 3: Write minimal implementation**

```python
# router_service/eval_laya.py
"""Évalue un LayaClassifier sur un split de validation
(gen_corpus/laya_val.jsonl) — exactitude simple (commande exacte, "aucune"
compris). Cf. docs/superpowers/specs/2026-09-28-routeur-laya-design.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(classifier, val_path: Path) -> float:
    total = 0
    correct = 0
    for line in Path(val_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        expected = row["answers"]["command"]
        command, _ = classifier.classify(row["state"])
        predicted = command or "aucune"
        total += 1
        if predicted == expected:
            correct += 1
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--val", default="gen_corpus/laya_val.jsonl")
    args = parser.parse_args()

    from router_service.laya_classifier import LayaClassifier
    classifier = LayaClassifier(args.checkpoint)
    accuracy = evaluate(classifier, Path(args.val))
    print(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest router_service/test_eval_laya.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add router_service/eval_laya.py router_service/test_eval_laya.py
git commit -m "feat(router): script d'évaluation hors-ligne du classifieur

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5 : Brancher LayaClassifier dans server.py

**Files:**
- Modify: `router_service/server.py`
- Modify: `router_service/test_router.py` (le reste du fichier au-delà des
  16 premières lignes remplacées à la Task 2 — les tests HTTP/`call_adapter`)

**Interfaces:**
- Consumes: `LayaClassifier` (Task 3), `embed_bge_m3` (Task 2).
- Produces: `route_message(message: str) -> dict` avec la même forme de
  retour qu'aujourd'hui (`{"status": "ok"|"no_match"|"answer", ...}`) —
  c'est l'interface que `derisk-deleg` (`callRouter`) consomme déjà, elle
  ne change pas.

- [ ] **Step 1: Write the failing test**

Remplacer tout le contenu de `router_service/test_router.py` **à partir de
la ligne `import json`** (juste après les deux tests ajoutés à la Task 2,
`test_embed_bge_m3_returns_normalized_vectors` et
`test_command_sets_disjoint`, qui restent en tête de fichier inchangés) par :

```python
# router_service/test_router.py (tout le fichier à partir de la ligne
# `import json` — les deux tests embed_bge_m3/command_sets_disjoint de la
# Task 2 restent au-dessus, inchangés)
import json
import threading
import time
import urllib.request

from router_service import server as router_server


class _FakeClassifier:
    def __init__(self, command, prob=0.9):
        self._command = command
        self._prob = prob

    def classify(self, message):
        return self._command, self._prob


def test_route_endpoint_end_to_end():
    router_server._classifier = _FakeClassifier(None)
    router_server._faq = None
    httpd = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
        ("127.0.0.1", 0), router_server.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/route",
            data=json.dumps({"message": "/wikistatus"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.load(urllib.request.urlopen(req, timeout=5))
        assert resp["status"] in ("ok", "no_match")
    finally:
        httpd.shutdown()


def test_explicit_command_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier(("boom")))
    r = router_server.route_message("/wikistatus")
    assert r == {"status": "ok", "command": "/wikistatus", "args": ""}


def test_explicit_gog_command_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/inbox")
    assert r == {"status": "ok", "command": "/inbox", "args": ""}


def test_explicit_command_empty_required_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/c")
    assert r == {"status": "answer", "reply": "Usage : /c <argument>"}


def test_explicit_r_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/r transformers attention")
    assert r == {"status": "ok", "command": "/r", "args": "transformers attention"}


def test_explicit_r_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/r")
    assert r == {"status": "answer", "reply": "Usage : /r <argument>"}


def test_explicit_supprimer_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/supprimer src-a")
    assert r == {"status": "ok", "command": "/supprimer", "args": "src-a"}


def test_explicit_supprimer_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/supprimer")
    assert r == {"status": "answer", "reply": "Usage : /supprimer <argument>"}


def test_explicit_relire_no_arg_needed(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    assert router_server.route_message("/relire") == {"status": "ok", "command": "/relire", "args": ""}


def test_explicit_verifie_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/verifie src-a")
    assert r == {"status": "ok", "command": "/verifie", "args": "src-a"}


def test_explicit_verifie_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/verifie")
    assert r == {"status": "answer", "reply": "Usage : /verifie <argument>"}


def test_explicit_tags_kbupdate_no_arg_needed(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    assert router_server.route_message("/tags") == {"status": "ok", "command": "/tags", "args": ""}
    assert router_server.route_message("/kbupdate") == {"status": "ok", "command": "/kbupdate", "args": ""}


def test_explicit_lire_bypasses_slm(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("boom"))
    r = router_server.route_message("/lire 18ab3f2")
    assert r == {"status": "ok", "command": "/lire", "args": "18ab3f2"}


def test_explicit_lire_empty_arg_returns_usage(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/ingest"))
    r = router_server.route_message("/lire")
    assert r == {"status": "answer", "reply": "Usage : /lire <argument>"}


def test_unknown_slash_command_still_reaches_slm(monkeypatch):
    called = {"n": 0}

    class _Counting:
        def classify(self, message):
            called["n"] += 1
            return None, 0.0

    monkeypatch.setattr(router_server, "_classifier", _Counting())
    router_server.route_message("/inconnu bla")
    assert called["n"] == 1


def test_nl_inferred_command_uses_full_message_as_args(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/q", 0.92))
    r = router_server.route_message("qu'est-ce que le SPLADE ?")
    assert r == {"status": "ok", "command": "/q", "args": "qu'est-ce que le SPLADE ?"}


def test_nl_no_match_returns_no_match(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier(None))
    r = router_server.route_message("il fait beau aujourd'hui")
    assert r == {"status": "no_match"}


def test_gog_command_below_confidence_threshold_returns_no_match(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/inbox", 0.30))
    r = router_server.route_message("y a-t-il du nouveau ?")
    assert r == {"status": "no_match"}


def test_gog_command_above_confidence_threshold_is_ok(monkeypatch):
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/inbox", 0.80))
    r = router_server.route_message("y a-t-il du nouveau ?")
    assert r == {"status": "ok", "command": "/inbox", "args": "y a-t-il du nouveau ?"}


def test_wiki_command_from_slm_does_not_use_confidence_threshold(monkeypatch):
    # Seules les commandes gog passent par le seuil de confiance — le
    # classifieur ne renvoie de toute façon que des commandes connues.
    monkeypatch.setattr(router_server, "_classifier", _FakeClassifier("/q", 0.10))
    r = router_server.route_message("dis-moi ce que dit le wiki sur X")
    assert r == {"status": "ok", "command": "/q", "args": "dis-moi ce que dit le wiki sur X"}
```

(Les tests FAQ/`faq.py` ne changent pas — non reproduits ici s'il en
existe dans le fichier d'origine ; les conserver tels quels.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest router_service/test_router.py -v`
Expected: FAIL — `AttributeError: module 'router_service.server' has no attribute '_classifier'`

- [ ] **Step 3: Write minimal implementation**

Remplacer le contenu de `router_service/server.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service HTTP du routeur Tiron : POST /route {message} -> {status, command, args}."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router_service.router import WIKI_CMDS, GOG_CMDS, embed_bge_m3
from router_service.faq import FaqIndex, FAQ_PATH
from router_service.laya_classifier import LayaClassifier

LAYA_CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "convaiinnovations/laya")
SEUIL_GOG = 0.50

# Commandes qui exigent un argument : un appel sans argument est une erreur
# d'usage, pas une capture/requête vide à déléguer.
NEEDS_ARG = {"/c", "/q", "/source", "/chercher", "/repondre", "/r", "/lire", "/supprimer", "/verifie"}

_classifier = None  # chargé au démarrage (Step 5)
_faq = None         # FaqIndex, chargé au démarrage


def route_message(message: str) -> dict:
    # Commande explicite (l'utilisateur a tapé /c, /q, …) : honorée telle quelle,
    # jamais soumise à Laya. Le routeur ne classe que le texte libre ; re-classer
    # une commande explicite ne peut que la corrompre (ex. /c pris pour /source).
    stripped = message.strip()
    if stripped.startswith("/"):
        parts = stripped.split(None, 1)
        cmd = parts[0]
        if cmd in WIKI_CMDS or cmd in GOG_CMDS:
            args = parts[1] if len(parts) > 1 else ""
            if cmd in NEEDS_ARG and not args.strip():
                return {"status": "answer", "reply": f"Usage : {cmd} <argument>"}
            return {"status": "ok", "command": cmd, "args": args}
    if _faq is not None and not message.lstrip().startswith("/"):
        try:
            entry = _faq.lookup(message)
        except Exception:
            entry = None
        if entry is not None:
            return {"status": "answer", "reply": entry["answer"]}

    command, probability = _classifier.classify(message)
    if command is None:
        return {"status": "no_match"}
    if command in GOG_CMDS and probability < SEUIL_GOG:
        return {"status": "no_match"}
    # args : jamais généré, toujours le message brut pour une commande
    # inférée en langage naturel (cf. décision du 2026-09-28).
    return {"status": "ok", "command": command, "args": message}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/route":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        result = route_message(body.get("message", ""))
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # silence les logs d'accès par défaut


def main() -> None:
    global _classifier, _faq
    print("Chargement du classifieur Laya...", flush=True)
    _classifier = LayaClassifier(LAYA_CHECKPOINT)
    _faq = FaqIndex(embed_bge_m3)
    print(f"FAQ chargée ({FAQ_PATH})", flush=True)
    print("Prêt, écoute sur :8999", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8999), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest router_service/test_router.py -v`
Expected: PASS (tous les tests)

- [ ] **Step 5: Commit**

```bash
git add router_service/server.py router_service/test_router.py
git commit -m "feat(router): branche LayaClassifier, retire call_adapter et GogGate

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6 : [Manuel] Entraîner le checkpoint sur Kaggle et l'évaluer

Cette tâche s'exécute hors du dépôt, par l'utilisateur — pas de code à
écrire, mais un résultat vérifiable avant de continuer.

- [ ] **Step 1** : Ouvrir `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
  du dépôt `github.com/NandhaKishorM/laya` sur Kaggle (2× T4 gratuits).
- [ ] **Step 2** : Uploader `gen_corpus/laya_train.jsonl` et
  `gen_corpus/laya_val.jsonl` (produits à la Task 1) comme dataset d'entrée
  du notebook, à la place de son propre jeu de démonstration.
- [ ] **Step 3** : Lancer l'entraînement (construction dataset → RLCD →
  calibration de température → évaluation → push Hub), suivre les
  instructions du notebook telles quelles.
- [ ] **Step 4** : Noter l'identifiant du checkpoint poussé sur Hugging Face
  Hub (ex. `votre-org/laya-tiron-router`).
- [ ] **Step 5** : Vérifier hors-ligne avec le script de la Task 4 :
  `python3 -m router_service.eval_laya votre-org/laya-tiron-router`
  — noter l'exactitude rapportée.
- [ ] **Step 6** : Comparer à l'exactitude du LoRA phi-4-mini actuel (valeur
  déjà mesurée par le passé, cf. mémoire projet routage) — si Laya est du
  même ordre de grandeur ou meilleur, passer à la Task 7 ; sinon, revenir à
  la Task 1 (corpus/critères) avant de continuer, ne pas déployer un
  classifieur moins bon sans discussion. **Angle mort confirmé en revue
  finale** : `/supprimer`, `/relire` et `/verifie` n'ont aucun exemple dans
  le corpus d'entraînement — regarder l'exactitude par classe (pas
  seulement l'exactitude globale) avant de conclure "du même ordre ou
  meilleur", en particulier sur ces trois classes-là. Le LoRA actuel a la
  même lacune (pas de régression introduite par Laya), mais ça reste un
  angle mort à connaître avant de déployer.

---

### Task 7 : Déployer le checkpoint sur sanroque

**Files:**
- Modify: fichier d'environnement du service `router_service` sur sanroque
  (à localiser : `grep -rl LAYA_CHECKPOINT` ou l'unité systemd du service
  routeur — non retrouvé dans cette session, à identifier au moment de
  l'exécution).

**Note d'exécution** : cette tâche touche la machine hôte réelle (sanroque),
pas le worktree — elle s'exécute depuis une session ayant accès à
sanroque directement (hors de ce worktree isolé), après que les Tasks 1-5
aient été fusionnées.

- [ ] **Step 1** : Ajouter `LAYA_CHECKPOINT=<identifiant Hub de la Task 6>`
  à l'environnement du service `router_service` (fichier `.env` ou unité
  systemd, selon ce qui est trouvé).
- [ ] **Step 2** : `pip install laya` (ou `laya[onnx]` pour de meilleures
  performances CPU, cf. spec) dans l'environnement Python du service.
- [ ] **Step 3** : Redémarrer le service (`systemctl --user restart
  <nom-du-service-routeur>` — confirmer le nom exact avant de lancer,
  demander confirmation à l'utilisateur avant le restart comme d'habitude).
- [ ] **Step 4** : Fumée-test :
  `curl -s -X POST http://127.0.0.1:8999/route -H "Content-Type: application/json" -d '{"message":"quel est l'\''état du wiki ?"}'`
  — vérifier une réponse `{"status": "ok", "command": "/wikistatus", ...}`
  ou `no_match`, pas d'erreur 500.
- [ ] **Step 5** : Tester quelques messages réels côté Telegram (dev,
  `@secretarius_tiron_bot`) couvrant wiki, gog, et un message hors sujet.

---

### Task 8 : Décommissionner phi-4-mini/llama.cpp

**Files:** aucun fichier de code — nettoyage opérationnel.

**Note d'exécution** : comme la Task 7, touche la machine hôte réelle,
hors de ce worktree isolé.

- [ ] **Step 1** : Confirmer que plus rien n'appelle
  `http://127.0.0.1:8998` (recherche : `grep -rn "8998"` sur le dépôt
  fusionné, hors documentation/mémoire). Vérifier et corriger en particulier
  les orphelins identifiés en revue finale (2026-09-28), qui référencent
  encore l'ancien routeur génératif phi-4-mini/`call_adapter()` :
  - `switch-brain.sh` (hors worktree, tourne sur sanroque) écrit encore
    `TIRON_LLAMA_BASE`/`TIRON_LLAMA_KEY` dans l'environnement du routeur —
    variables que `router_service/server.py` ne lit plus depuis le passage
    à Laya ; à retirer ou adapter.
  - `tests/test_switch_brain.py` — vérifier s'il teste encore ces variables
    d'environnement obsolètes.
  - `openclaw-config/install.sh` — vérifier les mentions de l'ancien
    endpoint 8998 dans les étapes d'installation/config.
  - `README.md` — vérifier les mentions de l'ancien endpoint 8998.
  - la ligne `Description=` de l'unité systemd `tiron-router.service`, qui
    mentionne encore « BGE-M3 gate » (garde-fou GogGate retiré à la Task 2,
    remplacé par le score calibré Laya) — à mettre à jour.
- [ ] **Step 2** : Arrêter et désactiver le service systemd llama.cpp
  correspondant (nom exact à confirmer sur sanroque — demander confirmation
  avant `systemctl stop`/`disable`, comme pour toute action `systemctl`).
- [ ] **Step 3** : Mettre à jour la documentation qui mentionne ce service
  (`docs/architecture/` ou équivalent — rechercher `8998` dans `docs/`) pour
  refléter le remplacement par Laya.
- [ ] **Step 4** : Commit de la mise à jour documentaire.

```bash
git add -A
git commit -m "docs: routeur Tiron migré vers Laya, phi-4-mini/llama.cpp décommissionné

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Couverture de la spec** : format de données (Task 1), suppression
GogGate + extraction embed_bge_m3 (Task 2), classifieur Laya (Task 3),
évaluation hors-ligne (Task 4), branchement server.py + args heuristique +
seuil de confiance (Task 5), entraînement Kaggle (Task 6, manuel),
déploiement sanroque (Task 7, hors worktree), décommissionnement
phi-4-mini (Task 8, hors worktree). Santiago explicitement hors périmètre
(spec). FAQ explicitement inchangée (Task 2 la préserve via `embed_bge_m3`).

**Incertitude levée en cours d'exécution** : la Task 3 signalait initialement
que l'appel d'inférence exact du SDK Laya restait à vérifier. Confirmé
pendant la revue de cette tâche (2026-09-28, README `NandhaKishorM/laya`) :
`agent.predict(state, questions)`, réponse
`result["answers"][q]["choice"]`/`["confidence"]`. Le code ci-dessus est à
jour avec cette forme confirmée.

**Cohérence des types** : `LayaClassifier.classify()` retourne
`tuple[str | None, float]` partout (Task 3 le définit, Task 4 et Task 5 le
consomment à l'identique). `COMMAND_CRITERIA` défini une seule fois (Task
1) et importé tel quel par la Task 3, jamais redéfini.

**Corrigé lors du scan pré-vol (2026-09-28, avant dispatch de la Task 1)** :
(1) toutes les commandes `Run:`/`git commit` préfixaient `cd ~/Secretarius`
— incompatible avec l'exécution en worktree isolé (le harnais bloque le
`cd` hors du worktree) — préfixe retiré partout, les commandes s'exécutent
depuis la racine du worktree. (2) `gen_corpus` n'avait pas de
`__init__.py` — ajouté à la Task 1 pour garantir la résolution de l'import
`from gen_corpus.to_laya_format import ...` à la Task 3. (3) La Task 2
renvoyait par erreur à « la Task 4 » pour la suite du fichier de test —
corrigé en « la Task 5 » (c'est elle qui modifie `server.py`/le reste de
`test_router.py`).
