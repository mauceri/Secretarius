# Génération de page wiki par phi-4 via passages centraux — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer l'appel LLM externe qui génère la page source à l'ingestion du wiki par une version locale phi-4, rendue possible par une sélection de passages centraux (PacSum).

**Architecture:** Un module de sélection (PacSum sur embeddings BGE-M3) comprime la source à ~6000 caractères ; phi-4 nu produit un JSON structuré ; un assembleur déterministe fabrique `src-<slug>.md` au format existant. Seule la méthode `_generate_source_page` d'`ingest.py` change.

**Tech Stack:** Python 3 (numpy, nltk, sentence-transformers/BGE-M3), llama.cpp phi-4 nu sur 8998, juge DeepSeek. Environnement : venv `~/Secretarius/Wiki_LM/.venv` (déps : numpy, nltk, sentence-transformers).

## Global Constraints

- Ne modifier QUE la méthode `_generate_source_page` dans `Wiki_LM/tools/ingest.py`. Ne pas toucher aux pages concept/entité (elles gardent `self.llm.complete`). Format de page produit **inchangé** (frontmatter + `## Résumé` + `## Points clés` + `## Concepts et entités mentionnés` avec lignes `- concept:` / `- entité:` + `## Liens internes suggérés`).
- phi-4 nu via `"lora":[{"id":0,"scale":0}]` par-requête sur `http://127.0.0.1:8998`. Ne JAMAIS changer le `-c 2048` du serveur ni toucher au port prod.
- Budget de sélection : **6000 caractères** (≈1600 tokens) par défaut, surchargeable.
- Encodeur : BGE-M3 (`SentenceTransformer("BAAI/bge-m3")`, `normalize_embeddings=True`).
- Sélection PacSum ; **l'ordre d'origine des phrases retenues est préservé**.
- phi-4 émet le JSON `{resume, points_cles, concepts, entites, tags}` ; le **code** assemble le markdown (phi-4 ne rédige pas le YAML).
- Tests lancés depuis `~/Secretarius/Wiki_LM/tools` (imports plats, ex. `from central_passages import ...`) avec le python du venv Wiki_LM.

---

### Task 1 : `central_passages.py` — sélection PacSum

**Files:**
- Create: `Wiki_LM/tools/central_passages.py`
- Test: `Wiki_LM/tools/test_central_passages.py`

**Interfaces:**
- Produces : `select_central_passages(text: str, budget_chars: int = 6000, embed_fn=None) -> str` ; `pacsum_scores(embeddings: np.ndarray, lambda1, lambda2, beta) -> np.ndarray` ; `clean_text(str) -> str` ; `split_sentences(str) -> list[str]`. `embed_fn(list[str]) -> np.ndarray` renvoie une matrice `[n, d]` L2-normalisée.

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# Wiki_LM/tools/test_central_passages.py
import numpy as np
from central_passages import pacsum_scores, select_central_passages, split_sentences


def test_split_sentences_fr():
    s = split_sentences("Bonjour le monde. Ceci est un test.")
    assert s == ["Bonjour le monde.", "Ceci est un test."]


def test_pacsum_favorise_la_phrase_centrale():
    # phrase 0 proche de 1 et 2 ; 1 et 2 éloignées entre elles -> 0 centrale
    e = np.array([[1.0, 0.0], [0.9, 0.436], [0.9, -0.436]], dtype=np.float32)
    # normaliser
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    sc = pacsum_scores(e, lambda1=1.0, lambda2=1.0, beta=0.0)
    assert sc.argmax() == 0


def test_texte_court_renvoye_tel_quel():
    txt = "Une note courte."
    assert select_central_passages(txt, budget_chars=6000) == "Une note courte."


def test_selection_respecte_budget_et_ordre():
    phrases = [f"Phrase numero {i} avec du texte de remplissage." for i in range(20)]
    texte = " ".join(phrases)

    def stub_embed(sents):
        # phrase i : vecteur one-hot bruité, la phrase 0 similaire à toutes (centrale)
        n = len(sents)
        m = np.eye(n, dtype=np.float32)
        m[:, 0] += 0.5  # tout le monde ressemble un peu à la phrase 0
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    out = select_central_passages(texte, budget_chars=200, embed_fn=stub_embed)
    assert len(out) <= 200
    # ordre d'origine préservé : les numéros retenus sont croissants
    import re
    nums = [int(x) for x in re.findall(r"numero (\d+)", out)]
    assert nums == sorted(nums)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_central_passages.py -v`
Expected: FAIL (`ModuleNotFoundError: central_passages`).

- [ ] **Step 3 : Écrire `central_passages.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sélection de passages centraux (PacSum sur embeddings BGE-M3) pour comprimer une
source à un budget de caractères, en vue d'une génération phi-4.
clean_text/split_sentences sont copiés depuis ~/indexation_wiki40b/chunk_data.py."""
import os
import re
from typing import Callable, Optional

import numpy as np
import nltk

_MULTI_NL_RE = re.compile(r"\n{3,}")
_MULTI_SP_RE = re.compile(r"[ \t]{2,}")

LAMBDA1 = float(os.environ.get("PACSUM_LAMBDA1", "-0.2"))  # phrases précédentes (j<i)
LAMBDA2 = float(os.environ.get("PACSUM_LAMBDA2", "1.0"))   # phrases suivantes (j>i), biais position
BETA = float(os.environ.get("PACSUM_BETA", "0.0"))          # seuil soustrait aux similarités


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_NL_RE.sub("\n\n", s)
    s = _MULTI_SP_RE.sub(" ", s)
    return s.strip()


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def split_sentences(text: str) -> list[str]:
    _ensure_punkt()
    try:
        sents = nltk.sent_tokenize(text, language="french")
    except Exception:
        sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s and s.strip()]


def pacsum_scores(embeddings: np.ndarray, lambda1: float = LAMBDA1,
                  lambda2: float = LAMBDA2, beta: float = BETA) -> np.ndarray:
    sim = embeddings @ embeddings.T            # cosinus (vecteurs normalisés)
    np.fill_diagonal(sim, 0.0)
    e = sim - beta
    n = e.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        back = e[i, :i].sum() if i > 0 else 0.0
        fwd = e[i, i + 1:].sum() if i < n - 1 else 0.0
        scores[i] = lambda1 * back + lambda2 * fwd
    return scores


_prod_model = None


def _bge_m3_embed(sentences: list[str]) -> np.ndarray:
    global _prod_model
    if _prod_model is None:
        from sentence_transformers import SentenceTransformer
        _prod_model = SentenceTransformer("BAAI/bge-m3")
    return _prod_model.encode(sentences, convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)


def select_central_passages(text: str, budget_chars: int = 6000,
                            embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= budget_chars:
        return cleaned
    sentences = split_sentences(cleaned)
    if len(sentences) <= 1:
        return cleaned[:budget_chars]
    embed = embed_fn or _bge_m3_embed
    embeddings = embed(sentences)
    scores = pacsum_scores(embeddings)
    order = np.argsort(-scores)                # score décroissant
    chosen: list[int] = []
    total = 0
    for idx in order:
        s = sentences[int(idx)]
        if chosen and total + len(s) + 1 > budget_chars:
            break
        chosen.append(int(idx))
        total += len(s) + 1
    chosen.sort()                              # restituer l'ordre d'origine
    return " ".join(sentences[i] for i in chosen)
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_central_passages.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/central_passages.py Wiki_LM/tools/test_central_passages.py
git commit -m "feat(wiki): sélection de passages centraux PacSum (BGE-M3)"
```

---

### Task 2 : `page_phi4.py::generate_page_content` — génération phi-4 nu

**Files:**
- Create: `Wiki_LM/tools/page_phi4.py`
- Test: `Wiki_LM/tools/test_page_phi4.py`

**Interfaces:**
- Produces : `generate_page_content(passages: str, base_url: str = PHI4_BASE) -> dict` renvoyant `{"resume", "points_cles", "concepts", "entites", "tags"}` ; constante `PHI4_BASE`.

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# Wiki_LM/tools/test_page_phi4.py
import io
import json
import urllib.request
from page_phi4 import generate_page_content


def test_generate_envoie_lora_nu_et_schema(monkeypatch):
    captured = {}

    class FakeResp(io.BytesIO):
        pass

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        payload = {"choices": [{"message": {"content": json.dumps({
            "resume": "Un résumé.", "points_cles": ["p1"],
            "concepts": ["c1"], "entites": ["e1"], "tags": ["t1"]})}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = generate_page_content("des passages", base_url="http://x")
    assert out["resume"] == "Un résumé."
    assert out["entites"] == ["e1"]
    # phi-4 nu par-requête + schéma contraint
    assert captured["body"]["lora"] == [{"id": 0, "scale": 0}]
    assert "json_schema" in captured["body"]
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_page_phi4.py -v`
Expected: FAIL (`ModuleNotFoundError: page_phi4`).

- [ ] **Step 3 : Écrire `page_phi4.py` (partie génération)**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génération de la page source via phi-4 nu (JSON contraint) + assemblage
déterministe du markdown au format wiki existant."""
import json
import os
import urllib.request

PHI4_BASE = os.environ.get("PHI4_BASE_URL", "http://127.0.0.1:8998")

_SYSTEM = ("Tu es l'assistant d'un wiki personnel. À partir UNIQUEMENT des passages "
           "fournis, produis un résumé fidèle et concis en français. N'invente rien "
           "qui ne figure pas dans les passages.")

_SCHEMA = {
    "type": "object",
    "properties": {
        "resume": {"type": "string"},
        "points_cles": {"type": "array", "items": {"type": "string"}},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "entites": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["resume", "points_cles", "concepts", "entites", "tags"],
}


def generate_page_content(passages: str, base_url: str = PHI4_BASE) -> dict:
    user = ("Passages :\n" + passages + "\n\n"
            "Produis un JSON : resume (3 à 5 phrases), points_cles (liste de points), "
            "concepts (concepts abstraits cités), entites (personnes/outils/organisations "
            "cités), tags (3 à 6 mots-clés).")
    body = {
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": 600, "temperature": 0,
        "lora": [{"id": 0, "scale": 0}],   # phi-4 nu, par-requête (ne touche pas l'état global)
        "json_schema": _SCHEMA,
    }
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return json.loads(d["choices"][0]["message"]["content"])
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_page_phi4.py -v`
Expected: PASS.

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/page_phi4.py Wiki_LM/tools/test_page_phi4.py
git commit -m "feat(wiki): génération de page source via phi-4 nu (JSON contraint)"
```

---

### Task 3 : `page_phi4.py::assemble_source_page` — assemblage déterministe

**Files:**
- Modify: `Wiki_LM/tools/page_phi4.py`
- Test: `Wiki_LM/tools/test_page_phi4.py`

**Interfaces:**
- Consumes : le dict produit par `generate_page_content`.
- Produces : `assemble_source_page(title: str, today: str, data: dict, tags: list[str]) -> str` — markdown au format wiki existant, YAML valide.

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# à ajouter dans Wiki_LM/tools/test_page_phi4.py
import frontmatter
from page_phi4 import assemble_source_page


def test_assemble_format_wiki():
    data = {"resume": "Résumé ici.", "points_cles": ["pt un", "pt deux"],
            "concepts": ["mémex"], "entites": ["Vannevar Bush"], "tags": ["histoire"]}
    md = assemble_source_page("As We May Think", "2026-07-09", data, ["technique"])
    post = frontmatter.loads(md)                 # YAML valide, sinon lève
    assert post["category"] == "source"
    assert post["title"] == "As We May Think"
    assert "technique" in post["tags"] and "histoire" in post["tags"]
    body = post.content
    assert "## Résumé" in body and "Résumé ici." in body
    assert "## Points clés" in body and "- pt un" in body
    assert "- concept: mémex" in body
    assert "- entité: Vannevar Bush" in body
    assert "## Liens internes suggérés" in body
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_page_phi4.py::test_assemble_format_wiki -v`
Expected: FAIL (`ImportError: cannot import name 'assemble_source_page'`).

- [ ] **Step 3 : Ajouter `assemble_source_page` à `page_phi4.py`**

```python
# ajouter à la fin de page_phi4.py
def _bullets(items, prefix: str = "") -> str:
    return "\n".join(f"- {prefix}{x}" for x in items) if items else ""


def assemble_source_page(title: str, today: str, data: dict, tags: list[str]) -> str:
    tags_uniq = list(dict.fromkeys(tags))                  # dédup, ordre préservé
    tags_str = ", ".join(json.dumps(t, ensure_ascii=False) for t in tags_uniq)
    points = _bullets(data.get("points_cles", []))
    concepts = _bullets(data.get("concepts", []), "concept: ")
    entites = _bullets(data.get("entites", []), "entité: ")
    conc_ent = "\n".join(x for x in (concepts, entites) if x) or "Aucun"
    resume = (data.get("resume") or "").strip()
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        "category: source\n"
        f"tags: [{tags_str}]\n"
        f"created: {today}\n"
        "sources: []\n"
        "---\n\n"
        f"# {title}\n\n"
        "## Résumé\n\n"
        f"{resume}\n\n"
        "## Points clés\n\n"
        f"{points}\n\n"
        "## Concepts et entités mentionnés\n\n"
        f"{conc_ent}\n\n"
        "## Liens internes suggérés\n\n"
        "Aucun\n"
    )
```

- [ ] **Step 4 : Lancer toute la suite `test_page_phi4.py`, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_page_phi4.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/page_phi4.py Wiki_LM/tools/test_page_phi4.py
git commit -m "feat(wiki): assemblage déterministe de la page source (format inchangé)"
```

---

### Task 4 : Câblage dans `ingest.py::_generate_source_page`

**Files:**
- Modify: `Wiki_LM/tools/ingest.py` (imports en tête ; corps de `_generate_source_page`, lignes 1203-1220)
- Test: `Wiki_LM/tools/test_ingest_source_page.py`

**Interfaces:**
- Consumes : `select_central_passages` (Task 1), `generate_page_content` + `assemble_source_page` (Tasks 2-3).
- Produces : `_generate_source_page(self, content, source_name, extra_tags=None) -> str` renvoie désormais le markdown assemblé par phi-4 (même signature, même type de retour).

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# Wiki_LM/tools/test_ingest_source_page.py
import ingest


def test_generate_source_page_utilise_phi4(monkeypatch):
    monkeypatch.setattr(ingest, "select_central_passages", lambda text, **k: "passages réduits")
    monkeypatch.setattr(ingest, "generate_page_content", lambda passages, **k: {
        "resume": "R.", "points_cles": ["p"], "concepts": ["c"],
        "entites": ["Napoléon"], "tags": ["histoire"]})

    ing = ingest.Ingestor.__new__(ingest.Ingestor)   # bypass __init__ (pas de LLM/wiki_dir)
    ing.today = "2026-07-09"
    md = ing._generate_source_page("un long contenu source", "Ma Source", extra_tags=["arbath"])

    assert "category: source" in md
    assert "# Ma Source" in md
    assert "- entité: Napoléon" in md
    assert "arbath" in md and "histoire" in md   # tags fusionnés
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_ingest_source_page.py -v`
Expected: FAIL (l'ancienne méthode appelle `self.llm.complete` → `AttributeError` sur l'objet sans `__init__`, ou le format ne matche pas).

- [ ] **Step 3 : Câbler `ingest.py`**

Ajouter les imports près des autres imports plats en tête d'`ingest.py` (après `from llm import LLM`) :

```python
from central_passages import select_central_passages
from page_phi4 import generate_page_content, assemble_source_page
```

Remplacer le corps de `_generate_source_page` (lignes 1203-1220) par :

```python
    def _generate_source_page(
        self, content: str, source_name: str, extra_tags: list[str] | None = None
    ) -> str:
        passages = select_central_passages(content)
        data = generate_page_content(passages)
        tags = list(extra_tags or []) + list(data.get("tags", []))
        return assemble_source_page(source_name, self.today, data, tags)
```

Ne pas toucher au reste (`_generate_concept_page`, `_generate_entity_page` gardent `self.llm.complete`). Note : `_PROMPT_SOURCE_PAGE`, `_kb_context` et `_truncate` peuvent devenir inutilisés par cette seule méthode ; **vérifier avec `grep -n "_PROMPT_SOURCE_PAGE\|_kb_context\|_truncate" ingest.py`** — ne supprimer que ceux dont il ne reste AUCUN autre usage, laisser les autres intacts.

- [ ] **Step 4 : Lancer, vérifier le succès (+ non-régression des tests wiki existants)**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_ingest_source_page.py -v`
Expected: PASS.
Run: `cd ~/Secretarius/Wiki_LM && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest tests/ -q`
Expected: pas de nouvelle régression (les échecs préexistants éventuels — réseau/services — restent tels quels).

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/ingest.py Wiki_LM/tools/test_ingest_source_page.py
git commit -m "feat(wiki): _generate_source_page passe par phi-4 (passages centraux)"
```

---

### Task 5 : Harnais d'évaluation (juge fidélité + couverture)

**Files:**
- Create: `Wiki_LM/tools/eval_resume.py`
- Test: `Wiki_LM/tools/test_eval_resume.py`

**Interfaces:**
- Produces : `judge_resume(source: str, reference: str, candidate: str) -> dict` renvoyant `{"fidelite": int, "couverture": int}` (1..5) ; `run_eval(paires: list[dict]) -> list[dict]` qui, pour chaque `{source, reference}`, produit la page phi-4 (select→generate→assemble) et renvoie les deux notes.

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# Wiki_LM/tools/test_eval_resume.py
import io
import json
import urllib.request
import eval_resume


def test_judge_resume_parse_les_deux_notes(monkeypatch):
    def fake_urlopen(req, timeout=0):
        payload = {"choices": [{"message": {"content": json.dumps(
            {"fidelite": 5, "couverture": 4})}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    out = eval_resume.judge_resume("source", "reference", "candidat")
    assert out == {"fidelite": 5, "couverture": 4}
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_eval_resume.py -v`
Expected: FAIL (`ModuleNotFoundError: eval_resume`).

- [ ] **Step 3 : Écrire `eval_resume.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Éval du résumeur phi-4 : le juge DeepSeek note fidélité (candidat vs source) et
couverture (candidat vs page de référence Euria), sur 1..5."""
import json
import os
import urllib.request

from central_passages import select_central_passages
from page_phi4 import generate_page_content, assemble_source_page

_JUGE = ("Tu es un évaluateur strict. Compare un RÉSUMÉ CANDIDAT à sa SOURCE et à une "
         "page de RÉFÉRENCE. Note de 1 à 5 :\n"
         "- fidelite : le candidat n'affirme rien qui ne soit dans la SOURCE (5 = aucune "
         "invention, 1 = hallucinations).\n"
         "- couverture : le candidat couvre le contenu central de la RÉFÉRENCE (5 = tout "
         "l'essentiel, 1 = manque l'essentiel).\n"
         'Réponds UNIQUEMENT par un JSON {"fidelite": <int>, "couverture": <int>}.')


def judge_resume(source: str, reference: str, candidate: str) -> dict:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    user = f"SOURCE:\n{source}\n\nRÉFÉRENCE:\n{reference}\n\nCANDIDAT:\n{candidate}"
    body = {"model": "deepseek-chat",
            "messages": [{"role": "system", "content": _JUGE},
                         {"role": "user", "content": user}],
            "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    raw = d["choices"][0]["message"]["content"]
    start, end = raw.find("{"), raw.rfind("}")
    parsed = json.loads(raw[start:end + 1])
    return {"fidelite": int(parsed["fidelite"]), "couverture": int(parsed["couverture"])}


def run_eval(paires: list[dict]) -> list[dict]:
    """paires : [{"source": str, "reference": str, "titre": str}]"""
    resultats = []
    for p in paires:
        passages = select_central_passages(p["source"])
        data = generate_page_content(passages)
        candidate = assemble_source_page(p.get("titre", "Source"), "2026-07-09",
                                         data, data.get("tags", []))
        notes = judge_resume(passages, p["reference"], candidate)
        resultats.append({"titre": p.get("titre"), **notes})
    return resultats
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `cd ~/Secretarius/Wiki_LM/tools && ~/Secretarius/Wiki_LM/.venv/bin/python -m pytest test_eval_resume.py -v`
Expected: PASS.

- [ ] **Step 5 : Commit**

```bash
cd ~/Secretarius && git add Wiki_LM/tools/eval_resume.py Wiki_LM/tools/test_eval_resume.py
git commit -m "feat(wiki): harnais d'éval résumé (juge fidélité + couverture)"
```

---

## Après les 5 tâches — mesure réelle (manuelle, hors TDD)

1. **Assembler un mini jeu de test** (3-4 paires) dans `Wiki_LM/tools/eval_data/` : pour quelques pages `src-*.md` existantes (ex. jade, Eckhart), stocker `{titre, source: texte brut re-lu via ingest._read_source, reference: contenu de la page Euria actuelle}`.
2. Lancer `run_eval` avec `DEEPSEEK_API_KEY` défini ; relever les notes fidélité/couverture moyennes.
3. **Figer le seuil « potable »** (comme pour la FAQ) à partir de ce premier run — c'est le critère de succès chiffré, à consigner dans le verdict.
4. Comparer la page phi-4 à la page Euria sur 1-2 exemples réels (lecture humaine) pour confirmer le juge.

## Déploiement (après validation)

- Redémarrer le service d'ingestion / le worker wiki pour charger le nouveau code. phi-4 nu est déjà servi par 8998 (scale lora 0 par-requête). Aucun changement de `-c`.
- Surveiller le temps d'ingestion (prefill phi-4 sur passages ~6k car. — acceptable en async).

## Hors périmètre (rappel)

Enrichissement des pages d'entités (reste `self.llm`/Euria) ; RST/EDU ; longs documents techniques ; verticaux métier.
