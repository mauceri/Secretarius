# Détection et réponse « question-Secretarius » — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesurer hors-ligne si un 4ᵉ centroïde BGE-M3 « secretarius » sépare fiablement les questions générales sur Secretarius des commandes wiki/gog, et si phi-4 nu y répond correctement — pour décider (verdict) avant toute intégration.

**Architecture:** On réutilise le classifieur par centroïdes `GogGate` (BGE-M3) déjà en production, sans le modifier : un nouveau module `gen_corpus_qa/` compose un `GogGate` et lui ajoute un centroïde « secretarius » plus une méthode `classify`. La réponse réutilise `set_lora_scale`+`infer_llama` de `eval_qa.py` contre un llama-server de **test** (port 8996, jamais la prod 8998). Un harnais produit une matrice de confusion + des notes de réponse, puis un verdict.

**Tech Stack:** Python 3 (BGE-M3 via transformers/torch), llama.cpp `build-rocm/bin/llama-server`, DeepSeek API (juge), pytest.

## Global Constraints

- Spec de référence : `docs/superpowers/specs/2026-07-08-detection-question-secretarius-design.md`.
- **Périmètre = validation locale hors-ligne.** Ne pas modifier `router_service/router.py`, `router_service/server.py`, ni `derisk-deleg` (intégration = chantier suivant). Tout le nouveau code va dans `gen_corpus_qa/`.
- **Ne jamais toucher le service prod `slm-llama_cpp` (port 8998, adaptateur routeur `--lora`).** L'évaluation de réponse lance un llama-server de **test sur le port 8996** avec le même modèle+adaptateur.
- Classes de sortie du classifieur : exactement `"wiki"`, `"gog"`, `"secretarius"`, `"null"`.
- **Priorité des classes :** une vraie commande l'emporte sur `secretarius` — `secretarius` ne gagne que s'il est au-dessus d'un seuil dédié ET strictement supérieur aux similarités wiki et gog.
- Venv des tests/harnais nécessitant BGE-M3 : `/usr/bin/python3` (transformers 4.57.3 + pytest 9 déjà présents ; le service routeur l'utilise déjà). Depuis `~/Secretarius`, lancer avec `PYTHONPATH=.` pour importer `router_service`.
- Réutiliser `gen_corpus_qa/eval_qa.py` (`set_lora_scale`, `infer_llama`, `judge_score`, `SYSTEM_PROMPT_QA`) — ne pas dupliquer.
- Binaire llama-server : `~/llama.cpp/build-rocm/bin/llama-server`, avec `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- Teacher/juge = DeepSeek (`DEEPSEEK_API_KEY` dans l'environnement).
- Langue de tout contenu produit : français.

---

### Task 1: Document Secretarius unifié

**Files:**
- Create: `gen_corpus_qa/documents/secretarius.md`
- Test: `gen_corpus_qa/tests/test_secretarius_doc.py`

**Interfaces:**
- Produit : le fichier `secretarius.md`, concaténation des 3 documents seed, consommé par les Tasks 4 et 6.

- [ ] **Step 1: Écrire le test**

Créer `gen_corpus_qa/tests/test_secretarius_doc.py` :

```python
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "documents" / "secretarius.md"


def test_document_unifie_contient_les_trois_sections():
    txt = DOC.read_text(encoding="utf-8")
    assert "sanroque" in txt          # config matériel
    assert "/wikistatus" in txt       # capacités wiki
    assert "/chercher" in txt         # capacités gog


def test_document_reste_petit():
    # borne large en mots (~617 tokens mesurés ≈ < 900 mots)
    txt = DOC.read_text(encoding="utf-8")
    assert 0 < len(txt.split()) < 900
```

- [ ] **Step 2: Lancer le test, vérifier l'échec**

Run : `cd ~/Secretarius && /usr/bin/python3 -m pytest gen_corpus_qa/tests/test_secretarius_doc.py -q`
Expected : FAIL (fichier `secretarius.md` absent).

- [ ] **Step 3: Créer le document unifié**

Concaténer les trois documents seed existants dans `gen_corpus_qa/documents/secretarius.md`, séparés par une ligne vide, en préservant leur contenu exact :

```bash
cd ~/Secretarius/gen_corpus_qa/documents
{ cat config-materiel-logiciel.md; echo; echo; cat capacites-wiki.md; echo; echo; cat capacites-gog.md; } > secretarius.md
```

- [ ] **Step 4: Lancer le test, vérifier le succès**

Run : `cd ~/Secretarius && /usr/bin/python3 -m pytest gen_corpus_qa/tests/test_secretarius_doc.py -q`
Expected : PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/documents/secretarius.md gen_corpus_qa/tests/test_secretarius_doc.py
git commit -m "feat(secretarius): document unifié config+wiki+gog + test"
```

---

### Task 2: Jeu de données étiqueté (centroïde + test)

**Files:**
- Create: `gen_corpus_qa/labeled_data.py`
- Test: `gen_corpus_qa/tests/test_labeled_data.py`

**Interfaces:**
- Consomme : `gen_corpus_qa/corpus_qa.jsonl` (champ `question` → étiquette `secretarius`), `gen_corpus/corpus.jsonl` (champs `text`, `intention`, `variante`).
- Produit : `build_labeled_data(n_centroid=60, seed=42) -> dict` avec deux clés : `"centroid"` = `list[str]` (questions pour calculer le centroïde), `"test"` = `list[tuple[str, str]]` (message, label) où label ∈ {wiki, gog, secretarius, null}. Les questions du centroïde et celles étiquetées `secretarius` dans `test` sont disjointes. Consommé par les Tasks 3 et 6.

- [ ] **Step 1: Écrire le test**

Créer `gen_corpus_qa/tests/test_labeled_data.py` :

```python
from labeled_data import build_labeled_data


def test_centroid_et_test_secretarius_disjoints():
    d = build_labeled_data(n_centroid=60, seed=42)
    cent = set(d["centroid"])
    test_sec = {t for t, lab in d["test"] if lab == "secretarius"}
    assert cent, "centroïde vide"
    assert test_sec, "pas d'exemple secretarius de test"
    assert cent.isdisjoint(test_sec)


def test_toutes_les_classes_presentes():
    d = build_labeled_data(n_centroid=60, seed=42)
    labels = {lab for _, lab in d["test"]}
    assert labels == {"wiki", "gog", "secretarius", "null"}


def test_determinisme_par_seed():
    a = build_labeled_data(n_centroid=60, seed=42)
    b = build_labeled_data(n_centroid=60, seed=42)
    assert a["centroid"] == b["centroid"]
    assert a["test"] == b["test"]
```

- [ ] **Step 2: Lancer le test, vérifier l'échec**

Run : `cd ~/Secretarius/gen_corpus_qa && /usr/bin/python3 -m pytest tests/test_labeled_data.py -q`
Expected : FAIL (`ModuleNotFoundError: labeled_data`).

- [ ] **Step 3: Écrire l'implémentation**

Créer `gen_corpus_qa/labeled_data.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construit un jeu étiqueté {wiki, gog, secretarius, null} à partir des corpus
existants, avec centroïde et test disjoints pour secretarius."""
import json
import random
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CORPUS_QA = _HERE / "corpus_qa.jsonl"                 # questions secretarius
CORPUS_ROUTEUR = _HERE.parent / "gen_corpus" / "corpus.jsonl"

WIKI_INT = {"wiki_capture", "wiki_ingest", "wiki_status", "wiki_query", "source_read"}
GOG_INT = {"gog_search", "gog_connect", "gog_inbox", "gog_reply", "gog_drive"}
NULL_VAR = {"aide_generale", "conversation_libre"}


def build_labeled_data(n_centroid=60, seed=42, n_par_classe=90, n_null=60):
    rng = random.Random(seed)

    questions = [json.loads(l)["question"] for l in open(CORPUS_QA, encoding="utf-8")
                 if l.strip()]
    rng.shuffle(questions)
    centroid = questions[:n_centroid]
    test_sec = [(q, "secretarius") for q in questions[n_centroid:n_centroid + n_par_classe]]

    wiki, gog, null = [], [], []
    for l in open(CORPUS_ROUTEUR, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        intent, var, txt = r["intention"], r.get("variante"), r["text"]
        if intent in WIKI_INT:
            wiki.append((txt, "wiki"))
        elif intent in GOG_INT:
            gog.append((txt, "gog"))
        elif intent == "out_of_scope" and var in NULL_VAR:
            null.append((txt, "null"))

    rng.shuffle(wiki)
    rng.shuffle(gog)
    rng.shuffle(null)
    test = test_sec + wiki[:n_par_classe] + gog[:n_par_classe] + null[:n_null]
    rng.shuffle(test)
    return {"centroid": centroid, "test": test}
```

- [ ] **Step 4: Lancer le test, vérifier le succès**

Run : `cd ~/Secretarius/gen_corpus_qa && /usr/bin/python3 -m pytest tests/test_labeled_data.py -q`
Expected : PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/labeled_data.py gen_corpus_qa/tests/test_labeled_data.py
git commit -m "feat(secretarius): jeu étiqueté centroïde+test (disjoints) depuis corpus existants"
```

---

### Task 3: Classifieur secretarius (4ᵉ centroïde)

**Files:**
- Create: `gen_corpus_qa/classify_secretarius.py`
- Test: `gen_corpus_qa/tests/test_classify_secretarius.py`

**Interfaces:**
- Consomme : `router_service.router.GogGate` (méthode `_embed(texts)->tensor[N,1024]` normalisé, attribut `_cmat` = centroïdes empilés `[wiki, gog, null]`), et `build_labeled_data` (Task 2) pour les questions du centroïde.
- Produit : `class SecretariusClassifier(questions_secretarius, gate=None, seuil=0.5)` avec méthode `classify(message: str) -> str` renvoyant `"wiki"|"gog"|"secretarius"|"null"`. Consommé par la Task 6.

- [ ] **Step 1: Écrire le test**

Créer `gen_corpus_qa/tests/test_classify_secretarius.py`. Le test instancie un vrai `GogGate` (charge BGE-M3, comme `test_router.py` existant) une seule fois via fixture module :

```python
import pytest
from labeled_data import build_labeled_data
from classify_secretarius import SecretariusClassifier


@pytest.fixture(scope="module")
def clf():
    data = build_labeled_data(n_centroid=60, seed=42)
    return SecretariusClassifier(data["centroid"])


def test_sortie_toujours_dans_les_quatre_classes(clf):
    for msg in ["ingère le wiki", "cherche les mails de Paul",
                "quel modèle vous anime ?", "il fait beau aujourd'hui"]:
        assert clf.classify(msg) in {"wiki", "gog", "secretarius", "null"}


def test_commande_gog_evidente_non_volee_par_secretarius(clf):
    # propriété critique : la règle de priorité protège le routage existant.
    assert clf.classify("cherche les mails de Paul cette semaine") != "secretarius"


def test_seuil_haut_desactive_secretarius():
    # règle déterministe : avec un seuil > 1 (cosinus max de vecteurs
    # normalisés = 1), la condition sim_sec >= seuil est toujours fausse,
    # donc secretarius ne gagne jamais — teste le code, pas la sémantique.
    data = build_labeled_data(n_centroid=60, seed=42)
    clf_strict = SecretariusClassifier(data["centroid"], seuil=1.01)
    for txt, _ in data["test"][:30]:
        assert clf_strict.classify(txt) != "secretarius"
```

- [ ] **Step 2: Lancer le test, vérifier l'échec**

Run : `cd ~/Secretarius/gen_corpus_qa && PYTHONPATH=.. /usr/bin/python3 -m pytest tests/test_classify_secretarius.py -q`
Expected : FAIL (`ModuleNotFoundError: classify_secretarius`).

- [ ] **Step 3: Écrire l'implémentation**

Créer `gen_corpus_qa/classify_secretarius.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classifieur à 4 classes {wiki, gog, secretarius, null} réutilisant le GogGate
BGE-M3 (centroïdes wiki/gog/null) et lui ajoutant un centroïde secretarius.
Ne modifie pas router_service/router.py."""
import torch
from router_service.router import GogGate

SEUIL_SECRETARIUS = 0.5


class SecretariusClassifier:
    def __init__(self, questions_secretarius, gate=None, seuil=SEUIL_SECRETARIUS):
        self.gate = gate if gate is not None else GogGate()
        self.seuil = seuil
        # centroïde secretarius = moyenne normalisée des embeddings des questions
        self._cent_sec = self.gate._embed(questions_secretarius).mean(0, keepdim=True)

    def classify(self, message: str) -> str:
        e = self.gate._embed([message])                       # [1,1024]
        sims = (e @ self.gate._cmat.T).squeeze(0)             # [wiki, gog, null]
        sim_wiki, sim_gog, sim_null = (float(sims[0]), float(sims[1]), float(sims[2]))
        sim_sec = float((e @ self._cent_sec.T).squeeze())
        # priorité commande : secretarius ne gagne que s'il dépasse le seuil ET
        # est strictement supérieur aux similarités wiki et gog
        if sim_sec >= self.seuil and sim_sec > sim_wiki and sim_sec > sim_gog:
            return "secretarius"
        idx = int(torch.tensor([sim_wiki, sim_gog, sim_null]).argmax())
        return ["wiki", "gog", "null"][idx]
```

- [ ] **Step 4: Lancer le test, vérifier le succès**

Run : `cd ~/Secretarius/gen_corpus_qa && PYTHONPATH=.. /usr/bin/python3 -m pytest tests/test_classify_secretarius.py -q`
Expected : PASS (3 tests). Ces tests vérifient la robustesse de sortie, la non-régression (priorité commande) et la mécanique du seuil — pas la qualité sémantique de la détection, qui est mesurée au harnais (Task 6). Si `test_commande_gog_evidente_non_volee_par_secretarius` échoue, c'est un vrai signal (règle de priorité insuffisante) à remonter au contrôleur, pas à contourner.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/classify_secretarius.py gen_corpus_qa/tests/test_classify_secretarius.py
git commit -m "feat(secretarius): classifieur 4 classes (centroïde secretarius sur GogGate)"
```

---

### Task 4: Fonction de réponse

**Files:**
- Create: `gen_corpus_qa/repondre_secretarius.py`
- Test: `gen_corpus_qa/tests/test_repondre_secretarius.py`

**Interfaces:**
- Consomme : `eval_qa.set_lora_scale(base_url, scale)`, `eval_qa.infer_llama(base_url, document, question)`, et `documents/secretarius.md` (Task 1).
- Produit : `repondre_secretarius(question, base_url="http://127.0.0.1:8996", doc_path=<secretarius.md>) -> str`. Met le scale de l'adaptateur à 0 (phi-4 nu) puis interroge avec le document injecté. Consommé par la Task 6.

- [ ] **Step 1: Écrire le test (mock, sans réseau)**

Créer `gen_corpus_qa/tests/test_repondre_secretarius.py` :

```python
import repondre_secretarius as rs


def test_met_le_scale_a_zero_et_injecte_le_document(monkeypatch):
    appels = {}

    def fake_set_scale(base_url, scale):
        appels["scale"] = scale
        appels["scale_url"] = base_url

    def fake_infer(base_url, document, question):
        appels["document"] = document
        appels["question"] = question
        appels["infer_url"] = base_url
        return "réponse simulée"

    monkeypatch.setattr(rs, "set_lora_scale", fake_set_scale)
    monkeypatch.setattr(rs, "infer_llama", fake_infer)

    out = rs.repondre_secretarius("quel modèle vous anime ?", base_url="http://x:8996")

    assert out == "réponse simulée"
    assert appels["scale"] == 0.0                 # phi-4 nu
    assert appels["scale_url"] == "http://x:8996"
    assert appels["infer_url"] == "http://x:8996"
    assert "sanroque" in appels["document"]       # document injecté
    assert appels["question"] == "quel modèle vous anime ?"
```

- [ ] **Step 2: Lancer le test, vérifier l'échec**

Run : `cd ~/Secretarius/gen_corpus_qa && /usr/bin/python3 -m pytest tests/test_repondre_secretarius.py -q`
Expected : FAIL (`ModuleNotFoundError: repondre_secretarius`).

- [ ] **Step 3: Écrire l'implémentation**

Créer `gen_corpus_qa/repondre_secretarius.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Répond à une question sur Secretarius avec phi-4 nu (adaptateur à scale 0)
et le document Secretarius injecté en contexte. Cible un llama-server de TEST
(8996 par défaut), jamais la prod 8998."""
from pathlib import Path
from eval_qa import set_lora_scale, infer_llama

_DOC = Path(__file__).resolve().parent / "documents" / "secretarius.md"
TEST_BASE_URL = "http://127.0.0.1:8996"


def repondre_secretarius(question: str, base_url: str = TEST_BASE_URL,
                         doc_path: Path = _DOC) -> str:
    document = Path(doc_path).read_text(encoding="utf-8")
    set_lora_scale(base_url, 0.0)   # désactive l'adaptateur routeur -> phi-4 nu
    return infer_llama(base_url, document, question)
```

- [ ] **Step 4: Lancer le test, vérifier le succès**

Run : `cd ~/Secretarius/gen_corpus_qa && /usr/bin/python3 -m pytest tests/test_repondre_secretarius.py -q`
Expected : PASS (1 test).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/repondre_secretarius.py gen_corpus_qa/tests/test_repondre_secretarius.py
git commit -m "feat(secretarius): réponse phi-4 nu (scale 0) + document injecté, serveur de test"
```

---

### Task 5: Métriques du harnais (matrice de confusion)

**Files:**
- Create: `gen_corpus_qa/mesure_secretarius.py`
- Test: `gen_corpus_qa/tests/test_mesure_secretarius.py`

**Interfaces:**
- Produit : `confusion_matrix(pairs: list[tuple[str,str]]) -> dict[str, dict[str,int]]` (clé externe = vrai label, interne = prédiction) et `taux_commandes_volees(m: dict) -> float` (fraction des vraies commandes wiki+gog classées `secretarius`) et `rappel(m: dict, label: str) -> float`. Fonctions pures, testables sans BGE-M3. Consommé par la Task 6.

- [ ] **Step 1: Écrire le test**

Créer `gen_corpus_qa/tests/test_mesure_secretarius.py` :

```python
from mesure_secretarius import confusion_matrix, taux_commandes_volees, rappel


def test_confusion_matrix_compte():
    pairs = [("wiki", "wiki"), ("wiki", "secretarius"),
             ("gog", "gog"), ("secretarius", "secretarius"),
             ("null", "wiki")]
    m = confusion_matrix(pairs)
    assert m["wiki"]["wiki"] == 1
    assert m["wiki"]["secretarius"] == 1
    assert m["gog"]["gog"] == 1
    assert m["null"]["wiki"] == 1


def test_taux_commandes_volees():
    # 1 wiki volé + 0 gog volé sur 2 commandes = 0.5
    pairs = [("wiki", "secretarius"), ("gog", "gog")]
    assert taux_commandes_volees(confusion_matrix(pairs)) == 0.5


def test_rappel():
    pairs = [("secretarius", "secretarius"), ("secretarius", "null")]
    assert rappel(confusion_matrix(pairs), "secretarius") == 0.5


def test_matrices_vides_ne_plantent_pas():
    m = confusion_matrix([])
    assert taux_commandes_volees(m) == 0.0
    assert rappel(m, "secretarius") == 0.0
```

- [ ] **Step 2: Lancer le test, vérifier l'échec**

Run : `cd ~/Secretarius/gen_corpus_qa && /usr/bin/python3 -m pytest tests/test_mesure_secretarius.py -q`
Expected : FAIL (`ModuleNotFoundError: mesure_secretarius`).

- [ ] **Step 3: Écrire l'implémentation**

Créer `gen_corpus_qa/mesure_secretarius.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Métriques du harnais de validation secretarius (fonctions pures)."""

LABELS = ["wiki", "gog", "secretarius", "null"]


def confusion_matrix(pairs):
    m = {t: {p: 0 for p in LABELS} for t in LABELS}
    for vrai, pred in pairs:
        m[vrai][pred] += 1
    return m


def taux_commandes_volees(m):
    total = sum(m["wiki"].values()) + sum(m["gog"].values())
    if total == 0:
        return 0.0
    volees = m["wiki"]["secretarius"] + m["gog"]["secretarius"]
    return volees / total


def rappel(m, label):
    total = sum(m[label].values())
    if total == 0:
        return 0.0
    return m[label][label] / total
```

- [ ] **Step 4: Lancer le test, vérifier le succès**

Run : `cd ~/Secretarius/gen_corpus_qa && /usr/bin/python3 -m pytest tests/test_mesure_secretarius.py -q`
Expected : PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/mesure_secretarius.py gen_corpus_qa/tests/test_mesure_secretarius.py
git commit -m "feat(secretarius): métriques harnais (confusion, commandes volées, rappel)"
```

---

### Task 6: Exécution bout-en-bout + verdict

**Files:**
- Create: `gen_corpus_qa/run_mesure_secretarius.py`
- Create: `gen_corpus_qa/RESULTATS_SECRETARIUS.md` (produit par l'exécution)

**Interfaces:**
- Consomme : `build_labeled_data` (Task 2), `SecretariusClassifier` (Task 3), `repondre_secretarius` (Task 4), `confusion_matrix`/`taux_commandes_volees`/`rappel` (Task 5), `eval_qa.judge_score` (juge DeepSeek).
- Produit : le fichier verdict `RESULTATS_SECRETARIUS.md`.

Cette tâche est une **exécution** (pas du code unitaire nouveau) avec un point de
décision. Elle nécessite BGE-M3 (CPU), un llama-server de **test** sur 8996, et
`DEEPSEEK_API_KEY`.

- [ ] **Step 1: Écrire le script d'exécution**

Créer `gen_corpus_qa/run_mesure_secretarius.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harnais de validation locale : matrice de confusion du classifieur +
qualité des réponses phi-4 nu. Écrit RESULTATS_SECRETARIUS.md."""
import random
from pathlib import Path

from labeled_data import build_labeled_data
from classify_secretarius import SecretariusClassifier
from repondre_secretarius import repondre_secretarius
from mesure_secretarius import confusion_matrix, taux_commandes_volees, rappel
from eval_qa import judge_score

TEST_BASE_URL = "http://127.0.0.1:8996"
N_ECHANTILLON_REPONSE = 20


def main():
    data = build_labeled_data(n_centroid=60, seed=42)
    clf = SecretariusClassifier(data["centroid"])

    # 1) Classification du jeu de test complet
    pairs = [(lab, clf.classify(txt)) for txt, lab in data["test"]]
    m = confusion_matrix(pairs)
    voles = taux_commandes_volees(m)
    rap_sec = rappel(m, "secretarius")

    # 2) Qualité des réponses phi-4 nu sur un échantillon secretarius
    sec_txts = [txt for txt, lab in data["test"] if lab == "secretarius"]
    rng = random.Random(42)
    rng.shuffle(sec_txts)
    ech = sec_txts[:N_ECHANTILLON_REPONSE]
    doc = (Path(__file__).resolve().parent / "documents" / "secretarius.md").read_text(encoding="utf-8")
    notes, apercus = [], []
    for q in ech:
        rep = repondre_secretarius(q, base_url=TEST_BASE_URL)
        note = judge_score(doc, q, rep) / 5.0
        notes.append(note)
        apercus.append((q, rep, note))
    note_moy = sum(notes) / len(notes) if notes else 0.0

    # 3) Écriture du verdict
    lignes = ["# Verdict — détection & réponse question-Secretarius", "",
              "## Détection (classifieur centroïde)", "",
              "| vrai \\ prédit | wiki | gog | secretarius | null |",
              "|---|---|---|---|---|"]
    for t in ["wiki", "gog", "secretarius", "null"]:
        r = m[t]
        lignes.append(f"| {t} | {r['wiki']} | {r['gog']} | {r['secretarius']} | {r['null']} |")
    lignes += ["",
               f"- Rappel secretarius : **{rap_sec:.3f}**",
               f"- Taux de commandes wiki/gog détournées vers secretarius : **{voles:.3f}**",
               "",
               "## Réponse (phi-4 nu + document)", "",
               f"- Note moyenne juge DeepSeek sur {len(ech)} questions : **{note_moy:.3f}**",
               "", "### Aperçus", ""]
    for q, rep, note in apercus:
        lignes.append(f"- [{note:.1f}] Q: {q!r}\n  R: {rep[:200]!r}")
    Path(__file__).resolve().parent.joinpath("RESULTATS_SECRETARIUS.md").write_text(
        "\n".join(lignes), encoding="utf-8")
    print(f"rappel_sec={rap_sec:.3f} voles={voles:.3f} note_rep={note_moy:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lancer un llama-server de TEST sur 8996 (jamais 8998)**

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 nohup ~/llama.cpp/build-rocm/bin/llama-server \
  -m /home/mauceric/Modèles/Phi-4-mini-instruct-Q6_K.gguf \
  --lora /home/mauceric/lora_slm/tiron-unified-lora-f16.gguf \
  --host 127.0.0.1 --port 8996 -c 2048 -ngl 99 \
  > /tmp/llama_test_8996.log 2>&1 &
```

Attendre le démarrage (`until curl -s http://127.0.0.1:8996/health | grep -q ok; do sleep 2; done`).
Vérifier que le service prod 8998 est **toujours actif** : `systemctl --user is-active slm-llama_cpp.service`.
Si le GPU sature (contention avec la prod, cf. incidents Task 7 du chantier QA), **repli** : mesurer la réponse en PEFT/CPU (le classifieur, lui, tourne en CPU sans souci). La partie détection ne dépend pas de ce serveur.

- [ ] **Step 3: Exécuter le harnais**

```bash
cd ~/Secretarius/gen_corpus_qa
set -a; source ~/.config/secrets.env; set +a
PYTHONPATH=.. /usr/bin/python3 run_mesure_secretarius.py
```

Expected : affiche `rappel_sec=… voles=… note_rep=…` et écrit `RESULTATS_SECRETARIUS.md`.

- [ ] **Step 4: Arrêter le serveur de test**

```bash
pkill -f "port 8996"   # ne touche pas 8998
```

- [ ] **Step 5: Verdict et décision**

Lire `RESULTATS_SECRETARIUS.md`. Critère (spec) :
- **Détection OK** si `voles` est quasi nul (seuil indicatif < ~0.03) ET `rappel_sec` élevé.
- **Réponse OK** si `note_rep` proche de la référence 0.82.

Si les deux passent → feu vert pour le chantier d'intégration OpenClaw. Si la
détection échoue → documenter et escalader d'un cran (classifieur léger sur
embeddings, puis LoRA), chaque cran étant un chantier distinct. Un balayage du
`seuil` de `SecretariusClassifier` peut être fait ici avant de conclure à l'échec.

- [ ] **Step 6: Commit du verdict**

```bash
cd ~/Secretarius
git add gen_corpus_qa/run_mesure_secretarius.py gen_corpus_qa/RESULTATS_SECRETARIUS.md
git commit -m "feat(secretarius): harnais de mesure + verdict de validation locale"
```

---

## Notes de risque (rappel du spec)

1. **Frontière commande/question** — « comment interroger le wiki ? » vs « /q … ». Révélée par `voles` dans la matrice de confusion.
2. **Représentativité du centroïde** — exemples issus du corpus QA (DeepSeek), pas de vrais messages Telegram.
3. **Contamination du `null`** — vérifier dans la matrice que les questions secretarius ne partent pas en `null`.
4. **Contention GPU** — le serveur de test 8996 partage l'iGPU avec la prod 8998 ; repli PEFT/CPU pour la réponse si saturation.
