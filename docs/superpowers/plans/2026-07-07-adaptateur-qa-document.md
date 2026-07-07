# Adaptateur LoRA « QA-sur-document » générique — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produire et valider (A/B contre le modèle nu) un adaptateur LoRA phi-4-mini unique qui répond à une question en s'appuyant strictement sur un document fourni en contexte, avec refus propre hors-document.

**Architecture:** On généralise le pipeline `gen_corpus/` (routeur d'intentions) dans un nouveau répertoire `gen_corpus_qa/`. Un seed de documents + exemples amorces alimente une génération DeepSeek (via GEPA) de triplets (document, question, réponse). Ces triplets sont convertis en ChatML, servent à entraîner un adaptateur LoRA (recette éprouvée `lora_slm/lora_train.py`), puis un harnais A/B compare phi-4 nu vs phi-4+adaptateur via un juge DeepSeek, sur deux backends (llama.cpp GGUF et PEFT/Jupyter).

**Tech Stack:** Python 3 (dspy, torch+ROCm/peft/transformers), DeepSeek API (génération + juge), llama.cpp `build-rocm/bin/llama-server`, Jupyter.

## Global Constraints

- Spec de référence : `docs/superpowers/specs/2026-07-07-adaptateur-qa-document-design.md`.
- **Ne pas toucher `gen_corpus/`** (pipeline routeur en production) — tout le nouveau code va dans `gen_corpus_qa/`.
- **Ne pas toucher le service `slm-llama_cpp.service`** (port 8998, sert l'adaptateur routeur en prod). L'évaluation GGUF lance un llama-server de test sur un **port dédié 8996**.
- Teacher de génération ET juge d'évaluation = **DeepSeek** (`DEEPSEEK_API_KEY` requis dans l'environnement ; `model=openai/deepseek-chat`, `api_base=https://api.deepseek.com`).
- Binaire llama-server : `~/llama.cpp/build-rocm/bin/llama-server` (jamais `build/bin/`), avec `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- Venv génération/tests dspy : réutiliser `gen_corpus/.venv` (mêmes dépendances dspy+pytest ; pas de duplication).
- Venv entraînement/PEFT : `lora_slm/lenv/bin/python` (torch ROCm, peft, transformers).
- Hyperparamètres LoRA de départ (éprouvés sur le routeur) : `--lora_r 16 --lora_alpha 32 --lr 2e-4 --epochs 6`, mais `--max_len 2048` (le document en contexte allonge le prompt, contrairement au routeur).
- Checkpoints et GGUF hors dépôt : `/home/mauceric/lora_slm/checkpoints/qa-document/` et `/home/mauceric/lora_slm/qa-document-lora-f16.gguf`.
- Langue de tout le contenu généré : français.

---

### Task 1: Structure `gen_corpus_qa/`, documents seed, taxonomie, exemples amorces

**Files:**
- Create: `gen_corpus_qa/documents/config-materiel-logiciel.md`
- Create: `gen_corpus_qa/documents/capacites-wiki.md`
- Create: `gen_corpus_qa/documents/capacites-gog.md`
- Create: `gen_corpus_qa/domaines.json`
- Create: `gen_corpus_qa/registres.json`
- Create: `gen_corpus_qa/seed.json`
- Create: `gen_corpus_qa/prompt-init.txt`
- Test: `gen_corpus_qa/tests/test_seed_valide.py`

**Interfaces:**
- Produit : `domaines.json` (liste d'objets `{domaine, document, types_question}`), `seed.json` (liste d'objets `{document_id, type_question, registre, question, answer}`), consommés par les Tasks 2 et 4.

- [ ] **Step 1: Créer l'arborescence**

```bash
mkdir -p ~/Secretarius/gen_corpus_qa/documents ~/Secretarius/gen_corpus_qa/tests
```

- [ ] **Step 2: Rédiger le document `config-materiel-logiciel.md`**

Écrire dans `gen_corpus_qa/documents/config-materiel-logiciel.md` :

```markdown
# Configuration matérielle et logicielle de Secretarius (machine sanroque)

## Matériel
- Machine : sanroque, ordinateur portable.
- Processeur : AMD Ryzen 9 6900HX.
- Carte graphique intégrée (iGPU) : AMD Radeon 680M (architecture RDNA2, identifiant gfx1035).
- Mémoire vive : 30 Go, partagée entre le processeur et l'iGPU.

## Modèle de langage
- Le modèle qui anime Tiron est Phi-4-mini-instruct, quantifié en Q6_K, augmenté d'adaptateurs LoRA spécialisés.
- L'extraction d'expressions du wiki utilise un second modèle : Phi-4-mini affiné sur Wikipédia en français.

## Services actifs (systemd utilisateur)
- slm-llama_cpp : serveur llama.cpp sur le port 8998, sert Phi-4-mini + l'adaptateur de routage, accéléré par ROCm.
- tiron-router : service de routage sur le port 8999, classe le message et sélectionne la commande.
- openclaw-gateway : passerelle reliée à Telegram, exécute Tiron.
- llama.cpp extracteur : port 8989, modèle Wikipédia FR.
```

- [ ] **Step 3: Rédiger le document `capacites-wiki.md`**

Écrire dans `gen_corpus_qa/documents/capacites-wiki.md` :

```markdown
# Capacités wiki de Secretarius

Le wiki (Wiki_LM) est la base de connaissances personnelle de l'utilisateur, stockée en fichiers Markdown (coffre Obsidian).

## Commandes
- /c <url|note> : capturer une page web ou une note dans le wiki.
- /ingest : lancer le traitement des captures en attente (opération asynchrone).
- /q <question> : interroger la base de connaissances et obtenir une synthèse.
- /source <url> : lire immédiatement une page web externe via l'agent Scout (protection anti-injection), sans la sauvegarder.
- /wikistatus : afficher l'état de l'ingestion du wiki.

## Fonctionnement
Les captures passent d'abord dans une file, puis l'ingestion extrait les expressions, calcule des plongements (embeddings) et met à jour la base interrogeable par /q.
```

- [ ] **Step 4: Rédiger le document `capacites-gog.md`**

Écrire dans `gen_corpus_qa/documents/capacites-gog.md` :

```markdown
# Capacités Google (gog) de Secretarius

L'agent gog donne accès au compte Google de l'utilisateur : messagerie Gmail et fichiers Google Drive.

## Commandes
- /chercher <critères> : rechercher des emails dans Gmail (par mot-clé, expéditeur ou période).
- /inbox : lister les emails récents de la boîte de réception.
- /repondre <contexte> : préparer un brouillon de réponse à un email.
- /drive <critères> : rechercher des fichiers sur Google Drive.
- /connecter : autoriser l'accès au compte Google.

## Sécurité
Aucun email n'est envoyé automatiquement : /repondre prépare seulement un brouillon, qui n'est expédié qu'après confirmation explicite par la commande /confirm.
```

- [ ] **Step 5: Créer `domaines.json` et `registres.json`**

`gen_corpus_qa/domaines.json` :

```json
[
  {"domaine": "config", "document": "documents/config-materiel-logiciel.md",
   "types_question": ["factuelle", "reformulation", "hors_document"]},
  {"domaine": "wiki", "document": "documents/capacites-wiki.md",
   "types_question": ["factuelle", "reformulation", "hors_document"]},
  {"domaine": "gog", "document": "documents/capacites-gog.md",
   "types_question": ["factuelle", "reformulation", "hors_document"]}
]
```

`gen_corpus_qa/registres.json` :

```json
["formel", "familier", "télégraphique", "poli", "abrégé"]
```

- [ ] **Step 6: Créer `seed.json` (exemples amorces, dont négatifs)**

Écrire dans `gen_corpus_qa/seed.json` le contenu ci-dessous (9 exemples : 3 par domaine, dont 1 `hors_document` chacun). **Puis compléter jusqu'à ~30 exemples** en suivant exactement ce schéma (varier questions, registres, en s'appuyant sur le texte des trois documents ; garder au moins un tiers d'exemples `hors_document` dont la réponse est un refus) :

```json
[
  {"document_id": "config", "type_question": "factuelle", "registre": "poli",
   "question": "Quel modèle de langage vous anime ?",
   "answer": "Je suis animé par Phi-4-mini-instruct, quantifié en Q6_K, augmenté d'adaptateurs LoRA spécialisés."},
  {"document_id": "config", "type_question": "reformulation", "registre": "familier",
   "question": "C'est quoi ta carte graphique au fait ?",
   "answer": "L'iGPU est une AMD Radeon 680M (architecture RDNA2, identifiant gfx1035)."},
  {"document_id": "config", "type_question": "hors_document", "registre": "poli",
   "question": "Quelle est la météo à Paris aujourd'hui ?",
   "answer": "Cette information ne figure pas dans le document."},
  {"document_id": "wiki", "type_question": "factuelle", "registre": "télégraphique",
   "question": "commande pour capturer une url ?",
   "answer": "La commande /c permet de capturer une page web ou une note dans le wiki."},
  {"document_id": "wiki", "type_question": "reformulation", "registre": "poli",
   "question": "Comment puis-je poser une question à ma base de connaissances ?",
   "answer": "Utilisez la commande /q suivie de votre question pour interroger la base et obtenir une synthèse."},
  {"document_id": "wiki", "type_question": "hors_document", "registre": "formel",
   "question": "Combien de pages contient actuellement le wiki ?",
   "answer": "Le document ne précise pas le nombre de pages du wiki."},
  {"document_id": "gog", "type_question": "factuelle", "registre": "poli",
   "question": "Comment répondre à un email ?",
   "answer": "La commande /repondre prépare un brouillon de réponse ; il n'est envoyé qu'après confirmation par /confirm."},
  {"document_id": "gog", "type_question": "reformulation", "registre": "abrégé",
   "question": "chercher fichier drive comment",
   "answer": "Utilisez /drive suivi de vos critères pour rechercher des fichiers sur Google Drive."},
  {"document_id": "gog", "type_question": "hors_document", "registre": "familier",
   "question": "Tu peux m'envoyer un mail à ma mère directement ?",
   "answer": "Le document n'indique pas de moyen d'envoyer un email directement ; /repondre ne fait que préparer un brouillon soumis à confirmation."}
]
```

- [ ] **Step 7: Créer `prompt-init.txt` (prompt de génération de départ, avant GEPA)**

Écrire dans `gen_corpus_qa/prompt-init.txt` :

```text
À partir du DOCUMENT fourni, générez une paire (question, réponse) réaliste en français.

Règles :
- La question doit être formulée dans le registre indiqué (formel / familier / télégraphique / poli / abrégé).
- Si le type de question est "factuelle" : la question porte sur une information explicitement présente dans le document, et la réponse est exacte, concise, entièrement fondée sur le document.
- Si le type de question est "reformulation" : même chose, mais la question emploie d'autres mots que le document (synonymes, tournure indirecte).
- Si le type de question est "hors_document" : la question est plausible mais sa réponse NE figure PAS dans le document ; la réponse doit indiquer clairement que l'information n'est pas dans le document, sans rien inventer.
- La réponse est toujours en français, concise (1 à 3 phrases), et ne contient aucune information absente du document.
```

- [ ] **Step 8: Écrire le test de validation du seed et de la taxonomie**

Écrire dans `gen_corpus_qa/tests/test_seed_valide.py` :

```python
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _load(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_domaines_pointent_des_documents_existants():
    for d in _load("domaines.json"):
        assert (BASE / d["document"]).exists(), f"document manquant: {d['document']}"
        assert d["types_question"], "types_question vide"


def test_seed_schema_et_ids_connus():
    domaines = {d["domaine"] for d in _load("domaines.json")}
    seed = _load("seed.json")
    assert len(seed) >= 9
    for ex in seed:
        assert set(ex) == {"document_id", "type_question", "registre", "question", "answer"}
        assert ex["document_id"] in domaines
        assert ex["question"].strip() and ex["answer"].strip()


def test_seed_contient_des_exemples_negatifs():
    seed = _load("seed.json")
    negatifs = [e for e in seed if e["type_question"] == "hors_document"]
    assert len(negatifs) >= 3, "il faut des exemples hors_document (refus)"
```

- [ ] **Step 9: Lancer le test**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_seed_valide.py -v`
Expected: 3 tests PASS.

- [ ] **Step 10: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/documents gen_corpus_qa/domaines.json gen_corpus_qa/registres.json gen_corpus_qa/seed.json gen_corpus_qa/prompt-init.txt gen_corpus_qa/tests/test_seed_valide.py
git commit -m "feat(qa): documents seed, taxonomie de domaines et exemples amorces QA"
```

---

### Task 2: `generate_corpus_qa.py` — génération DeepSeek des triplets

**Files:**
- Create: `gen_corpus_qa/generate_corpus_qa.py`
- Test: `gen_corpus_qa/tests/test_generate_qa.py`

**Interfaces:**
- Consomme : `domaines.json`, `registres.json`, `prompt-init.txt` (ou `GEPAPrompt.txt`), documents seed.
- Produit : `corpus_qa.jsonl` (lignes JSON `{document_id, document, question, answer, type_question, registre}`), consommé par la Task 3. Fonction pure testable `build_entry(result, document_id, document, type_question, registre) -> dict`.

- [ ] **Step 1: Écrire le test (fonction pure, sans appel réseau)**

Écrire dans `gen_corpus_qa/tests/test_generate_qa.py` :

```python
import sys, types
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from generate_corpus_qa import build_entry


class FakeResult:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


def test_build_entry_structure():
    r = FakeResult("  Quel modèle ? ", "  Phi-4-mini.  ")
    e = build_entry(r, "config", "DOC-TEXTE", "factuelle", "poli")
    assert e == {
        "document_id": "config",
        "document": "DOC-TEXTE",
        "question": "Quel modèle ?",
        "answer": "Phi-4-mini.",
        "type_question": "factuelle",
        "registre": "poli",
    }
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_generate_qa.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'generate_corpus_qa'`).

- [ ] **Step 3: Écrire `generate_corpus_qa.py`**

Écrire dans `gen_corpus_qa/generate_corpus_qa.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère corpus_qa.jsonl (triplets document/question/réponse) via DeepSeek.

Généralisation de gen_corpus/generate_corpus.py : pour chaque domaine, charge
le document associé et demande au modèle une paire (question, réponse ancrée)
selon un type de question et un registre tirés au hasard.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy.clients import configure_cache as dspy_configure_cache

try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None


@dataclass
class Config:
    count: int = 1500
    batch_size: int = 50
    report_every: int = 50
    prompt_path: str = "GEPAPrompt.txt"
    prompt_fallback: str = "prompt-init.txt"
    domaines_path: str = "domaines.json"
    registres_path: str = "registres.json"
    output: str = "corpus_qa.jsonl"
    generator_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    temperature: float = 0.9


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--prompt", default="GEPAPrompt.txt")
    p.add_argument("--domaines", default="domaines.json")
    p.add_argument("--registres", default="registres.json")
    p.add_argument("--output", default="corpus_qa.jsonl")
    p.add_argument("--model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=0.9)
    a = p.parse_args(argv)
    return Config(count=a.count, batch_size=a.batch_size, report_every=a.report_every,
                  prompt_path=a.prompt, domaines_path=a.domaines, registres_path=a.registres,
                  output=a.output, generator_model=a.model,
                  deepseek_api_base=a.deepseek_api_base, temperature=a.temperature)


def _build_signature(prompt_text: str):
    class GenerateQA(dspy.Signature):
        __doc__ = prompt_text
        document:      str = dspy.InputField(desc="Texte du document de référence")
        type_question: str = dspy.InputField(desc="factuelle, reformulation ou hors_document")
        registre:      str = dspy.InputField(desc="Registre de la question")
        question: str = dspy.OutputField(desc="Question utilisateur en français")
        answer:   str = dspy.OutputField(desc="Réponse ancrée dans le document, ou refus si hors_document")
    return GenerateQA


def build_entry(result, document_id: str, document: str, type_question: str, registre: str) -> dict:
    return {"document_id": document_id, "document": document,
            "question": result.question.strip(), "answer": result.answer.strip(),
            "type_question": type_question, "registre": registre}


def _load_documents(domaines: list[dict], base: Path) -> dict:
    return {d["domaine"]: (base / d["document"]).read_text(encoding="utf-8") for d in domaines}


def main(argv=None) -> int:
    cfg = parse_args(argv)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    base_url = os.getenv("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
    lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base_url,
                 model_type="chat", temperature=cfg.temperature, max_tokens=512, cache=False)
    dspy.settings.configure(lm=lm)

    here = Path(cfg.domaines_path).resolve().parent
    prompt_p = Path(cfg.prompt_path)
    prompt_text = (prompt_p if prompt_p.exists() else Path(cfg.prompt_fallback)).read_text(encoding="utf-8")
    predict = dspy.Predict(_build_signature(prompt_text))
    domaines = json.loads(Path(cfg.domaines_path).read_text(encoding="utf-8"))
    registres = json.loads(Path(cfg.registres_path).read_text(encoding="utf-8"))
    documents = _load_documents(domaines, here)

    buffer = []
    stime = time.time()
    consecutive_errors = 0
    max_consecutive = max(10, cfg.count // 10)
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in range(cfg.count):
            dom = random.choice(domaines)
            doc_text = documents[dom["domaine"]]
            tq = random.choice(dom["types_question"])
            reg = random.choice(registres)
            try:
                result = predict(document=doc_text, type_question=tq, registre=reg)
                buffer.append(build_entry(result, dom["domaine"], doc_text, tq, reg))
                consecutive_errors = 0
            except Exception as e:
                print(f"[{i+1}] Erreur: {e}", flush=True)
                consecutive_errors += 1
                if consecutive_errors > max_consecutive:
                    raise RuntimeError(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt.") from e
                continue
            if len(buffer) >= cfg.batch_size:
                for e in buffer:
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                buffer = []
            if (i + 1) % cfg.report_every == 0:
                print(f"[{i+1}/{cfg.count}] {time.time()-stime:.1f}s", flush=True)
        for e in buffer:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Corpus sauvegardé dans {cfg.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_generate_qa.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/generate_corpus_qa.py gen_corpus_qa/tests/test_generate_qa.py
git commit -m "feat(qa): générateur de corpus QA via DeepSeek (build_entry testé)"
```

---

### Task 3: `to_lora_format_qa.py` — conversion en ChatML

**Files:**
- Create: `gen_corpus_qa/to_lora_format_qa.py`
- Test: `gen_corpus_qa/tests/test_to_lora_qa.py`

**Interfaces:**
- Consomme : `corpus_qa.jsonl` (Task 2).
- Produit : `corpus_qa_train.jsonl` / `corpus_qa_eval.jsonl` (ChatML, colonne `messages`), consommés par les Tasks 5 et 7. Constante `SYSTEM_PROMPT_QA` (str) et fonction `convert_entry_qa(entry) -> dict` réutilisées par la Task 5.

- [ ] **Step 1: Écrire le test**

Écrire dans `gen_corpus_qa/tests/test_to_lora_qa.py` :

```python
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from to_lora_format_qa import convert_entry_qa, SYSTEM_PROMPT_QA


def test_convert_entry_qa_chatml():
    entry = {"document_id": "config", "document": "DOC", "question": "Q ?",
             "answer": "R.", "type_question": "factuelle", "registre": "poli"}
    out = convert_entry_qa(entry)
    msgs = out["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT_QA}
    assert msgs[1] == {"role": "user", "content": "Document:\nDOC\n\nQuestion: Q ?"}
    assert msgs[2] == {"role": "assistant", "content": "R."}
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_to_lora_qa.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Écrire `to_lora_format_qa.py`**

Écrire dans `gen_corpus_qa/to_lora_format_qa.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convertit corpus_qa.jsonl en ChatML pour fine-tuning LoRA (phi-4-mini)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT_QA = (
    "Vous êtes Tiron. Répondez à la question en vous appuyant uniquement sur le "
    "document fourni. Soyez concis et répondez en français. Si la réponse ne figure "
    "pas dans le document, indiquez-le clairement sans rien inventer."
)


def convert_entry_qa(entry: dict) -> dict:
    user = f"Document:\n{entry['document']}\n\nQuestion: {entry['question']}"
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT_QA},
        {"role": "user", "content": user},
        {"role": "assistant", "content": entry["answer"]},
    ]}


def to_lora(corpus_path: str, train_path: str, eval_path: str,
            eval_ratio: float = 0.1, seed: int = 42) -> None:
    lines = [l for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    converted = [convert_entry_qa(json.loads(l)) for l in lines]
    random.seed(seed)
    random.shuffle(converted)
    n_eval = max(1, int(len(converted) * eval_ratio))
    for path, data in [(train_path, converted[n_eval:]), (eval_path, converted[:n_eval])]:
        Path(path).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in data), encoding="utf-8"
        )
    print(f"Total: {len(converted)} | Train: {len(converted)-n_eval} | Eval: {n_eval}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus_qa.jsonl")
    p.add_argument("--train", default="corpus_qa_train.jsonl")
    p.add_argument("--eval", default="corpus_qa_eval.jsonl")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    a = p.parse_args()
    to_lora(a.corpus, a.train, a.eval, a.eval_ratio)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_to_lora_qa.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/to_lora_format_qa.py gen_corpus_qa/tests/test_to_lora_qa.py
git commit -m "feat(qa): conversion ChatML du corpus QA (convert_entry_qa testé)"
```

---

### Task 4: `promptGenGEPA_qa.py` — optimisation du prompt de génération

**Files:**
- Create: `gen_corpus_qa/promptGenGEPA_qa.py`
- Test: `gen_corpus_qa/tests/test_gepa_qa.py`

**Interfaces:**
- Consomme : `seed.json`, `prompt-init.txt`, documents seed.
- Produit : `GEPAPrompt.txt` (prompt optimisé), consommé par la Task 2 en génération réelle. Fonction pure testable `note_paire(judge_score, type_question, answer) -> float`.

- [ ] **Step 1: Écrire le test de la métrique (fonction pure)**

La métrique combine (a) une note de qualité 1..5 rendue par le juge DeepSeek et (b) un contrôle déterministe du refus : pour un `hors_document`, la réponse doit ressembler à un refus (contenir un marqueur du type « ne figure pas », « ne précise pas », « pas dans le document »). `note_paire` normalise et pénalise un refus manquant.

Écrire dans `gen_corpus_qa/tests/test_gepa_qa.py` :

```python
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from promptGenGEPA_qa import note_paire


def test_note_factuelle_suit_le_juge():
    assert note_paire(5, "factuelle", "Phi-4-mini.") == 1.0
    assert note_paire(3, "factuelle", "Phi-4-mini.") == 0.6


def test_hors_document_avec_refus_ok():
    assert note_paire(5, "hors_document", "Cette information ne figure pas dans le document.") == 1.0


def test_hors_document_sans_refus_penalise():
    # réponse inventée au lieu d'un refus -> plafonnée à 0.2
    assert note_paire(5, "hors_document", "Il fait 22 degrés à Paris.") == 0.2
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_gepa_qa.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Écrire `promptGenGEPA_qa.py`**

Écrire dans `gen_corpus_qa/promptGenGEPA_qa.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation GEPA du prompt de génération QA (généralise gen_corpus/promptGenGEPA.py)."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy.clients import configure_cache as dspy_configure_cache

try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None

REFUS_MARQUEURS = ("ne figure pas", "ne précise pas", "n'est pas dans", "pas dans le document",
                   "aucune information", "ne mentionne pas", "ne contient pas", "n'indique pas")


def _ressemble_refus(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUS_MARQUEURS)


def note_paire(judge_score: int, type_question: str, answer: str) -> float:
    """Note 0..1 d'une paire générée. judge_score est un entier 1..5.
    Pour hors_document, un refus est exigé : sinon la note est plafonnée à 0.2."""
    base = max(1, min(5, int(judge_score))) / 5.0
    if type_question == "hors_document" and not _ressemble_refus(answer):
        return 0.2
    return base


@dataclass
class Config:
    seed_path: str = "seed.json"
    domaines_path: str = "domaines.json"
    prompt_path: str = "prompt-init.txt"
    gepa_prompt_path: str = "GEPAPrompt.txt"
    generator_model: str = "openai/deepseek-chat"
    eval_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    reflection_temperature: float = 1.0
    max_metric_calls: int = 200


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="seed.json")
    p.add_argument("--domaines", default="domaines.json")
    p.add_argument("--prompt", default="prompt-init.txt")
    p.add_argument("--gepa-prompt", default="GEPAPrompt.txt")
    p.add_argument("--generator-model", default="openai/deepseek-chat")
    p.add_argument("--eval-model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-metric-calls", type=int, default=200)
    a = p.parse_args(argv)
    return Config(seed_path=a.seed, domaines_path=a.domaines, prompt_path=a.prompt,
                  gepa_prompt_path=a.gepa_prompt, generator_model=a.generator_model,
                  eval_model=a.eval_model, deepseek_api_base=a.deepseek_api_base,
                  reflection_temperature=a.temperature, max_metric_calls=a.max_metric_calls)


def _ensure_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    return key


def configure_lm(model, temperature, api_base) -> dspy.LM:
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    lm = dspy.LM(model=model, api_key=_ensure_key(), api_base=base, model_type="chat",
                 temperature=temperature, max_tokens=512, cache=False)
    dspy.settings.configure(lm=lm)
    return lm


def configure_eval_lm(model, api_base) -> dspy.LM:
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    return dspy.LM(model=model, api_key=_ensure_key(), api_base=base, model_type="chat",
                   temperature=0.0, max_tokens=16, cache=False)


def build_example_generator(prompt_text: str, documents: dict) -> dspy.Module:
    class GenerateQA(dspy.Signature):
        __doc__ = prompt_text
        document:      str = dspy.InputField(desc="Texte du document de référence")
        type_question: str = dspy.InputField(desc="factuelle, reformulation ou hors_document")
        registre:      str = dspy.InputField(desc="Registre de la question")
        question: str = dspy.OutputField(desc="Question utilisateur en français")
        answer:   str = dspy.OutputField(desc="Réponse ancrée, ou refus si hors_document")

    class QAGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(GenerateQA)

        def forward(self, document_id, type_question, registre):
            doc = documents[document_id]
            r = self.generate(document=doc, type_question=type_question, registre=registre)
            return {"question": r.question, "answer": r.answer,
                    "type_question": type_question, "document_id": document_id}

    return QAGenerator()


def build_trainset(seed: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(document_id=ex["document_id"], type_question=ex["type_question"],
                     registre=ex["registre"], question=ex["question"], answer=ex["answer"]
                     ).with_inputs("document_id", "type_question", "registre")
        for ex in seed
    ]


class EvalQualite(dspy.Signature):
    """La réponse est-elle exacte, concise et entièrement fondée sur le document ?
    Répondre avec un entier 1..5 uniquement, sans commentaire."""
    document: str = dspy.InputField()
    question: str = dspy.InputField()
    answer:   str = dspy.InputField()
    score:    int = dspy.OutputField(desc="Entier 1..5")


def make_metric(eval_lm: dspy.LM, documents: dict):
    judge = dspy.Predict(EvalQualite)
    counter = {"n": 0}

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        p = pred or {}
        answer = str(p.get("answer") or "")
        if not answer:
            return 0.0
        doc = documents[p["document_id"]]
        with dspy.settings.context(lm=eval_lm):
            out = judge(document=doc, question=str(p.get("question") or ""), answer=answer)
        try:
            js = int(out.score)
        except Exception:
            js = 3
        s = note_paire(js, p.get("type_question", ""), answer)
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"[métrique] appel {counter['n']} note={s:.2f}")
        return s

    return metric


def _extract_best_prompt(compiled, teleprompter, initial: str) -> str:
    candidates = []
    for attr in ("best_prompt", "best_prompt_str", "best_prompt_text"):
        v = getattr(teleprompter, attr, None)
        if v:
            candidates.append(str(v))
    bp = getattr(teleprompter, "best_prompts", None)
    if isinstance(bp, dict):
        candidates.extend(str(v) for v in bp.values() if v)
    sig = getattr(getattr(compiled, "generate", None), "signature", None)
    if sig:
        instr = getattr(sig, "instructions", None) or getattr(sig, "__doc__", None)
        if instr:
            candidates.append(str(instr))
    candidates.append(initial)
    for c in candidates:
        if c and len(c.strip()) > 40 and "given the fields" not in c.lower():
            return c
    return initial


def main(argv=None) -> int:
    cfg = parse_args(argv)
    here = Path(cfg.domaines_path).resolve().parent
    domaines = json.loads(Path(cfg.domaines_path).read_text(encoding="utf-8"))
    documents = {d["domaine"]: (here / d["document"]).read_text(encoding="utf-8") for d in domaines}
    initial_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")
    seed = json.loads(Path(cfg.seed_path).read_text(encoding="utf-8"))
    configure_lm(cfg.generator_model, cfg.reflection_temperature, cfg.deepseek_api_base)
    eval_lm = configure_eval_lm(cfg.eval_model, cfg.deepseek_api_base)
    generator = build_example_generator(initial_prompt, documents)
    trainset = build_trainset(seed)
    teleprompter = dspy.GEPA(
        metric=make_metric(eval_lm, documents),
        reflection_lm=dspy.settings.lm,
        max_metric_calls=cfg.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
    )
    compiled = teleprompter.compile(generator, trainset=trainset)
    best_prompt = _extract_best_prompt(compiled, teleprompter, initial_prompt)
    Path(cfg.gepa_prompt_path).write_text(best_prompt, encoding="utf-8")
    print(f"Prompt optimisé sauvegardé dans {cfg.gepa_prompt_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(filename="gepa_qa_llm_calls.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    raise SystemExit(main())
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_gepa_qa.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/promptGenGEPA_qa.py gen_corpus_qa/tests/test_gepa_qa.py
git commit -m "feat(qa): optimisation GEPA du prompt QA (métrique note_paire testée)"
```

---

### Task 5: `eval_qa.py` — harnais d'évaluation A/B (llama.cpp + PEFT)

**Files:**
- Create: `gen_corpus_qa/eval_qa.py`
- Test: `gen_corpus_qa/tests/test_eval_qa.py`

**Interfaces:**
- Consomme : `corpus_qa_eval.jsonl` (Task 3), un backend d'inférence (llama-server sur `--base-url`, ou PEFT en Python), le juge DeepSeek.
- Produit : un rapport texte comparant nu vs adaptateur (scores agrégés ancrage/exactitude/refus). Fonctions pures testables : `parse_eval_row(row) -> dict` et `aggregate(scores) -> dict`.

- [ ] **Step 1: Écrire le test des fonctions pures**

`parse_eval_row` extrait, d'une ligne ChatML de `corpus_qa_eval.jsonl`, le document + la question (depuis le `user`) et la réponse de référence (depuis l'`assistant`), et déduit si c'est un cas de refus (réponse de référence ressemblant à un refus). `aggregate` moyenne des scores 0..1.

Écrire dans `gen_corpus_qa/tests/test_eval_qa.py` :

```python
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from eval_qa import parse_eval_row, aggregate


def test_parse_eval_row():
    row = {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Document:\nDOC\n\nQuestion: Q ?"},
        {"role": "assistant", "content": "Cette information ne figure pas dans le document."},
    ]}
    p = parse_eval_row(row)
    assert p["document"] == "DOC"
    assert p["question"] == "Q ?"
    assert p["reference"] == "Cette information ne figure pas dans le document."
    assert p["is_refus"] is True


def test_aggregate_moyenne():
    assert aggregate([1.0, 0.5, 0.0]) == 0.5
    assert aggregate([]) == 0.0
```

- [ ] **Step 2: Lancer le test pour vérifier qu'il échoue**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_eval_qa.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Écrire `eval_qa.py`**

Écrire dans `gen_corpus_qa/eval_qa.py` :

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Évaluation A/B d'un adaptateur QA : compare phi-4 nu vs phi-4+adaptateur.

Deux backends d'inférence :
  --backend llama  : interroge un llama-server (--base-url) ; le mode "nu" vs
                     "adapté" est réglé par le scale de l'adaptateur via
                     POST /lora-adapters (scale 0 = nu, scale 1 = adapté).
  --backend peft   : charge phi-4 + adaptateur PEFT en Python ; le mode "nu"
                     désactive l'adaptateur (model.disable_adapter()).
Le juge DeepSeek note chaque réponse candidate 1..5 (exactitude + ancrage).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT_QA = (
    "Vous êtes Tiron. Répondez à la question en vous appuyant uniquement sur le "
    "document fourni. Soyez concis et répondez en français. Si la réponse ne figure "
    "pas dans le document, indiquez-le clairement sans rien inventer."
)
REFUS_MARQUEURS = ("ne figure pas", "ne précise pas", "n'est pas dans", "pas dans le document",
                   "aucune information", "ne mentionne pas", "ne contient pas", "n'indique pas")


def _ressemble_refus(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in REFUS_MARQUEURS)


def parse_eval_row(row: dict) -> dict:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    ref = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    doc = user.split("Document:\n", 1)[1].split("\n\nQuestion: ", 1)[0]
    question = user.split("\n\nQuestion: ", 1)[1]
    return {"document": doc, "question": question, "reference": ref,
            "is_refus": _ressemble_refus(ref)}


def aggregate(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


# ---- backend llama-server -------------------------------------------------

def _http_json(url: str, body: dict, timeout=60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def set_lora_scale(base_url: str, scale: float) -> None:
    # id 0 = l'unique adaptateur chargé via --lora
    _http_json(base_url + "/lora-adapters", [{"id": 0, "scale": scale}])


def infer_llama(base_url: str, document: str, question: str) -> str:
    body = {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT_QA},
        {"role": "user", "content": f"Document:\n{document}\n\nQuestion: {question}"}],
        "max_tokens": 200, "temperature": 0}
    d = _http_json(base_url + "/v1/chat/completions", body)
    return d["choices"][0]["message"]["content"].strip()


# ---- backend PEFT (Jupyter/CLI) -------------------------------------------

def load_peft(model_path: str, adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tok


def infer_peft(model, tok, document: str, question: str, use_adapter: bool) -> str:
    import torch
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_QA},
            {"role": "user", "content": f"Document:\n{document}\n\nQuestion: {question}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ctx = model.disable_adapter() if not use_adapter else _nullctx()
    with torch.no_grad(), ctx:
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ---- juge DeepSeek --------------------------------------------------------

def judge_score(document: str, question: str, answer: str) -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY")
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    prompt = (f"Document:\n{document}\n\nQuestion: {question}\n\nRéponse: {answer}\n\n"
              "La réponse est-elle exacte, concise et entièrement fondée sur le document "
              "(refus correct si l'information est absente) ? Répondez par un entier 1 à 5 uniquement.")
    body = {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    txt = d["choices"][0]["message"]["content"].strip()
    digits = "".join(c for c in txt if c.isdigit())
    return max(1, min(5, int(digits))) if digits else 3


def run_condition(rows, infer_fn) -> dict:
    scores = []
    for r in rows:
        p = parse_eval_row(r)
        answer = infer_fn(p["document"], p["question"])
        js = judge_score(p["document"], p["question"], answer)
        scores.append(js / 5.0)
    return {"note_moyenne": aggregate(scores), "n": len(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="corpus_qa_eval.jsonl")
    ap.add_argument("--backend", choices=["llama", "peft"], default="llama")
    ap.add_argument("--base-url", default="http://127.0.0.1:8996")
    ap.add_argument("--model-path", default="/home/mauceric/Modèles/phi4")
    ap.add_argument("--adapter-path", default="/home/mauceric/lora_slm/checkpoints/qa-document")
    ap.add_argument("--limit", type=int, default=0, help="0 = tout le jeu d'éval")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    if args.backend == "llama":
        set_lora_scale(args.base_url, 0.0)
        nu = run_condition(rows, lambda d, q: infer_llama(args.base_url, d, q))
        set_lora_scale(args.base_url, 1.0)
        ad = run_condition(rows, lambda d, q: infer_llama(args.base_url, d, q))
    else:
        model, tok = load_peft(args.model_path, args.adapter_path)
        nu = run_condition(rows, lambda d, q: infer_peft(model, tok, d, q, use_adapter=False))
        ad = run_condition(rows, lambda d, q: infer_peft(model, tok, d, q, use_adapter=True))

    print(f"=== NU       : {nu['note_moyenne']:.3f} (n={nu['n']}) ===")
    print(f"=== ADAPTÉ   : {ad['note_moyenne']:.3f} (n={ad['n']}) ===")
    print(f"=== DELTA    : {ad['note_moyenne']-nu['note_moyenne']:+.3f} ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier qu'il passe**

Run: `cd ~/Secretarius/gen_corpus_qa && ../gen_corpus/.venv/bin/python -m pytest tests/test_eval_qa.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/eval_qa.py gen_corpus_qa/tests/test_eval_qa.py
git commit -m "feat(qa): harnais d'évaluation A/B nu-vs-adaptateur (backends llama + peft)"
```

---

### Task 6: Notebook Jupyter de test interactif

**Files:**
- Create: `gen_corpus_qa/test_adaptateur_qa.ipynb`

**Interfaces:**
- Consomme : `eval_qa.load_peft`, `eval_qa.infer_peft` (Task 5), le checkpoint PEFT (Task 7).

- [ ] **Step 1: Créer le notebook via un script Python**

Le notebook charge l'adaptateur en PEFT direct et permet de poser une question à la main sur un document, en comparant nu vs adapté. Créer le notebook avec ce script (installer `nbformat` d'abord s'il manque : `../lora_slm/lenv/bin/pip install nbformat`) :

```bash
cd ~/Secretarius/gen_corpus_qa
../lora_slm/lenv/bin/python - <<'PY'
import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# Test interactif de l'adaptateur QA-sur-document\n"
        "Charge phi-4-mini + adaptateur PEFT et compare la réponse nue vs adaptée."),
    nbf.v4.new_code_cell(
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "from eval_qa import load_peft, infer_peft\n"
        "MODEL='/home/mauceric/Modèles/phi4'\n"
        "ADAPTER='/home/mauceric/lora_slm/checkpoints/qa-document'\n"
        "model, tok = load_peft(MODEL, ADAPTER)"),
    nbf.v4.new_code_cell(
        "document = open('documents/config-materiel-logiciel.md').read()\n"
        "question = 'Quel modèle de langage vous anime ?'\n"
        "print('NU     :', infer_peft(model, tok, document, question, use_adapter=False))\n"
        "print('ADAPTÉ :', infer_peft(model, tok, document, question, use_adapter=True))"),
    nbf.v4.new_code_cell(
        "# Cas hors-document (doit refuser)\n"
        "question = 'Quelle est la météo à Paris ?'\n"
        "print('NU     :', infer_peft(model, tok, document, question, use_adapter=False))\n"
        "print('ADAPTÉ :', infer_peft(model, tok, document, question, use_adapter=True))"),
]
nb["cells"] = cells
nbf.write(nb, "test_adaptateur_qa.ipynb")
print("notebook écrit")
PY
```

- [ ] **Step 2: Vérifier que le notebook est un JSON valide**

Run: `cd ~/Secretarius/gen_corpus_qa && python3 -c "import json; json.load(open('test_adaptateur_qa.ipynb')); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
cd ~/Secretarius
git add gen_corpus_qa/test_adaptateur_qa.ipynb
git commit -m "feat(qa): notebook Jupyter de test interactif de l'adaptateur"
```

---

### Task 7: Exécution bout-en-bout et verdict A/B

**Files:**
- Aucun fichier de code créé — exécution des scripts des tâches précédentes. Produit (hors dépôt) : `gen_corpus_qa/corpus_qa.jsonl`, `corpus_qa_train.jsonl`, `corpus_qa_eval.jsonl`, `/home/mauceric/lora_slm/checkpoints/qa-document/`, `/home/mauceric/lora_slm/qa-document-lora-f16.gguf`.

**Interfaces:**
- Consomme tous les scripts des Tasks 2-5.

- [ ] **Step 1: Optimiser le prompt via GEPA (petit budget d'abord)**

```bash
cd ~/Secretarius/gen_corpus_qa
export DEEPSEEK_API_KEY="$(grep -oP '(?<=^DEEPSEEK_API_KEY=).*' ~/.config/secrets.env)"
../gen_corpus/.venv/bin/python promptGenGEPA_qa.py --max-metric-calls 120
```

Expected : `GEPAPrompt.txt` créé, non vide, plus détaillé que `prompt-init.txt`. Inspecter son contenu (`cat GEPAPrompt.txt`) — il doit rester une consigne de génération QA cohérente (pas un texte dégénéré). Si le contenu est incohérent, conserver `prompt-init.txt` comme prompt de génération (le renommer en `GEPAPrompt.txt`) et continuer.

- [ ] **Step 2: Générer un petit corpus témoin (30 exemples) et l'inspecter**

```bash
cd ~/Secretarius/gen_corpus_qa
../gen_corpus/.venv/bin/python generate_corpus_qa.py --count 30 --output corpus_qa_smoke.jsonl
../gen_corpus/.venv/bin/python -c "import json;[print(json.loads(l)['type_question'],'|',json.loads(l)['question'],'->',json.loads(l)['answer'][:80]) for l in open('corpus_qa_smoke.jsonl')]"
```

Expected : 30 lignes, questions plausibles, réponses ancrées, et les cas `hors_document` produisent bien un refus. Si la qualité est mauvaise (réponses inventées sur les `hors_document`, hors-sujet), retoucher `prompt-init.txt`/`seed.json` et refaire les Steps 1-2 avant de générer à grande échelle.

- [ ] **Step 2b: Vérifier la longueur des prompts (budget max_len)**

```bash
cd ~/Secretarius/gen_corpus_qa
../gen_corpus/.venv/bin/python to_lora_format_qa.py --corpus corpus_qa_smoke.jsonl --train /tmp/qa_tr.jsonl --eval /tmp/qa_ev.jsonl
../lora_slm/lenv/bin/python - <<'PY'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/mauceric/Modèles/phi4")
mx = 0
for l in open("/tmp/qa_tr.jsonl"):
    m = json.loads(l)["messages"]
    txt = tok.apply_chat_template(m, tokenize=False)
    mx = max(mx, len(tok(txt)["input_ids"]))
print("max tokens:", mx)
PY
```

Expected : `max tokens` bien en dessous de 2048. Si proche ou au-dessus, augmenter `--max_len` à l'entraînement (Step 4) en conséquence.

- [ ] **Step 3: Générer le corpus complet et le convertir**

```bash
cd ~/Secretarius/gen_corpus_qa
../gen_corpus/.venv/bin/python generate_corpus_qa.py --count 1500 --output corpus_qa.jsonl
../gen_corpus/.venv/bin/python to_lora_format_qa.py
wc -l corpus_qa_train.jsonl corpus_qa_eval.jsonl
```

Expected : ~1350 lignes train, ~150 lignes eval.

- [ ] **Step 4: Entraîner l'adaptateur**

```bash
mkdir -p /home/mauceric/lora_slm/checkpoints/qa-document
cd ~/Secretarius/lora_slm
./lenv/bin/python lora_train.py \
  --model_path /home/mauceric/Modèles/phi4 \
  --data_file /home/mauceric/Secretarius/gen_corpus_qa/corpus_qa_train.jsonl \
  --output_dir /home/mauceric/lora_slm/checkpoints/qa-document \
  --epochs 6 --lr 2e-4 --lora_r 16 --lora_alpha 32 --max_len 2048 \
  --log_file /home/mauceric/lora_slm/checkpoints/qa-document/training.log
```

Expected : la perte décroît dans `training.log` (comparer l'allure à `checkpoints/tiron-unified/training.log`). Entraînement long sur iGPU (memory-bound, ~plusieurs heures) — lancer en arrière-plan et surveiller le log.

- [ ] **Step 5: Test A/B backend PEFT (rapide, sans conversion GGUF)**

```bash
cd ~/Secretarius/gen_corpus_qa
export DEEPSEEK_API_KEY="$(grep -oP '(?<=^DEEPSEEK_API_KEY=).*' ~/.config/secrets.env)"
../lora_slm/lenv/bin/python eval_qa.py --backend peft --limit 40 \
  --model-path /home/mauceric/Modèles/phi4 \
  --adapter-path /home/mauceric/lora_slm/checkpoints/qa-document
```

Expected : trois lignes `NU`, `ADAPTÉ`, `DELTA`. **Critère de succès : DELTA nettement positif** (l'adaptateur bat le nu), en particulier grâce aux cas `hors_document`.

- [ ] **Step 6: Convertir en GGUF et test A/B backend llama.cpp (contrôle croisé)**

```bash
cd ~/llama.cpp
python convert_lora_to_gguf.py --base /home/mauceric/Modèles/phi4 \
  --outfile /home/mauceric/lora_slm/qa-document-lora-f16.gguf \
  /home/mauceric/lora_slm/checkpoints/qa-document

HSA_OVERRIDE_GFX_VERSION=10.3.0 nohup ~/llama.cpp/build-rocm/bin/llama-server \
  -m /home/mauceric/Modèles/Phi-4-mini-instruct-Q6_K.gguf \
  --lora /home/mauceric/lora_slm/qa-document-lora-f16.gguf \
  -c 2048 -ngl 99 --host 127.0.0.1 --port 8996 \
  > /tmp/eval_qa_server.log 2>&1 &
sleep 8
cd ~/Secretarius/gen_corpus_qa
export DEEPSEEK_API_KEY="$(grep -oP '(?<=^DEEPSEEK_API_KEY=).*' ~/.config/secrets.env)"
../gen_corpus/.venv/bin/python eval_qa.py --backend llama --base-url http://127.0.0.1:8996 --limit 40
kill %1
```

Expected : `DELTA` du même signe et proche de celui obtenu au Step 5 (le backend llama.cpp GGUF confirme le backend PEFT — une divergence forte signalerait un problème de conversion GGUF). Le port 8996 ne touche pas le service prod 8998.

- [ ] **Step 7: Consigner le verdict**

Créer `gen_corpus_qa/RESULTATS_AB.md` résumant : taille du corpus, note NU, note ADAPTÉ, DELTA (PEFT et llama.cpp), et la décision (poursuivre vers l'intégration OpenClaw si DELTA nettement positif, sinon documenter l'arrêt). Committer :

```bash
cd ~/Secretarius
git add gen_corpus_qa/RESULTATS_AB.md
git commit -m "docs(qa): verdict de l'évaluation A/B nu-vs-adaptateur QA"
```

(Les fichiers `corpus_qa*.jsonl`, checkpoints et `.gguf` restent hors dépôt, comme les adaptateurs précédents.)

---

## Risques / points ouverts pour l'exécution

- **Valeur ajoutée non garantie** (Step 5) : si le DELTA est marginal, ne pas poursuivre vers l'intégration — c'est le point de décision prévu par le spec.
- **Qualité GEPA** (Step 1) : GEPA peut produire un prompt dégénéré ; le repli sur `prompt-init.txt` est prévu.
- **Longueur de contexte** (Step 2b) : si un document seed est long, `--max_len 2048` peut devoir être relevé — vérifié avant l'entraînement.
- **Extraction de la clé DeepSeek** : les commandes lisent `~/.config/secrets.env` ; adapter si la clé est ailleurs. Ne jamais afficher la clé.
- **Fidélité GGUF** (Step 6) : contrôle croisé PEFT vs llama.cpp — panne de conversion GGUF déjà rencontrée par le passé.
