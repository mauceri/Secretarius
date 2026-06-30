# gen_corpus — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Créer `Secretarius/gen_corpus/`, un pipeline GEPA + DeepSeek qui génère un corpus LoRA associant des messages utilisateurs Tiron à leur action JSON `{"command": ..., "args": ...}`.

**Architecture:** Adaptation de `~/gen_corpus_gepa_codex` — même infrastructure DSPy/GEPA, domaine remplacé (notes de lecture → paires requête/action). Pipeline linéaire en 4 scripts : conversion seed → optimisation GEPA → génération masse → conversion ChatML.

**Tech Stack:** Python 3.9+, DSPy (même version que `gen_corpus_gepa_codex`), DeepSeek via API OpenAI-compatible.

## Global Constraints

- Répertoire cible : `~/Secretarius/gen_corpus/`
- `DEEPSEEK_API_KEY` requis pour les runs réelles ; les tests unitaires n'effectuent aucun appel réseau
- Format de sortie final : ChatML (phi-4-mini-instruct)
- System prompt LoRA (constant, ne pas modifier) : `Routeur de commandes Tiron. Pour chaque message, répondre uniquement avec un objet JSON : {"command": "/commande" ou null, "args": "arguments bruts ou chaîne vide"}.`
- Langue de génération : français uniquement
- Un commit par tâche

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `intentions.json` | 10 intentions + commande + variantes applicables |
| `registres.json` | 5 registres de message |
| `prompt-init.txt` | Prompt initial soumis à GEPA |
| `requirements.txt` | Dépendances Python |
| `convert_seed.py` | `corpus-intentions-seed.md` → `seed.json` |
| `promptGenGEPA.py` | Optimise le prompt via GEPA → `GEPAPrompt.txt` |
| `generate_corpus.py` | Génère `corpus.jsonl` avec le prompt optimisé |
| `to_lora_format.py` | `corpus.jsonl` → `corpus_lora*.jsonl` (ChatML, split 90/10) |
| `inspect_corpus.py` | Validation manuelle par échantillonnage |
| `tests/test_convert_seed.py` | Tests unitaires : parsing, extract_args, structural_score |
| `tests/test_generate.py` | Tests intégration : to_lora_format, mock LM |

---

### Task 1 : Scaffolding — structure et fichiers de données

**Files:**
- Create: `gen_corpus/requirements.txt`
- Create: `gen_corpus/intentions.json`
- Create: `gen_corpus/registres.json`
- Create: `gen_corpus/prompt-init.txt`
- Create: `gen_corpus/tests/__init__.py`

**Interfaces:**
- Produces: `intentions.json` — `list[{intention, command, variantes}]` lu par tous les scripts suivants
- Produces: `registres.json` — `list[str]` lu par `promptGenGEPA.py` et `generate_corpus.py`
- Produces: `prompt-init.txt` — texte lu par `promptGenGEPA.py`

- [ ] **Step 1 : Créer la structure**

```bash
mkdir -p ~/Secretarius/gen_corpus/tests
touch ~/Secretarius/gen_corpus/tests/__init__.py
```

- [ ] **Step 2 : Créer requirements.txt**

```
# gen_corpus/requirements.txt
dspy>=2.5.0
```

Utiliser le même venv que `gen_corpus_gepa_codex` ou : `python3 -m venv gcenv && gcenv/bin/pip install dspy`.

- [ ] **Step 3 : Créer intentions.json**

```json
[
  {"intention": "wiki_capture",   "command": "/c",           "variantes": ["url_avec_tags","url_seule","note_sans_url","avec_directive_simple","avec_ref","avec_fichier"]},
  {"intention": "wiki_ingest",    "command": "/ingest",      "variantes": ["sans_args"]},
  {"intention": "wiki_status",    "command": "/wiki-status", "variantes": ["sans_args"]},
  {"intention": "wiki_query",     "command": "/q",           "variantes": ["question_courte","question_longue"]},
  {"intention": "source_read",    "command": "/source",      "variantes": ["url_seule","url_avec_consigne"]},
  {"intention": "gog_mail",       "command": "/mail",        "variantes": ["envoi","lecture","recherche"]},
  {"intention": "gog_calendar",   "command": "/agenda",      "variantes": ["creation","lecture","suppression"]},
  {"intention": "gog_drive",      "command": "/drive",       "variantes": ["recherche","liste","partage"]},
  {"intention": "meta_assistant", "command": "/help",        "variantes": ["sans_args"]},
  {"intention": "out_of_scope",   "command": null,           "variantes": ["action_impossible"]}
]
```

- [ ] **Step 4 : Créer registres.json**

```json
["formel", "familier", "télégraphique", "poli", "abrégé"]
```

- [ ] **Step 5 : Créer prompt-init.txt**

```
Générez un exemple réaliste de message utilisateur adressé à Tiron, un assistant personnel,
ainsi que la commande et les arguments correspondants.

Tiron reconnaît les commandes suivantes :
- /c [#tags] [directives] <url|texte> — capturer une URL ou une note dans le wiki
- /ingest — lancer l'ingestion des captures en attente (sans argument)
- /wiki-status — afficher l'état du wiki (sans argument)
- /q <question> — interroger le wiki
- /source <url> [consigne] — lire une page externe maintenant (sans la sauvegarder)
- /mail <description> — gérer l'email
- /agenda <description> — gérer le calendrier
- /drive <description> — gérer les fichiers Drive
- /help — aide sur l'assistant (sans argument)
- null — demande hors périmètre de Tiron

Directives pour /c : @simple (capture directe sans ingestion), file:<chemin> (inclure un fichier
local), ref:<slug> (wikilink vers un document existant).

Règles de génération :
- Le message doit être rédigé dans le registre indiqué (formel / familier / télégraphique / poli / abrégé).
- Le message doit illustrer le type de variante indiqué.
- Le champ args contient les arguments bruts passés après la commande (chaîne vide si la commande
  n'en prend pas).
- Langue : français uniquement.

=== CONTRAT IMMUTABLE — NE PAS MODIFIER ===
Entrées : intention, registre, variante
Sorties : text (message utilisateur), command (commande Tiron ou null), args (arguments bruts ou "")
Les prompts intermédiaires doivent être rédigés en français.
=== FIN CONTRAT IMMUTABLE ===
```

- [ ] **Step 6 : Commit**

```bash
git -C ~/Secretarius add gen_corpus/
git -C ~/Secretarius commit -m "feat: gen_corpus scaffolding — fichiers de données et prompt initial"
```

---

### Task 2 : convert_seed.py

**Files:**
- Create: `gen_corpus/convert_seed.py`
- Test: `gen_corpus/tests/test_convert_seed.py`

**Interfaces:**
- Consumes: `../corpus-intentions-seed.md`, `intentions.json`
- Produces: `seed.json` — `list[{text, intention, registre, variante, action: {command, args}}]` consommée par `promptGenGEPA.py`
- Produces: `extract_args(text: str, intention: str) -> str` — importable par les tests
- Produces: `infer_variante(text: str, intention: str) -> str` — importable par les tests
- Produces: `infer_registre(text: str) -> str` — importable par les tests
- Produces: `parse_seed(seed_md_path: str, intentions_data: list[dict]) -> list[dict]` — importable par les tests

- [ ] **Step 1 : Écrire les tests en premier**

```python
# gen_corpus/tests/test_convert_seed.py
import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_seed import extract_args, infer_variante, infer_registre, parse_seed

MINI_SEED = """\
## 1. `wiki_capture` — capturer

1. garde cet article pour moi : https://example.com #ia
2. /c #ia https://example.com
3. /c @simple #notion ma note
4. /c ref:bm25-intro note sur BM25
5. /c file:/home/user/doc.md #ia

## 2. `wiki_ingest` — ingérer

1. ingère les fichiers en attente
2. /ingest

## 10. `out_of_scope` — hors périmètre

1. commande une pizza
2. réserve un billet de train
"""

INTENTIONS = [
    {"intention": "wiki_capture",  "command": "/c",      "variantes": ["url_avec_tags","url_seule","note_sans_url","avec_directive_simple","avec_ref","avec_fichier"]},
    {"intention": "wiki_ingest",   "command": "/ingest", "variantes": ["sans_args"]},
    {"intention": "out_of_scope",  "command": None,      "variantes": ["action_impossible"]},
]


def test_extract_args_url_tags():
    result = extract_args("garde cet article : https://example.com #ia", "wiki_capture")
    assert "https://example.com" in result
    assert "#ia" in result

def test_extract_args_slash_command():
    result = extract_args("/c #ia https://example.com", "wiki_capture")
    assert "#ia" in result
    assert "https://example.com" in result

def test_extract_args_directive_simple():
    result = extract_args("/c @simple #notion ma note", "wiki_capture")
    assert "@simple" in result
    assert "#notion" in result

def test_extract_args_ref():
    result = extract_args("/c ref:bm25-intro note sur BM25", "wiki_capture")
    assert "ref:bm25-intro" in result

def test_extract_args_no_args_ingest():
    assert extract_args("ingère les fichiers en attente", "wiki_ingest") == ""

def test_extract_args_no_args_slash_ingest():
    assert extract_args("/ingest", "wiki_ingest") == ""

def test_infer_variante_url_tags():
    assert infer_variante("garde ce lien https://ex.com #ia", "wiki_capture") == "url_avec_tags"

def test_infer_variante_directive_simple():
    assert infer_variante("/c @simple #notion ma note", "wiki_capture") == "avec_directive_simple"

def test_infer_variante_ref():
    assert infer_variante("/c ref:bm25-intro note", "wiki_capture") == "avec_ref"

def test_infer_registre_slash():
    assert infer_registre("/c https://ex.com") == "télégraphique"

def test_infer_registre_stp():
    assert infer_registre("ingestion stp") == "familier"

def test_parse_seed_structure(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text(MINI_SEED, encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    assert len(entries) > 0
    for e in entries:
        assert set(e.keys()) >= {"text", "intention", "registre", "variante", "action"}
        assert set(e["action"].keys()) == {"command", "args"}

def test_out_of_scope_null_command(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text(MINI_SEED, encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    oos = [e for e in entries if e["intention"] == "out_of_scope"]
    assert len(oos) >= 2
    assert all(e["action"]["command"] is None for e in oos)

def test_wiki_ingest_no_args(tmp_path):
    seed_md = tmp_path / "seed.md"
    seed_md.write_text(MINI_SEED, encoding="utf-8")
    entries = parse_seed(str(seed_md), INTENTIONS)
    ingest = [e for e in entries if e["intention"] == "wiki_ingest"]
    assert all(e["action"]["args"] == "" for e in ingest)
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_convert_seed.py -v 2>&1 | head -5
```
Attendu : `ModuleNotFoundError: No module named 'convert_seed'`

- [ ] **Step 3 : Implémenter convert_seed.py**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convertit corpus-intentions-seed.md en seed.json."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any

REQUIRES_ARGS = {"wiki_capture", "wiki_query", "source_read", "gog_mail", "gog_calendar", "gog_drive"}

# 9 exemples manuels couvrant les 3 directives de /c (3 par directive)
MANUAL_DIRECTIVES = [
    {"text": "/c @simple #notion ma note sur la productivité",
     "intention": "wiki_capture", "registre": "télégraphique", "variante": "avec_directive_simple",
     "action": {"command": "/c", "args": "@simple #notion ma note sur la productivité"}},
    {"text": "capture @simple cette note rapide #idée",
     "intention": "wiki_capture", "registre": "familier", "variante": "avec_directive_simple",
     "action": {"command": "/c", "args": "@simple #idée cette note rapide"}},
    {"text": "sauvegarde directement dans sources @simple : réflexions sur le RAG",
     "intention": "wiki_capture", "registre": "poli", "variante": "avec_directive_simple",
     "action": {"command": "/c", "args": "@simple réflexions sur le RAG"}},
    {"text": "/c file:/home/user/notes.md #ia",
     "intention": "wiki_capture", "registre": "télégraphique", "variante": "avec_fichier",
     "action": {"command": "/c", "args": "file:/home/user/notes.md #ia"}},
    {"text": "capture le contenu de ce fichier : file:/home/user/rapport.md",
     "intention": "wiki_capture", "registre": "formel", "variante": "avec_fichier",
     "action": {"command": "/c", "args": "file:/home/user/rapport.md"}},
    {"text": "ajoute ce fichier à mon wiki file:/home/user/doc.txt #documentation",
     "intention": "wiki_capture", "registre": "poli", "variante": "avec_fichier",
     "action": {"command": "/c", "args": "file:/home/user/doc.txt #documentation"}},
    {"text": "/c ref:bm25-intro note complémentaire sur BM25",
     "intention": "wiki_capture", "registre": "télégraphique", "variante": "avec_ref",
     "action": {"command": "/c", "args": "ref:bm25-intro note complémentaire sur BM25"}},
    {"text": "capture avec référence à l'article zettelkasten ref:zettelkasten-intro",
     "intention": "wiki_capture", "registre": "poli", "variante": "avec_ref",
     "action": {"command": "/c", "args": "ref:zettelkasten-intro"}},
    {"text": "note liée ref:colbert-paper : les embeddings multi-vecteurs sont plus précis",
     "intention": "wiki_capture", "registre": "abrégé", "variante": "avec_ref",
     "action": {"command": "/c", "args": "ref:colbert-paper les embeddings multi-vecteurs sont plus précis"}},
]


def extract_args(text: str, intention: str) -> str:
    if intention not in REQUIRES_ARGS:
        return ""
    slash_match = re.match(r"^/\w[\w-]*\s*(.*)", text)
    raw = slash_match.group(1).strip() if slash_match else text
    if intention == "wiki_capture":
        directives = re.findall(r"@\w+", raw)
        refs = re.findall(r"\bref:\S+", raw)
        files = re.findall(r"\bfile:\S+", raw)
        urls = re.findall(r"https?://\S+", raw)
        tags = re.findall(r"#\w+", raw)
        parts = directives + refs + files + urls + tags
        return " ".join(parts).strip()
    if intention == "source_read":
        urls = re.findall(r"https?://\S+", raw)
        return urls[0] if urls else raw
    if intention == "wiki_query":
        cleaned = re.sub(
            r"^(que dit le wiki sur|cherche dans ma base\s*[:»]?|interroge le wiki sur|"
            r"d.après mes notes[,\s]+|que sais-je déjà sur|résume ce que ma base dit sur|"
            r"retrouve mes notes sur|d.après le wiki|qu.ai-je sauvegardé à propos de|"
            r"qu.est-ce que j.ai noté sur|dans mon wiki|dans mes documents|"
            r"que contient mon wiki au sujet de|résume mes notes sur|liste mes sources sur|"
            r"quelles pages parlent de|cherche\s+|résume\s+|fais une synthèse de mes sources sur)\s*",
            "", raw, flags=re.IGNORECASE
        ).strip().rstrip("?").strip()
        return cleaned or raw
    return raw


def infer_variante(text: str, intention: str) -> str:
    if intention == "wiki_capture":
        if "@simple" in text:
            return "avec_directive_simple"
        if "ref:" in text:
            return "avec_ref"
        if "file:" in text:
            return "avec_fichier"
        if "#" in text and "http" in text:
            return "url_avec_tags"
        if "http" in text:
            return "url_seule"
        return "note_sans_url"
    if intention == "wiki_query":
        return "question_longue" if len(text) > 60 else "question_courte"
    if intention == "source_read":
        m = re.search(r"https?://\S+", text)
        return "url_avec_consigne" if m and text[m.end():].strip() else "url_seule"
    if intention == "gog_mail":
        if any(w in text.lower() for w in ["envoie", "écris", "rédige", "réponds", "transfère"]):
            return "envoi"
        if any(w in text.lower() for w in ["lis", "relève", "reçu", "mails", "boîte", "résume", "liste"]):
            return "lecture"
        return "recherche"
    if intention == "gog_calendar":
        if any(w in text.lower() for w in ["crée", "ajoute", "planifie", "bloque", "réserve", "invite"]):
            return "creation"
        if any(w in text.lower() for w in ["annule", "déplace", "supprime", "décale"]):
            return "suppression"
        return "lecture"
    if intention == "gog_drive":
        if any(w in text.lower() for w in ["retrouve", "cherche", "où est", "ouvre", "trouve"]):
            return "recherche"
        if any(w in text.lower() for w in ["liste", "quels fichiers", "contenu"]):
            return "liste"
        return "partage"
    return "sans_args"


def infer_registre(text: str) -> str:
    if text.startswith("/"):
        return "télégraphique"
    if any(w in text.lower() for w in ["veuillez", "pourriez", "auriez-vous", "je vous"]):
        return "formel"
    if any(w in text.lower() for w in ["stp", "sil te plaît", "t'", "t'as"]):
        return "familier"
    if len(text.split()) <= 4:
        return "abrégé"
    return "poli"


def parse_seed(seed_md_path: str, intentions_data: list[dict]) -> list[dict]:
    intentions_map = {item["intention"]: item for item in intentions_data}
    text = Path(seed_md_path).read_text(encoding="utf-8")
    entries: list[dict] = []
    section_re = re.compile(
        r"##\s+\d+\.\s+`(\w+)`[^\n]*\n(.*?)(?=\n##\s+\d+\.|\Z)", re.DOTALL
    )
    for m in section_re.finditer(text):
        intention = m.group(1)
        if intention not in intentions_map:
            continue
        command = intentions_map[intention]["command"]
        for line in m.group(2).splitlines():
            item = re.match(r"^\d+\.\s+(.*)", line.strip())
            if not item:
                continue
            msg = item.group(1).strip()
            if not msg:
                continue
            entries.append({
                "text": msg,
                "intention": intention,
                "registre": infer_registre(msg),
                "variante": infer_variante(msg, intention),
                "action": {"command": command, "args": extract_args(msg, intention)},
            })
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-md", default="../corpus-intentions-seed.md")
    parser.add_argument("--intentions", default="intentions.json")
    parser.add_argument("--output", default="seed.json")
    a = parser.parse_args()
    intentions_data = json.loads(Path(a.intentions).read_text(encoding="utf-8"))
    entries = parse_seed(a.seed_md, intentions_data)
    entries.extend(MANUAL_DIRECTIVES)
    Path(a.output).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(entries)} entrées écrites dans {a.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer les tests**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_convert_seed.py -v
```
Attendu : tous les tests passent

- [ ] **Step 5 : Générer seed.json et vérifier**

```bash
cd ~/Secretarius/gen_corpus
python convert_seed.py
python -c "
import json
from collections import Counter
data = json.load(open('seed.json'))
c = Counter(e['intention'] for e in data)
for k, v in sorted(c.items()): print(f'{k:22s} {v}')
print('Total:', len(data))
"
```
Attendu : ~209 entrées, toutes les 10 intentions représentées

- [ ] **Step 6 : Commit**

```bash
git -C ~/Secretarius add gen_corpus/convert_seed.py gen_corpus/tests/test_convert_seed.py gen_corpus/seed.json
git -C ~/Secretarius commit -m "feat: gen_corpus convert_seed — seed.md → seed.json avec directives @simple/file/ref"
```

---

### Task 3 : promptGenGEPA.py

**Files:**
- Create: `gen_corpus/promptGenGEPA.py`
- Modify: `gen_corpus/tests/test_convert_seed.py` (ajouter tests métrique)

**Interfaces:**
- Consumes: `prompt-init.txt`, `intentions.json`, `registres.json`, `seed.json`
- Produces: `GEPAPrompt.txt` — texte du prompt optimisé, consommé par `generate_corpus.py`
- Produces: `structural_score(pred: dict, gold: dict) -> float` — importable
- Produces: `build_trainset(seed: list[dict]) -> list[dspy.Example]` — importable

- [ ] **Step 1 : Ajouter les tests de métrique à la fin de tests/test_convert_seed.py**

```python
# --- Tests structural_score et build_trainset ---

def test_structural_score_valid():
    from promptGenGEPA import structural_score
    assert structural_score({"command": "/c", "args": "https://ex.com"}, {"intention": "wiki_capture"}) == 1.0

def test_structural_score_missing_args():
    from promptGenGEPA import structural_score
    assert structural_score({"command": "/c", "args": ""}, {"intention": "wiki_capture"}) == 0.5

def test_structural_score_no_args_required():
    from promptGenGEPA import structural_score
    assert structural_score({"command": "/ingest", "args": ""}, {"intention": "wiki_ingest"}) == 1.0

def test_structural_score_out_of_scope():
    from promptGenGEPA import structural_score
    assert structural_score({"command": "null", "args": ""}, {"intention": "out_of_scope"}) == 1.0

def test_structural_score_wrong_command():
    from promptGenGEPA import structural_score
    assert structural_score({"command": "/inexistant", "args": "x"}, {"intention": "wiki_capture"}) < 1.0

def test_build_trainset():
    from promptGenGEPA import build_trainset
    seed = [{"text": "garde ce lien", "intention": "wiki_capture", "registre": "familier",
              "variante": "url_seule", "action": {"command": "/c", "args": "https://ex.com"}}]
    trainset = build_trainset(seed)
    assert len(trainset) == 1
    ex = trainset[0]
    assert ex.intention == "wiki_capture"
    assert ex.text == "garde ce lien"
    assert ex.command == "/c"
    assert ex.args == "https://ex.com"
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_convert_seed.py -k "structural_score or build_trainset" -v 2>&1 | head -5
```
Attendu : `ModuleNotFoundError: No module named 'promptGenGEPA'`

- [ ] **Step 3 : Implémenter promptGenGEPA.py**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation de prompt via GEPA pour le corpus d'intentions Tiron."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import dspy
from dspy.clients import configure_cache as dspy_configure_cache

logging.basicConfig(filename="gepa_llm_calls.log", filemode="a", level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None

COMMANDS_KNOWN = {"/c", "/ingest", "/wiki-status", "/q", "/source", "/mail", "/agenda", "/drive", "/help"}
REQUIRES_ARGS = {"wiki_capture", "wiki_query", "source_read", "gog_mail", "gog_calendar", "gog_drive"}


def _is_null_command(cmd: str) -> bool:
    return not cmd or cmd.strip().lower() in ("null", "none", "")


def structural_score(pred: dict, gold: dict) -> float:
    command = str(pred.get("command") or "").strip()
    args = str(pred.get("args") or "").strip()
    intention = str(gold.get("intention") or "").strip()
    cmd_ok = _is_null_command(command) if intention == "out_of_scope" else command in COMMANDS_KNOWN
    args_ok = bool(args) if intention in REQUIRES_ARGS else True
    return 0.5 * int(cmd_ok) + 0.5 * int(args_ok)


@dataclass
class Config:
    seed_path: str = "seed.json"
    prompt_path: str = "prompt-init.txt"
    gepa_prompt_path: str = "GEPAPrompt.txt"
    generator_model: str = "openai/deepseek-chat"
    eval_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    reflection_temperature: float = 1.0
    max_metric_calls: int = 300


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="seed.json")
    p.add_argument("--prompt", default="prompt-init.txt")
    p.add_argument("--gepa-prompt", default="GEPAPrompt.txt")
    p.add_argument("--generator-model", default="openai/deepseek-chat")
    p.add_argument("--eval-model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-metric-calls", type=int, default=300)
    a = p.parse_args(argv)
    return Config(seed_path=a.seed, prompt_path=a.prompt, gepa_prompt_path=a.gepa_prompt,
                  generator_model=a.generator_model, eval_model=a.eval_model,
                  deepseek_api_base=a.deepseek_api_base,
                  reflection_temperature=a.temperature, max_metric_calls=a.max_metric_calls)


def _ensure_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    return key


def configure_lm(model: str, temperature: float, api_base: str) -> dspy.LM:
    key = _ensure_key()
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    lm = dspy.LM(model=model, api_key=key, api_base=base, model_type="chat",
                 temperature=temperature, max_tokens=512, cache=False)
    dspy.settings.configure(lm=lm)
    return lm


def configure_eval_lm(model: str, api_base: str) -> dspy.LM:
    key = _ensure_key()
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    return dspy.LM(model=model, api_key=key, api_base=base, model_type="chat",
                   temperature=0.0, max_tokens=16, cache=False)


def build_example_generator(prompt_text: str) -> dspy.Module:
    class GenerateExample(dspy.Signature):
        f"""{prompt_text}"""
        intention: str = dspy.InputField(desc="Intention Tiron à illustrer (ex: wiki_capture)")
        registre:  str = dspy.InputField(desc="Registre du message: formel, familier, télégraphique, poli, abrégé")
        variante:  str = dspy.InputField(desc="Type de variante (ex: url_avec_tags, question_courte, sans_args…)")
        text:    str = dspy.OutputField(desc="Message utilisateur réaliste en français")
        command: str = dspy.OutputField(desc="Commande Tiron (/c /ingest /wiki-status /q /source /mail /agenda /drive /help) ou null")
        args:    str = dspy.OutputField(desc="Arguments bruts de la commande (chaîne vide si pas d'args)")

    class ExampleGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(GenerateExample)

        def forward(self, intention, registre, variante):
            r = self.generate(intention=intention, registre=registre, variante=variante)
            return {"text": r.text, "intention": intention, "command": r.command,
                    "args": r.args, "registre": registre, "variante": variante}

    return ExampleGenerator()


def build_trainset(seed: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(
            intention=ex["intention"],
            registre=ex.get("registre", "poli"),
            variante=ex.get("variante", "sans_args"),
            text=ex["text"],
            command=ex["action"]["command"] if ex["action"]["command"] is not None else "null",
            args=ex["action"]["args"],
        ).with_inputs("intention", "registre", "variante")
        for ex in seed
    ]


class EvalRealisme(dspy.Signature):
    """Ce message ressemble-t-il à une vraie requête utilisateur adressée à un assistant ?
    Répondre avec un entier 1..5 uniquement, sans commentaire."""
    text:  str = dspy.InputField(desc="Message utilisateur généré")
    score: int = dspy.OutputField(desc="Entier 1..5")


def make_metric(eval_lm: dspy.LM):
    class Evaluator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict(EvalRealisme)

        def forward(self, text: str) -> int:
            with dspy.settings.context(lm=eval_lm):
                out = self.pred(text=text)
            try:
                return max(1, min(5, int(out.score)))
            except Exception:
                return 3

    evaluator = Evaluator()
    counter = {"n": 0}

    def metric(gold, pred, trace=None):
        text = str((pred or {}).get("text") or "")
        if not text:
            return 0.0
        s_struct = structural_score(pred, gold)
        try:
            s_real = evaluator(text=text) / 5.0
        except Exception:
            s_real = 0.6
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"[métrique] appel {counter['n']} struct={s_struct:.2f} réalisme={s_real:.2f}")
        return 0.5 * s_real + 0.5 * s_struct

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
    initial_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")
    seed = json.loads(Path(cfg.seed_path).read_text(encoding="utf-8"))
    configure_lm(cfg.generator_model, cfg.reflection_temperature, cfg.deepseek_api_base)
    eval_lm = configure_eval_lm(cfg.eval_model, cfg.deepseek_api_base)
    generator = build_example_generator(initial_prompt)
    trainset = build_trainset(seed)
    teleprompter = dspy.GEPA(
        metric=make_metric(eval_lm),
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
    raise SystemExit(main())
```

- [ ] **Step 4 : Lancer tous les tests**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_convert_seed.py -v
```
Attendu : tous les tests passent, y compris les tests structural_score et build_trainset

- [ ] **Step 5 : Commit**

```bash
git -C ~/Secretarius add gen_corpus/promptGenGEPA.py gen_corpus/tests/test_convert_seed.py
git -C ~/Secretarius commit -m "feat: gen_corpus promptGenGEPA — GEPA avec métrique hybride règles+LLM"
```

---

### Task 4 : generate_corpus.py

**Files:**
- Create: `gen_corpus/generate_corpus.py`
- Create: `gen_corpus/tests/test_generate.py`

**Interfaces:**
- Consumes: `GEPAPrompt.txt` (fallback : `prompt-init.txt`), `intentions.json`, `registres.json`
- Produces: `corpus.jsonl` — une ligne JSON par exemple : `{text, intention, registre, variante, action: {command, args}}`
- Produces: `generate_one(predict, intention: str, registre: str, variante: str) -> dict` — importable

- [ ] **Step 1 : Écrire les tests**

```python
# gen_corpus/tests/test_generate.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_generate_one_structure():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "garde cet article https://example.com"
    mock_result.command = "/c"
    mock_result.args = "https://example.com"
    mock_predict = MagicMock(return_value=mock_result)

    entry = generate_one(mock_predict, "wiki_capture", "familier", "url_seule")

    assert entry["text"] == "garde cet article https://example.com"
    assert entry["intention"] == "wiki_capture"
    assert entry["registre"] == "familier"
    assert entry["variante"] == "url_seule"
    assert entry["action"]["command"] == "/c"
    assert entry["action"]["args"] == "https://example.com"


def test_generate_one_null_command():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "commande une pizza"
    mock_result.command = "null"
    mock_result.args = ""
    mock_predict = MagicMock(return_value=mock_result)

    entry = generate_one(mock_predict, "out_of_scope", "familier", "action_impossible")
    assert entry["action"]["command"] is None
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_generate.py -v 2>&1 | head -5
```
Attendu : `ModuleNotFoundError: No module named 'generate_corpus'`

- [ ] **Step 3 : Implémenter generate_corpus.py**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère corpus.jsonl à partir du prompt optimisé par GEPA."""
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
    count: int = 1000
    batch_size: int = 50
    report_every: int = 50
    prompt_path: str = "GEPAPrompt.txt"
    prompt_fallback: str = "prompt-init.txt"
    intentions_path: str = "intentions.json"
    registres_path: str = "registres.json"
    output: str = "corpus.jsonl"
    generator_model: str = "openai/deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    temperature: float = 0.9


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--prompt", default="GEPAPrompt.txt")
    p.add_argument("--intentions", default="intentions.json")
    p.add_argument("--registres", default="registres.json")
    p.add_argument("--output", default="corpus.jsonl")
    p.add_argument("--model", default="openai/deepseek-chat")
    p.add_argument("--deepseek-api-base", default="https://api.deepseek.com")
    p.add_argument("--temperature", type=float, default=0.9)
    a = p.parse_args(argv)
    return Config(count=a.count, batch_size=a.batch_size, report_every=a.report_every,
                  prompt_path=a.prompt, intentions_path=a.intentions, registres_path=a.registres,
                  output=a.output, generator_model=a.model,
                  deepseek_api_base=a.deepseek_api_base, temperature=a.temperature)


def _build_signature(prompt_text: str):
    class GenerateExample(dspy.Signature):
        f"""{prompt_text}"""
        intention: str = dspy.InputField(desc="Intention Tiron à illustrer")
        registre:  str = dspy.InputField(desc="Registre du message")
        variante:  str = dspy.InputField(desc="Type de variante")
        text:    str = dspy.OutputField(desc="Message utilisateur réaliste en français")
        command: str = dspy.OutputField(desc="Commande Tiron ou null")
        args:    str = dspy.OutputField(desc="Arguments bruts (chaîne vide si sans args)")
    return GenerateExample


def generate_one(predict, intention: str, registre: str, variante: str) -> dict:
    result = predict(intention=intention, registre=registre, variante=variante)
    cmd = result.command if result.command and result.command.lower() not in ("null", "none") else None
    return {"text": result.text, "intention": intention, "registre": registre,
            "variante": variante, "action": {"command": cmd, "args": result.args}}


def main(argv=None) -> int:
    cfg = parse_args(argv)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez DEEPSEEK_API_KEY dans l'environnement")
    base = os.getenv("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
    lm = dspy.LM(model=cfg.generator_model, api_key=api_key, api_base=base,
                 model_type="chat", temperature=cfg.temperature, max_tokens=256, cache=False)
    dspy.settings.configure(lm=lm)

    prompt_p = Path(cfg.prompt_path)
    prompt_text = (prompt_p if prompt_p.exists() else Path(cfg.prompt_fallback)).read_text(encoding="utf-8")
    predict = dspy.Predict(_build_signature(prompt_text))
    intentions = json.loads(Path(cfg.intentions_path).read_text(encoding="utf-8"))
    registres = json.loads(Path(cfg.registres_path).read_text(encoding="utf-8"))

    buffer = []
    stime = time.time()
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in range(cfg.count):
            obj = random.choice(intentions)
            try:
                entry = generate_one(predict, obj["intention"], random.choice(registres),
                                     random.choice(obj["variantes"]))
                buffer.append(entry)
            except Exception as e:
                print(f"[{i+1}] Erreur: {e}", flush=True)
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

- [ ] **Step 4 : Lancer les tests**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_generate.py -v
```
Attendu : les deux tests passent

- [ ] **Step 5 : Commit**

```bash
git -C ~/Secretarius add gen_corpus/generate_corpus.py gen_corpus/tests/test_generate.py
git -C ~/Secretarius commit -m "feat: gen_corpus generate_corpus — génération masse corpus.jsonl"
```

---

### Task 5 : to_lora_format.py

**Files:**
- Create: `gen_corpus/to_lora_format.py`
- Modify: `gen_corpus/tests/test_generate.py` (ajouter tests ChatML)

**Interfaces:**
- Consumes: `corpus.jsonl`
- Produces: `corpus_lora.jsonl`, `corpus_lora_train.jsonl`, `corpus_lora_eval.jsonl`
- Produces: `convert_entry(entry: dict) -> dict` — importable
- Produces: `to_lora(corpus_path, out_path, train_path, eval_path, eval_ratio, seed) -> None` — importable
- Produces: `SYSTEM_PROMPT: str` — constante importable

- [ ] **Step 1 : Ajouter les tests à la fin de tests/test_generate.py**

```python
def test_convert_entry_chatML():
    from to_lora_format import convert_entry, SYSTEM_PROMPT
    entry = {"text": "garde ce lien", "intention": "wiki_capture",
             "action": {"command": "/c", "args": "https://ex.com"}}
    result = convert_entry(entry)
    msgs = result["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1] == {"role": "user", "content": "garde ce lien"}
    assert msgs[2]["role"] == "assistant"
    parsed = json.loads(msgs[2]["content"])
    assert parsed == {"command": "/c", "args": "https://ex.com"}


def test_convert_entry_out_of_scope():
    from to_lora_format import convert_entry
    entry = {"text": "commande une pizza", "intention": "out_of_scope",
             "action": {"command": None, "args": ""}}
    parsed = json.loads(convert_entry(entry)["messages"][2]["content"])
    assert parsed["command"] is None
    assert parsed["args"] == ""


def test_split_90_10(tmp_path):
    from to_lora_format import to_lora
    entries = [{"text": f"msg {i}", "intention": "wiki_capture",
                "action": {"command": "/c", "args": f"https://ex{i}.com"}}
               for i in range(100)]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    out, train, eval_ = tmp_path / "out.jsonl", tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    to_lora(str(corpus), str(out), str(train), str(eval_))
    assert len(train.read_text().strip().splitlines()) == 90
    assert len(eval_.read_text().strip().splitlines()) == 10
    for line in out.read_text().strip().splitlines():
        msg = json.loads(line)
        assert len(msg["messages"]) == 3
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/test_generate.py -k "chatML or out_of_scope or split" -v 2>&1 | head -5
```
Attendu : `ModuleNotFoundError: No module named 'to_lora_format'`

- [ ] **Step 3 : Implémenter to_lora_format.py**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convertit corpus.jsonl en format ChatML pour fine-tuning LoRA (phi-4-mini-instruct)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    'Routeur de commandes Tiron. Pour chaque message, répondre uniquement avec un objet JSON : '
    '{"command": "/commande" ou null, "args": "arguments bruts ou chaîne vide"}.'
)


def convert_entry(entry: dict) -> dict:
    action = entry["action"]
    return {"messages": [
        {"role": "system",   "content": SYSTEM_PROMPT},
        {"role": "user",     "content": entry["text"]},
        {"role": "assistant","content": json.dumps(
            {"command": action["command"], "args": action["args"]}, ensure_ascii=False
        )},
    ]}


def to_lora(corpus_path: str, out_path: str, train_path: str, eval_path: str,
            eval_ratio: float = 0.1, seed: int = 42) -> None:
    lines = [l for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    converted = [convert_entry(json.loads(l)) for l in lines]
    random.seed(seed)
    random.shuffle(converted)
    n_eval = max(1, int(len(converted) * eval_ratio))
    for path, data in [(out_path, converted),
                       (train_path, converted[n_eval:]),
                       (eval_path, converted[:n_eval])]:
        Path(path).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in data), encoding="utf-8"
        )
    print(f"Total: {len(converted)} | Train: {len(converted)-n_eval} | Eval: {n_eval}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus.jsonl")
    p.add_argument("--output", default="corpus_lora.jsonl")
    p.add_argument("--train", default="corpus_lora_train.jsonl")
    p.add_argument("--eval", default="corpus_lora_eval.jsonl")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    a = p.parse_args()
    to_lora(a.corpus, a.output, a.train, a.eval, a.eval_ratio)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer tous les tests**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/ -v
```
Attendu : tous les tests passent

- [ ] **Step 5 : Commit**

```bash
git -C ~/Secretarius add gen_corpus/to_lora_format.py gen_corpus/tests/test_generate.py
git -C ~/Secretarius commit -m "feat: gen_corpus to_lora_format — ChatML + split train/eval 90/10"
```

---

### Task 6 : inspect_corpus.py

**Files:**
- Create: `gen_corpus/inspect_corpus.py`

**Interfaces:**
- Consumes: tout fichier JSONL au format corpus (champs `text`, `intention`, `action`, optionnels `registre`, `variante`)
- Produces: rapport console (distribution + échantillon)

Pas de TDD — outil de validation manuelle, hors pipeline automatique.

- [ ] **Step 1 : Implémenter inspect_corpus.py**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation manuelle du corpus par échantillonnage et statistiques."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus.jsonl")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    lines = [l for l in Path(a.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print("Corpus vide.")
        return
    entries = [json.loads(l) for l in lines]

    int_c  = Counter(e["intention"]             for e in entries)
    reg_c  = Counter(e.get("registre", "?")     for e in entries)
    cmd_c  = Counter(str(e["action"]["command"]) for e in entries)

    print(f"\nCorpus : {len(entries)} entrées — {a.corpus}")
    print("\nDistribution des intentions :")
    for k, v in sorted(int_c.items()):
        bar = "█" * (v * 30 // len(entries))
        print(f"  {k:22s} {v:4d} ({v/len(entries)*100:4.1f}%) {bar}")
    print("\nDistribution des registres :")
    for k, v in sorted(reg_c.items()):
        print(f"  {k:15s} {v:4d} ({v/len(entries)*100:4.1f}%)")
    print("\nDistribution des commandes :")
    for k, v in sorted(cmd_c.items()):
        print(f"  {k:15s} {v:4d}")

    if a.seed is not None:
        random.seed(a.seed)
    sample = random.sample(entries, min(a.sample, len(entries)))
    print(f"\n{'─'*60}\nÉchantillon ({len(sample)}) :\n{'─'*60}")
    for e in sample:
        print(f"\n[{e['intention']}] registre={e.get('registre','?')} variante={e.get('variante','?')}")
        print(f"  TEXT : {e['text']}")
        print(f"  CMD  : {e['action']['command']}   ARGS : {e['action']['args']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Vérifier avec le seed**

```bash
cd ~/Secretarius/gen_corpus
python inspect_corpus.py --corpus seed.json --sample 5
```
Attendu : distribution des 10 intentions + 5 exemples affichés sans erreur

- [ ] **Step 3 : Lancer la suite complète**

```bash
cd ~/Secretarius/gen_corpus
python -m pytest tests/ -v
```
Attendu : tous les tests passent

- [ ] **Step 4 : Commit final**

```bash
git -C ~/Secretarius add gen_corpus/inspect_corpus.py
git -C ~/Secretarius commit -m "feat: gen_corpus inspect_corpus — validation manuelle par échantillonnage"
```

---

## Ordre d'exécution du pipeline complet

Une fois toutes les tâches implémentées :

```bash
cd ~/Secretarius/gen_corpus

# 1. Convertir le seed
python convert_seed.py

# 2. Optimiser le prompt via GEPA (nécessite DEEPSEEK_API_KEY)
python promptGenGEPA.py --max-metric-calls 100   # run test rapide

# 3. Générer le corpus
python generate_corpus.py --count 1000 --output corpus.jsonl

# 4. Valider manuellement
python inspect_corpus.py --corpus corpus.jsonl --sample 30

# 5. Convertir en ChatML LoRA
python to_lora_format.py
```
