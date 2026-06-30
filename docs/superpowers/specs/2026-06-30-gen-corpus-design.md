# Design — gen_corpus : générateur de corpus LoRA pour Tiron

**Date :** 2026-06-30
**Statut :** approuvé
**But :** créer `Secretarius/gen_corpus/`, un pipeline GEPA + DeepSeek qui génère un corpus d'entraînement LoRA associant des messages utilisateurs Tiron à l'action JSON correspondante `{"command": ..., "args": ...}`.

---

## 1. Contexte et finalité

Le corpus produit sert à entraîner un adaptateur LoRA sur phi-4-mini-instruct (ou modèle SLM équivalent) pour qu'il joue le rôle de cerveau routeur de Tiron : à partir d'un message utilisateur en langage naturel, le modèle fine-tuné doit produire la commande Tiron et ses arguments bruts.

Base de départ : `~/gen_corpus_gepa_codex` (pipeline GEPA éprouvé pour la génération de notes) et `~/Secretarius/corpus-intentions-seed.md` (200 exemples graines couvrant 10 intentions).

---

## 2. Architecture et structure

```
Secretarius/gen_corpus/
├── convert_seed.py        # conversion corpus-intentions-seed.md → seed.json
├── promptGenGEPA.py       # optimisation GEPA du prompt de génération
├── generate_corpus.py     # génération masse avec le prompt optimisé
├── to_lora_format.py      # conversion corpus.jsonl → ChatML LoRA
├── inspect_corpus.py      # validation manuelle par échantillonnage
├── prompt-init.txt        # prompt initial soumis à GEPA
├── intentions.json        # 10 intentions + commande + variantes applicables
├── registres.json         # 5 registres de message
├── seed.json              # exemples convertis (produit par convert_seed.py)
├── GEPAPrompt.txt         # meilleur prompt trouvé (produit par promptGenGEPA.py)
├── corpus.jsonl           # corpus brut (produit par generate_corpus.py)
├── corpus_lora.jsonl      # corpus ChatML complet (produit par to_lora_format.py)
├── corpus_lora_train.jsonl
├── corpus_lora_eval.jsonl
├── requirements.txt
└── tests/
    ├── test_convert_seed.py
    └── test_generate.py
```

### Pipeline en quatre étapes

1. `convert_seed.py` → `seed.json`
2. `promptGenGEPA.py` → `GEPAPrompt.txt`
3. `generate_corpus.py` → `corpus.jsonl`
4. `to_lora_format.py` → `corpus_lora.jsonl` + splits

---

## 3. Fichiers de données

### `intentions.json`

```json
[
  {"intention": "wiki_capture",  "command": "/c",
   "variantes": ["url_avec_tags","url_seule","note_sans_url","avec_directive_simple","avec_ref","avec_fichier"]},
  {"intention": "wiki_ingest",   "command": "/ingest",      "variantes": ["sans_args"]},
  {"intention": "wiki_status",   "command": "/wiki-status", "variantes": ["sans_args"]},
  {"intention": "wiki_query",    "command": "/q",           "variantes": ["question_courte","question_longue"]},
  {"intention": "source_read",   "command": "/source",      "variantes": ["url_seule","url_avec_consigne"]},
  {"intention": "gog_mail",      "command": "/mail",        "variantes": ["envoi","lecture","recherche"]},
  {"intention": "gog_calendar",  "command": "/agenda",      "variantes": ["creation","lecture","suppression"]},
  {"intention": "gog_drive",     "command": "/drive",       "variantes": ["recherche","liste","partage"]},
  {"intention": "meta_assistant","command": "/help",        "variantes": ["sans_args"]},
  {"intention": "out_of_scope",  "command": null,           "variantes": ["action_impossible"]}
]
```

### `registres.json`

```json
["formel", "familier", "télégraphique", "poli", "abrégé"]
```

### `seed.json` — format par entrée

```json
{"text": "garde cet article pour moi : https://example.com #ia",
 "intention": "wiki_capture",
 "action": {"command": "/c", "args": "https://example.com #ia"}}
```

Conventions :
- `out_of_scope` : `"action": {"command": null, "args": ""}`
- Intentions sans args (`wiki_ingest`, `wiki_status`, `meta_assistant`) : `"args": ""`
- Directives `/c` incluses : `@simple`, `file:<path>`, `ref:<slug>`

`convert_seed.py` produit `seed.json` depuis `corpus-intentions-seed.md` par extraction regex (URL, tags, texte résiduel) et ajoute 3 exemples par directive (`@simple`, `file:`, `ref:`), soit 9 exemples directives écrits à la main.

---

## 4. Signatures DSPy

### Générateur

```python
class GenerateExample(dspy.Signature):
    """<prompt-init.txt injecté ici — optimisé par GEPA>"""

    intention: str = dspy.InputField(desc="Intention Tiron à illustrer (ex: wiki_capture)")
    registre:  str = dspy.InputField(desc="Registre du message: formel, familier, télégraphique, poli, abrégé")
    variante:  str = dspy.InputField(desc="Type de contenu attendu (ex: url_avec_tags, question_courte…)")

    text:    str = dspy.OutputField(desc="Message utilisateur réaliste en français")
    command: str = dspy.OutputField(desc="Commande Tiron (/c, /ingest, /q, /source, /mail, /agenda, /drive, /help) ou null")
    args:    str = dspy.OutputField(desc="Arguments bruts de la commande (chaîne vide si pas d'args)")
```

### Évaluateur (métrique GEPA — hybride règles + LLM)

Pour éviter le biais "juge et partie" (DeepSeek génère ET évalue), la métrique est divisée en deux critères indépendants :

**Critère 1 — validité structurelle (règle-based, sans LLM) :**
- La commande est dans la liste des commandes connues ou vaut `null` → 0.5 pt
- Si l'intention requiert des args (`wiki_capture`, `wiki_query`, `source_read`, `gog_mail`, `gog_calendar`, `gog_drive`), `args` est non vide → 0.5 pt
- Score : 0 ou 0.5 ou 1

**Critère 2 — réalisme (DeepSeek) :**

```python
class EvalRealisme(dspy.Signature):
    """Ce message ressemble-t-il à une vraie requête utilisateur adressée à un assistant ?
    Répondre avec un entier 1..5 uniquement, sans commentaire."""

    text: str = dspy.InputField(desc="Message utilisateur généré")
    score: int = dspy.OutputField(desc="Entier 1..5")
```

**Score final :** `0.5 × réalisme/5 + 0.5 × validité_structurelle`

DeepSeek ne juge que le réalisme (critère subjectif). La validité structurelle, critère objectif et décisif pour un routeur, est vérifiée hors LLM — ce qui élimine la circularité sur la moitié du score.

---

## 5. Format de sortie LoRA

Format **ChatML** (phi-4-mini-instruct, unsloth, axolotl, LLaMA-Factory).

### Entrée type

```json
{"messages": [
  {"role": "system",
   "content": "Routeur de commandes Tiron. Pour chaque message, répondre uniquement avec un objet JSON : {\"command\": \"/commande\" ou null, \"args\": \"arguments bruts ou chaîne vide\"}."},
  {"role": "user",
   "content": "garde cet article pour moi : https://example.com #ia"},
  {"role": "assistant",
   "content": "{\"command\": \"/c\", \"args\": \"https://example.com #ia\"}"}
]}
```

Le system prompt est **impératif neutre** (sans pronom personnel) pour rester cohérent quel que soit le registre du message utilisateur.

### Cas `out_of_scope`

```json
{"role": "assistant", "content": "{\"command\": null, \"args\": \"\"}"}
```

### Fichiers produits par `to_lora_format.py`

- `corpus_lora.jsonl` — corpus complet
- `corpus_lora_train.jsonl` — 90 %
- `corpus_lora_eval.jsonl` — 10 %

---

## 6. Tests

### Unitaires (`tests/test_convert_seed.py`)

- Sortie `seed.json` bien formée pour chaque intention
- Cas limites : intention sans args, `out_of_scope`, directives `@simple` / `file:` / `ref:`
- Aucune dépendance réseau

### Intégration (`tests/test_generate.py`)

- `GenerateExample` produit les trois champs (`text`, `command`, `args`) avec LM mocké
- `to_lora_format.py` produit du ChatML valide et split 90/10 cohérent
- 3 exemples générés par appel LM réel en CI

### Validation manuelle (hors pipeline automatique)

```bash
python inspect_corpus.py --sample 20
```

Vérifier : distribution équilibrée des intentions, cohérence `(text, command)`, diversité des registres. Étape humaine obligatoire avant lancement du fine-tuning.

---

## 7. Dépendances et configuration

- `DEEPSEEK_API_KEY` dans l'environnement (même var que `gen_corpus_gepa_codex`)
- `dspy` (version compatible GEPA)
- Python 3.9+
- Modèles : `openai/deepseek-chat` (générateur + évaluateur)

Le répertoire `gen_corpus/` est indépendant du reste de Secretarius (pas d'import depuis `Wiki_LM/`). Il lit `../corpus-intentions-seed.md` uniquement lors de `convert_seed.py`.
