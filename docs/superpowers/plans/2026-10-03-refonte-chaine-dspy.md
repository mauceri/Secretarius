# Refonte de la chaîne de génération de corpus (DSPy/GEPA) — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** produire des exemples de corpus variés et fidèles, en utilisant DSPy/GEPA comme prévu : retour écrit pour la réflexion, modèles séparés, exemples de départ réellement montrés au générateur, diversité imposée, programme optimisé transmis tel quel à la génération.

**Contexte :** tâche 13 du plan cron (`docs/superpowers/plans/2026-10-02-cron-tiron.md`). Le lot de 250 exemples `cron_schedule` généré le 2026-10-02 est inutilisable : 134/250 parlent du lundi, presque toujours à 8h ou 9h. Le diagnostic du 2026-10-03 a relevé cinq défauts dans la chaîne. Ils existaient avant le cron et touchent toutes les intentions :

| # | Défaut | Où |
|---|---|---|
| D1 | Les exemples de départ ne servent presque à rien. La métrique ne lit jamais le texte de référence, et `generate_corpus.py` ne charge pas `seed.json`. | `promptGenGEPA.py:build_trainset`, `metric` ; `generate_corpus.py` |
| D2 | La métrique renvoie un simple nombre. La réflexion de GEPA n'a aucun retour écrit (« why ») sur lequel s'appuyer, et toutes ses propositions ont été rejetées : le prompt initial est resté le meilleur. | `promptGenGEPA.py:metric` |
| D3 | Un seul modèle sert à la fois de générateur évalué et de modèle de réflexion, à température 1,0. Les scores sont trop bruités pour départager les candidats. | `promptGenGEPA.py:configure_lm` |
| D4 | Rien n'impose la diversité. On répète le même appel avec les trois mêmes entrées, et le modèle retombe sur son mode le plus probable. | `generate_corpus.py:main`, `generate_one` |
| D5 | Seul le texte d'instruction passe de GEPA à la génération, extrait par heuristique puis collé sur une autre signature (descriptions de champs différentes). Le programme compilé n'est jamais enregistré. | `_extract_best_prompt`, `generate_corpus.py:_build_signature` |

**Architecture :** un module partagé `gen_corpus/chaine.py` porte ce que les deux scripts doivent avoir en commun : signature, descriptions des intentions, juges, grille de contraintes, tirage des exemples montrés, détection des quasi-doublons. `promptGenGEPA.py` optimise le programme et l'enregistre (`gepa_program.json`) ; `generate_corpus.py` recharge exactement ce programme.

**Tech Stack:** Python 3.12, DSPy 3.2.1 (vérifié : `Predict` accepte `demos=` à l'appel, `predict.py:142` ; GEPA attend `dspy.Prediction(score, feedback)`, `teleprompt/gepa/gepa.py:48` ; `Module.save`/`load`), DeepSeek (`deepseek-chat`), pytest dans `gen_corpus/.venv`.

## Global Constraints

- Tests : `cd ~/Secretarius && gen_corpus/.venv/bin/python -m pytest gen_corpus/tests -q` (dspy n'existe que dans ce venv).
- La commande reste imposée par l'intention (`intentions.json`), jamais choisie par le modèle (décision existante, `generate_one`).
- Les juges ne voient jamais le prompt optimisé (décision existante : le juge ne dépend pas de ce qu'il évalue).
- Aucun appel DeepSeek dans les tests. Les exécutions réelles (tâche 5) attendent l'accord de l'utilisateur.
- Compatibilité : `generate_one(predict, fidelity_check, intention, registre, variante, command)` reste appelable tel quel (tests existants). Les nouveaux paramètres sont nommés, avec des valeurs par défaut.

## Critères d'acceptation (mesurés à la tâche 5, sur 250 exemples `cron_schedule`)

- Aucune forme horaire (lundi, « demain », « dans … », etc.) ne dépasse 25 % des exemples.
- Quasi-doublons gardés : moins de 2 %.
- Longueur moyenne : 15 mots au plus.
- Au plus 5 % commencent par « Bonjour » ou « Tiron ».

## Review Focus

- **Exemples montrés sans champ `contraintes`** (les exemples de départ n'en ont pas) : l'adaptateur DSPy doit les formater sans planter → test dans la tâche 1.
- **Intention sans aucun exemple de départ** : le tirage renvoie une liste vide et la génération continue → test dans la tâche 1.
- **Juge qui renvoie un texte non numérique** : score par défaut et retour écrit qui le signale, sans exception → test dans la tâche 2.
- **`gepa_program.json` absent** : la génération retombe sur `GEPAPrompt.txt`, puis `prompt-init.txt`, comme aujourd'hui → test dans la tâche 3.
- **Quasi-doublon après le nombre maximal d'essais** : l'exemple est écarté, pas gardé, et le rapport le compte → test dans la tâche 3.

---

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `gen_corpus/chaine.py` (créé) | signature `GenerateExample` (+ entrée `contraintes`), `INTENTION_DESCRIPTIONS` (unique), juges avec `raison`, `GRILLES`, `tirer_contraintes`, `tirer_demos`, `est_quasi_doublon`, `construire_feedback` |
| `gen_corpus/promptGenGEPA.py` | générateur et réflexion séparés, trainset avec contraintes, métrique score + retour écrit, `compiled.save("gepa_program.json")` |
| `gen_corpus/generate_corpus.py` | recharge `gepa_program.json`, contraintes et exemples montrés à chaque appel, rejet des quasi-doublons, rapport de diversité |
| `gen_corpus/seed.json` | 12 exemples `cron_schedule` remplacés par 30 exemples variés |
| `gen_corpus/tests/test_chaine.py`, `test_gepa_metric.py`, `test_generate.py` | tests |

---

### Task 1: `gen_corpus/chaine.py` — module partagé

**Files:**
- Create: `gen_corpus/chaine.py`
- Test: `gen_corpus/tests/test_chaine.py`

**Interfaces:**
- Produces:
  - `INTENTION_DESCRIPTIONS: dict[str, str]` (déplacé depuis les deux scripts, contenu identique à celui de `promptGenGEPA.py`, `cron_schedule` compris).
  - `build_signature(prompt_text: str) -> type[dspy.Signature]` : entrées `intention`, `registre`, `variante`, `contraintes` ; sorties `text`, `command`, `args`.
  - `EvalRealisme`, `EvalFidelite` : sorties `score` et `raison` (une phrase).
  - `GRILLES: dict[str, dict[str, list[str]]]` (clé `"*"` = grille générique ; `"cron_schedule"` = grille propre).
  - `tirer_contraintes(intention: str, rng: random.Random) -> str`.
  - `tirer_demos(seed: list[dict], intention: str, k: int, rng: random.Random) -> list[dspy.Example]`.
  - `est_quasi_doublon(text: str, deja_vus: list[str], seuil: float = 0.8) -> bool`.

- [ ] **Step 1: Écrire les tests**

```python
# gen_corpus/tests/test_chaine.py
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = [
    {"text": "rappelle-moi jeudi 18h de sortir le bac vert", "intention": "cron_schedule", "registre": "familier",
     "variante": "rappel_ponctuel_date", "action": {"command": "/cron", "args": "x"}},
    {"text": "/c https://a.fr", "intention": "wiki_capture", "registre": "abrégé",
     "variante": "url_seule", "action": {"command": "/c", "args": "https://a.fr"}},
]


def test_contraintes_cron_deterministes_et_completes():
    from chaine import tirer_contraintes
    a = tirer_contraintes("cron_schedule", random.Random(3))
    assert a == tirer_contraintes("cron_schedule", random.Random(3))
    for cle in ("moment :", "heure :", "canal :", "type de tâche :", "sujet :", "longueur :"):
        assert cle in a


def test_contraintes_cron_varient():
    from chaine import tirer_contraintes
    rng = random.Random(0)
    moments = {tirer_contraintes("cron_schedule", rng).split(" ; ")[0] for _ in range(200)}
    assert len(moments) >= 8


def test_contraintes_generiques_pour_les_autres_intentions():
    from chaine import tirer_contraintes
    c = tirer_contraintes("wiki_query", random.Random(1))
    assert "sujet :" in c and "moment :" not in c


def test_demos_meme_intention_seulement_et_k_borne():
    from chaine import tirer_demos
    demos = tirer_demos(SEED, "cron_schedule", 4, random.Random(0))
    assert len(demos) == 1 and demos[0].text.startswith("rappelle-moi jeudi")
    assert demos[0].contraintes == "aucune"


def test_demos_intention_sans_exemple():
    from chaine import tirer_demos
    assert tirer_demos(SEED, "wiki_tags", 4, random.Random(0)) == []


def test_quasi_doublon():
    from chaine import est_quasi_doublon
    vus = ["Rappelle-moi chaque lundi à 8h de payer le loyer."]
    assert est_quasi_doublon("rappelle moi chaque lundi a 8h de payer le loyer", vus)
    assert not est_quasi_doublon("tous les soirs à 21h résume le wiki", vus)


def test_signature_a_l_entree_contraintes():
    from chaine import build_signature
    sig = build_signature("Consigne de test assez longue pour être une instruction.")
    assert "contraintes" in sig.input_fields and set(sig.output_fields) == {"text", "command", "args"}


def test_demo_sans_contraintes_formatable():
    import dspy
    from chaine import build_signature, tirer_demos
    from dspy.adapters import ChatAdapter
    sig = build_signature("Consigne de test assez longue pour être une instruction.")
    demos = tirer_demos(SEED, "cron_schedule", 1, random.Random(0))
    msgs = ChatAdapter().format(sig, demos, {"intention": "cron_schedule", "registre": "poli",
                                             "variante": "avec_canal", "contraintes": "moment : demain"})
    assert any("rappelle-moi jeudi" in m["content"] for m in msgs)
```

- [ ] **Step 2: Vérifier l'échec** — `gen_corpus/.venv/bin/python -m pytest gen_corpus/tests/test_chaine.py -q` → FAIL (`No module named 'chaine'`).

- [ ] **Step 3: Implémenter**

```python
# gen_corpus/chaine.py
"""Éléments partagés par promptGenGEPA.py (optimisation) et generate_corpus.py
(génération) : un seul exemplaire de la signature, des descriptions, des juges
et de la grille de diversité. Cf. docs/superpowers/plans/2026-10-03-refonte-chaine-dspy.md."""
from __future__ import annotations

import random
import re

import dspy

INTENTION_DESCRIPTIONS = {
    # … recopier ici, à l'identique, le dictionnaire de promptGenGEPA.py
    # (18 entrées, cron_schedule et out_of_scope compris), puis le supprimer
    # des deux scripts au profit de `from chaine import INTENTION_DESCRIPTIONS`.
}


def build_signature(prompt_text: str):
    class GenerateExample(dspy.Signature):
        __doc__ = prompt_text
        intention: str = dspy.InputField(desc="Intention Tiron à illustrer (ex: wiki_capture)")
        registre: str = dspy.InputField(desc="Registre du message : formel, familier, télégraphique, poli, abrégé")
        variante: str = dspy.InputField(desc="Type de variante (ex: url_avec_tags, question_courte, sans_args…)")
        contraintes: str = dspy.InputField(desc="Contraintes à respecter impérativement (moment, sujet, longueur…)")
        text: str = dspy.OutputField(desc="Message utilisateur réaliste en français")
        command: str = dspy.OutputField(desc="Commande Tiron ou null")
        args: str = dspy.OutputField(desc="Arguments bruts de la commande (chaîne vide si pas d'args)")
    return GenerateExample


class EvalRealisme(dspy.Signature):
    """Ce message ressemble-t-il à une vraie requête tapée par un utilisateur à son
    assistant (brève, naturelle, sans formule de politesse développée) ?"""
    text: str = dspy.InputField(desc="Message utilisateur généré")
    score: int = dspy.OutputField(desc="Entier 1..5")
    raison: str = dspy.OutputField(desc="Une phrase : ce qui le rend peu réaliste, ou « rien »")


class EvalFidelite(dspy.Signature):
    """Ce message correspond-il vraiment à l'intention décrite, et à elle seule
    (pas à une intention voisine ni à une autre commande) ? 1 si sans ambiguïté
    cette intention précise, 0 sinon — y compris si le message est plausible
    pour une AUTRE intention que celle donnée."""
    text: str = dspy.InputField(desc="Message utilisateur généré")
    intention_description: str = dspy.InputField(desc="Description de l'intention que le message doit illustrer")
    score: int = dspy.OutputField(desc="0 ou 1")
    raison: str = dspy.OutputField(desc="Une phrase : pourquoi, ou quelle autre intention il évoque")


_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
         "septembre", "octobre", "novembre", "décembre"]
_SUJETS = ["santé", "factures", "jardin", "famille", "travail", "voiture", "courses", "sport",
           "administratif", "lecture", "cuisine", "animaux", "maison", "études", "voyage",
           "informatique", "musique", "rendez-vous médical", "anniversaire", "wiki personnel"]
GRILLES: dict[str, dict[str, list[str]]] = {
    "*": {
        "sujet": _SUJETS,
        "forme": ["question", "ordre", "phrase nominale", "constat"],
        "longueur": ["très court (≤ 5 mots)", "court (6 à 12 mots)", "moyen (13 à 20 mots)"],
        "frappe": ["sans faute", "une abréviation ou faute de frappe"],
    },
    "cron_schedule": {
        "moment": [f"chaque {j}" for j in _JOURS] + [f"tous les {j}s" for j in _JOURS]
                  + ["tous les jours", "chaque matin", "chaque soir", "en semaine (lundi à vendredi)",
                     "le week-end", "le 1er de chaque mois", "le 15 de chaque mois",
                     "demain", "après-demain", "ce soir", "dans N minutes", "dans N heures",
                     "dans N jours"] + [f"le N {m}" for m in _MOIS],
        "heure": [f"{h}h{m}" for h in range(6, 23) for m in ("", "15", "30", "45")] + ["sans heure"],
        "canal": ["non précisé"] * 4 + ["SMS", "WhatsApp", "Telegram", "Obsidian"],
        "type de tâche": ["rappel"] * 3 + ["commande Tiron (/wikistatus, /tags, /relire, /ingest…)",
                                            "consigne à l'assistant (résumer, préparer, vérifier…)"],
        "sujet": _SUJETS,
        "longueur": ["très court (≤ 5 mots)", "court (6 à 12 mots)", "moyen (13 à 20 mots)"],
    },
}


def tirer_contraintes(intention: str, rng: random.Random) -> str:
    grille = GRILLES.get(intention, GRILLES["*"])
    parts = []
    for cle, valeurs in grille.items():
        v = rng.choice(valeurs).replace("N", str(rng.randint(2, 28)))
        parts.append(f"{cle} : {v}")
    return " ; ".join(parts)


def tirer_demos(seed: list[dict], intention: str, k: int, rng: random.Random) -> list:
    pool = [ex for ex in seed if ex.get("intention") == intention]
    choisis = rng.sample(pool, min(k, len(pool)))
    return [dspy.Example(intention=ex["intention"], registre=ex.get("registre", "poli"),
                         variante=ex.get("variante", "sans_args"), contraintes="aucune",
                         text=ex["text"], command=ex["action"]["command"] or "null",
                         args=ex["action"]["args"]).with_inputs("intention", "registre", "variante", "contraintes")
            for ex in choisis]


def _mots(t: str) -> set[str]:
    t = re.sub(r"[^\w\s]", " ", t.lower())
    t = t.translate(str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))
    return set(t.split())


def est_quasi_doublon(text: str, deja_vus: list[str], seuil: float = 0.8) -> bool:
    a = _mots(text)
    for v in deja_vus:
        b = _mots(v)
        if a and b and len(a & b) / len(a | b) >= seuil:
            return True
    return False
```

Le dictionnaire `INTENTION_DESCRIPTIONS` est recopié tel quel depuis `promptGenGEPA.py` au moment de l'implémentation : c'est un déplacement, pas un nouveau contenu. Le commentaire de l'extrait ci-dessus n'est là que pour ne pas dupliquer 18 lignes dans ce plan.

- [ ] **Step 4: Vérifier le succès** — `… pytest gen_corpus/tests/test_chaine.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(corpus): module partagé chaine.py (signature, juges, diversité, quasi-doublons)"`

---

### Task 2: `promptGenGEPA.py` — métrique avec retour écrit, modèles séparés, programme enregistré

**Files:**
- Modify: `gen_corpus/promptGenGEPA.py` (imports, `configure_lm`, `build_example_generator`, `build_trainset`, `make_metric`, `main` ; supprimer `EvalRealisme`, `EvalFidelite`, `INTENTION_DESCRIPTIONS` locaux et `_extract_best_prompt` devenu inutile)
- Test: `gen_corpus/tests/test_gepa_metric.py`

**Interfaces:**
- Consumes: tâche 1.
- Produces: `construire_feedback(text: str, s_struct: float, s_real: float, raison_real: str, s_fidele: float, raison_fidele: str) -> str` ; `make_metric(eval_lm)` renvoie une fonction qui rend `dspy.Prediction(score=float, feedback=str)` ; `build_example_generator(prompt_text, seed, rng)` dont `forward(intention, registre, variante, contraintes)` passe `demos=tirer_demos(seed, intention, 3, rng)` ; fichier `gepa_program.json` écrit par `main`.

- [ ] **Step 1: Écrire les tests**

```python
# gen_corpus/tests/test_gepa_metric.py
from pathlib import Path
import random
import sys
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_feedback_signale_les_defauts():
    from promptGenGEPA import construire_feedback
    fb = construire_feedback("Bonjour Tiron, pourriez-vous avoir l'amabilité de bien vouloir me rappeler "
                             "chaque lundi à huit heures précises de payer le loyer de l'appartement svp",
                             1.0, 0.4, "trop poli", 0.0, "évoque /q")
    assert "trop poli" in fb and "évoque /q" in fb
    assert "Bonjour" in fb and "mots" in fb


def test_feedback_parfait():
    from promptGenGEPA import construire_feedback
    assert construire_feedback("rappel loyer lundi 8h", 1.0, 1.0, "rien", 1.0, "ok").startswith("Parfait")


def test_metrique_renvoie_score_et_feedback(monkeypatch):
    import dspy
    import promptGenGEPA as g
    real = MagicMock(return_value=dspy.Prediction(score=5, raison="rien"))
    fid = MagicMock(return_value=dspy.Prediction(score=0, raison="évoque /q"))
    monkeypatch.setattr(g.dspy, "Predict", MagicMock(side_effect=[real, fid]))
    metric = g.make_metric(MagicMock())
    gold = dspy.Example(intention="cron_schedule")
    out = metric(gold, {"text": "rappel loyer lundi 8h", "command": "/cron", "args": "x"})
    assert isinstance(out, dspy.Prediction)
    assert out.score == 0.25 * 1.0 + 0.25 * 1.0 + 0.5 * 0.0
    assert "évoque /q" in out.feedback


def test_juge_non_numerique_ne_plante_pas(monkeypatch):
    import dspy
    import promptGenGEPA as g
    real = MagicMock(return_value=dspy.Prediction(score="très", raison="?"))
    fid = MagicMock(return_value=dspy.Prediction(score="oui", raison="?"))
    monkeypatch.setattr(g.dspy, "Predict", MagicMock(side_effect=[real, fid]))
    out = g.make_metric(MagicMock())(dspy.Example(intention="wiki_tags"), {"text": "tags ?", "command": "/tags", "args": ""})
    assert 0.0 <= out.score <= 1.0 and "illisible" in out.feedback


def test_generateur_montre_des_exemples_et_les_contraintes(monkeypatch):
    import promptGenGEPA as g
    seed = [{"text": "rappel jeudi 18h bac vert", "intention": "cron_schedule", "registre": "abrégé",
             "variante": "x", "action": {"command": "/cron", "args": "x"}}]
    gen = g.build_example_generator("Consigne de test assez longue pour être une instruction.", seed, random.Random(0))
    fake = MagicMock(return_value=MagicMock(text="t", command="/cron", args="a"))
    gen.generate = fake
    gen(intention="cron_schedule", registre="poli", variante="x", contraintes="moment : demain")
    kw = fake.call_args.kwargs
    assert kw["contraintes"] == "moment : demain" and len(kw["demos"]) == 1
```

- [ ] **Step 2: Vérifier l'échec** — `… pytest gen_corpus/tests/test_gepa_metric.py -q` → FAIL.

- [ ] **Step 3: Implémenter**

Imports : `from chaine import EvalFidelite, EvalRealisme, INTENTION_DESCRIPTIONS, build_signature, tirer_contraintes, tirer_demos`.

`configure_lm` devient deux fonctions :
```python
def make_lm(model: str, temperature: float, api_base: str, max_tokens: int = 512) -> dspy.LM:
    key = _ensure_key()
    base = os.getenv("DEEPSEEK_API_BASE", api_base)
    return dspy.LM(model=model, api_key=key, api_base=base, model_type="chat",
                   temperature=temperature, max_tokens=max_tokens, cache=False)
```
Dans `main` : générateur évalué à `--student-temperature` (défaut 0.7) via `dspy.settings.configure(lm=make_lm(..., 0.7))` ; réflexion `reflection_lm=make_lm(..., cfg.reflection_temperature)` (défaut 1.0) ; juges `make_lm(cfg.eval_model, 0.0, …, max_tokens=128)` (les juges écrivent désormais une `raison`).

```python
def build_example_generator(prompt_text: str, seed: list[dict], rng: random.Random) -> dspy.Module:
    class ExampleGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(build_signature(prompt_text))

        def forward(self, intention, registre, variante, contraintes):
            r = self.generate(intention=intention, registre=registre, variante=variante,
                              contraintes=contraintes, demos=tirer_demos(seed, intention, 3, rng))
            return {"text": r.text, "intention": intention, "command": r.command,
                    "args": r.args, "registre": registre, "variante": variante}

    return ExampleGenerator()


def build_trainset(seed: list[dict], rng: random.Random) -> list[dspy.Example]:
    return [
        dspy.Example(intention=ex["intention"], registre=ex.get("registre", "poli"),
                     variante=ex.get("variante", "sans_args"),
                     contraintes=tirer_contraintes(ex["intention"], rng),
                     command=ex["action"]["command"] if ex["action"]["command"] is not None else "null",
                     args=ex["action"]["args"]).with_inputs("intention", "registre", "variante", "contraintes")
        for ex in seed
    ]


def construire_feedback(text: str, s_struct: float, s_real: float, raison_real: str,
                        s_fidele: float, raison_fidele: str) -> str:
    remarques = []
    if s_fidele < 1:
        remarques.append(f"Infidèle à l'intention : {raison_fidele}")
    if s_real < 0.8:
        remarques.append(f"Peu réaliste ({s_real * 5:.0f}/5) : {raison_real}")
    if s_struct < 1:
        remarques.append("Commande ou arguments manquants ou inconnus.")
    n = len(text.split())
    if n > 20:
        remarques.append(f"Trop long : {n} mots (viser moins de 20).")
    if text.lower().startswith(("bonjour", "tiron")):
        remarques.append("Commence par « Bonjour » ou « Tiron » : varier l'attaque.")
    return "Parfait." if not remarques else " ".join(remarques)
```

`make_metric` : mêmes juges et mêmes poids qu'aujourd'hui, mais chaque juge renvoie `(score, raison)`. Un score non convertible donne la valeur par défaut actuelle (réalisme 3/5, fidélité 0) et la raison `"réponse du juge illisible"`. La fonction rend `dspy.Prediction(score=0.25 * s_real + 0.25 * s_struct + 0.5 * s_fidele, feedback=construire_feedback(...))`.

Dans `main` : `rng = random.Random(0)` ; `generator = build_example_generator(initial_prompt, seed, rng)` ; `trainset, valset = split_trainset(build_trainset(seed, rng), cfg.valset_size)` ; après `compile` :
```python
    compiled.save("gepa_program.json")
    Path(cfg.gepa_prompt_path).write_text(compiled.generate.signature.instructions, encoding="utf-8")
```
(`GEPAPrompt.txt` reste écrit, pour lecture humaine et comme repli.)

- [ ] **Step 4: Vérifier** — `… pytest gen_corpus/tests -q` → PASS (tests existants compris ; adapter `test_gepa_split.py` seulement si l'import change).

- [ ] **Step 5: Commit** — `git commit -m "fix(gepa): métrique avec retour écrit, modèles séparés, exemples montrés, programme enregistré"`

---

### Task 3: `generate_corpus.py` — programme rechargé, contraintes, quasi-doublons, rapport

**Files:**
- Modify: `gen_corpus/generate_corpus.py`
- Test: `gen_corpus/tests/test_generate.py`

**Interfaces:**
- Consumes: tâches 1 et 2 (`gepa_program.json`, `build_example_generator`).
- Produces:
  - `charger_programme(prompt_path, fallback, program_path, seed, rng) -> dspy.Module` ;
  - `generate_one(predict, fidelity_check, intention, registre, variante, command, max_attempts=3, *, contraintes="", deja_vus=None) -> tuple[dict | None, bool]`. Elle renvoie `(None, False)` si le texte reste un quasi-doublon après `max_attempts` essais. L'entrée produite contient `contraintes` ;
  - `rapport_diversite(entries: list[dict]) -> str`.

- [ ] **Step 1: Écrire les tests** (ajouter à `gen_corpus/tests/test_generate.py`)

```python
def test_generate_one_transmet_les_contraintes_et_les_garde():
    from generate_corpus import generate_one
    res = MagicMock(text="rappel jeudi 18h bac vert", args="x")
    pred = MagicMock(return_value=res)
    entry, ok = generate_one(pred, lambda t, i: True, "cron_schedule", "abrégé", "x", "/cron",
                             contraintes="moment : chaque jeudi")
    assert pred.call_args.kwargs["contraintes"] == "moment : chaque jeudi"
    assert entry["contraintes"] == "moment : chaque jeudi"


def test_generate_one_ecarte_un_quasi_doublon_persistant():
    from generate_corpus import generate_one
    pred = MagicMock(return_value=MagicMock(text="rappelle-moi chaque lundi à 8h de payer le loyer", args=""))
    entry, ok = generate_one(pred, lambda t, i: True, "cron_schedule", "familier", "x", "/cron",
                             deja_vus=["Rappelle-moi chaque lundi à 8h de payer le loyer."])
    assert entry is None and ok is False and pred.call_count == 3


def test_charger_programme_sans_fichier_retombe_sur_le_prompt(tmp_path):
    import random
    from generate_corpus import charger_programme
    (tmp_path / "prompt-init.txt").write_text("Consigne initiale assez longue pour une instruction.", encoding="utf-8")
    prog = charger_programme(str(tmp_path / "GEPAPrompt.txt"), str(tmp_path / "prompt-init.txt"),
                             str(tmp_path / "gepa_program.json"), [], random.Random(0))
    assert prog.generate.signature.instructions.startswith("Consigne initiale")


def test_rapport_diversite_compte_les_moments():
    from generate_corpus import rapport_diversite
    entries = [{"intention": "cron_schedule", "text": t} for t in
               ["rappel lundi 8h", "chaque lundi 9h x", "demain 7h y", "dans 20 min z"]]
    r = rapport_diversite(entries)
    assert "lundi : 2/4 (50 %)" in r
```

Les tests existants de `generate_one` doivent passer sans modification.

- [ ] **Step 2: Vérifier l'échec** — `… pytest gen_corpus/tests/test_generate.py -q` → FAIL sur les 4 nouveaux tests.

- [ ] **Step 3: Implémenter**

- Supprimer `_build_signature`, `EvalFidelite` et `INTENTION_DESCRIPTIONS` locaux : on importe ceux de `chaine` (même contenu).
- `charger_programme` : construit `build_example_generator(texte_du_prompt, seed, rng)` (importé de `promptGenGEPA` ; ce module n'appelle DeepSeek qu'à l'exécution, pas à l'import). Le texte vient de `GEPAPrompt.txt` s'il existe, sinon de `prompt-init.txt`. Puis `prog.load(program_path)` si `gepa_program.json` existe.
- `generate_one` : appelle `predict(intention=…, registre=…, variante=…, contraintes=contraintes)`. Un texte dont `est_quasi_doublon(text, deja_vus)` est vrai compte comme un essai raté. Si tous les essais sont ratés par quasi-doublon, la fonction renvoie `(None, False)`.
- `main` :
  - charge `seed.json` et `rng = random.Random()` ;
  - à chaque exemple, tire `contraintes = tirer_contraintes(intention, rng)` ;
  - tient `deja_vus` par intention ;
  - saute les `None` et les compte (`ecartes_doublons`) ;
  - à la fin, affiche `rapport_diversite(toutes_les_entrées)` et le nombre d'exemples écartés.
- `rapport_diversite` : pour chaque intention, compte les marqueurs de moment (`lundi` … `dimanche`, `tous les jours`, `chaque matin`, `soir`, `demain`, `dans `, date `le \d`, `mois`), la longueur moyenne, les débuts « Bonjour »/« Tiron » et les doublons exacts. Format d'une ligne : `"{marqueur} : {n}/{total} ({pct:.0f} %)"`.

Le prédicteur passé à `generate_one` est `prog`, le module entier (`prog(intention=…, …)`), qui rend un dict. Pour un accès uniforme, `generate_one` lit `result["text"]` et `result["args"]` quand le résultat est un dict, et `result.text`/`result.args` sinon (cas des tests existants, qui passent un `MagicMock`).

- [ ] **Step 3b: Sondage en cours de génération (demande de l'utilisateur, 2026-10-05)**

Test d'abord (dans `test_generate.py`) :
```python
def test_sondage_alerte_sur_derive_de_moment():
    import random
    from generate_corpus import sonder
    entries = [{"intention": "cron_schedule", "text": f"rappel lundi {h}h x{h}"} for h in range(30)] + \
              [{"intention": "cron_schedule", "text": f"demain {h}h y{h}"} for h in range(20)]
    texte, alerte = sonder(entries, rejets_fidelite=0, random_=random.Random(0))
    assert "lundi : 30/50" in texte and texte.count("\n- ") == 5
    assert alerte and "lundi" in alerte


def test_sondage_alerte_sur_rejets_de_fidelite():
    import random
    from generate_corpus import sonder
    entries = [{"intention": "wiki_tags", "text": f"tags {i}"} for i in range(50)]
    _, alerte = sonder(entries, rejets_fidelite=20, random_=random.Random(0))
    assert alerte and "fidélité" in alerte


def test_sondage_sans_alerte():
    import random
    from generate_corpus import sonder
    entries = [{"intention": "cron_schedule", "text": t} for t in
               ["rappel lundi 8h a", "demain 9h b", "dans 20 min c", "tous les jours 7h d", "le 3 mars e"] * 10]
    assert sonder(entries, rejets_fidelite=2, random_=random.Random(0))[1] is None
```
Implémentation : `sonder(entries, rejets_fidelite, random_, k=5) -> tuple[str, str | None]`. Elle renvoie `rapport_diversite(entries)` suivi de `k` textes tirés au hasard (lignes `"\n- …"`), et une alerte si :
- un marqueur de moment dépasse 40 % des exemples `cron_schedule` (au moins 50 exemples) ;
- ou les exemples gardés malgré un échec de fidélité dépassent 30 % du total.

Dans `main`, l'option `--sondage-every` (défaut 50) déclenche l'affichage tous les N exemples. Une alerte arrête la génération proprement : le tampon est écrit, et le message affiché est `ARRÊT : dérive — <alerte>`. On analyse alors avant toute relance.

Note : l'ancien juge de ressemblance au texte de départ (`~/gen_corpus_gepa_codex/promptGenGEPA.py:376`) corrigeait, selon l'utilisateur, une dérive constatée. Si le sondage révèle une dérive de fond (hors sujet, ton), c'est la première piste à rétablir.

- [ ] **Step 4: Vérifier** — `… pytest gen_corpus/tests -q` → PASS (tous, anciens compris).

- [ ] **Step 5: Commit** — `git commit -m "fix(corpus): génération avec le programme GEPA, contraintes de diversité, rejet des quasi-doublons"`

---

### Task 4: exemples de départ `cron_schedule` — 30 exemples variés

**Files:**
- Modify: `gen_corpus/seed.json` (remplacer les 12 exemples `cron_schedule` ajoutés le 2026-10-02)
- Test: `gen_corpus/tests/test_chaine.py`

- [ ] **Step 1: Test de diversité des exemples de départ**

```python
def test_seed_cron_varie():
    import json, re
    seed = json.loads((Path(__file__).parent.parent / "seed.json").read_text(encoding="utf-8"))
    cron = [e["text"].lower() for e in seed if e["intention"] == "cron_schedule"]
    assert len(cron) >= 30
    for marqueur in ["lundi", "demain", "dans "]:
        assert sum(marqueur in t for t in cron) <= len(cron) * 0.2, marqueur
    assert sum(bool(re.search(r"\b(sms|whatsapp|telegram|obsidian)\b", t)) for t in cron) >= 6
```

- [ ] **Step 2: Vérifier l'échec** (12 exemples seulement) → FAIL.

- [ ] **Step 3: Remplacer les 12 exemples par ces 30** (format existant de `seed.json`, `action.args` en syntaxe normalisée, dates rapportées à octobre 2026) :

| texte | registre | variante | args |
|---|---|---|---|
| rappelle-moi jeudi 18h de sortir le bac vert | familier | rappel_ponctuel_date | `add 2026-10-08T18:00 rappel sortir le bac vert` |
| tous les soirs 21h30 : pilulier | télégraphique | rappel_recurrent | `add "30 21 * * *" rappel pilulier` |
| Pourriez-vous me rappeler le 3 novembre à 14h15 le contrôle technique ? | formel | rappel_ponctuel_date | `add 2026-11-03T14:15 rappel le contrôle technique` |
| dans 45 min rappelle-moi le four | abrégé | rappel_relatif | `add 45m rappel le four` |
| chaque 1er du mois à 9h, rappel loyer | télégraphique | rappel_recurrent | `add "0 9 1 * *" rappel loyer` |
| en semaine à 7h10 rappelle-moi de prendre le bus | familier | rappel_recurrent | `add "10 7 * * 1-5" rappel prendre le bus` |
| samedi matin 10h, penser au marché | abrégé | rappel_ponctuel_date | `add 2026-10-10T10:00 rappel penser au marché` |
| Merci de me rappeler après-demain à 16h d'appeler ma sœur. | poli | rappel_ponctuel_date | `add 2026-10-05T16:00 rappel appeler ma sœur` |
| tous les dimanches 19h /wikistatus | télégraphique | commande_recurrente | `add "0 19 * * 0" commande /wikistatus` |
| lance /ingest chaque nuit à 2h30 | abrégé | commande_recurrente | `add "30 2 * * *" commande /ingest` |
| chaque mercredi midi, donne-moi la page à relire | familier | commande_recurrente | `add "0 12 * * 3" commande /relire` |
| le vendredi à 17h45 fais-moi un point sur les nouveautés du wiki | familier | consigne_recurrente | `add "45 17 * * 5" consigne point sur les nouveautés du wiki` |
| Chaque matin à 6h30, préparez-moi un résumé de mes captures de la veille. | formel | consigne_recurrente | `add "30 6 * * *" consigne résumé des captures de la veille` |
| tous les 15 du mois vérifie les factures en attente | télégraphique | consigne_recurrente | `add "0 9 15 * *" consigne vérifie les factures en attente` |
| par SMS dans 2h : rappeler le plombier | abrégé | avec_canal | `add 2h sms rappel rappeler le plombier` |
| sur WhatsApp mardi 8h, réunion parents-profs | télégraphique | avec_canal | `add 2026-10-06T08:00 whatsapp rappel réunion parents-profs` |
| note dans Obsidian ce soir 20h : idées pour l'article | abrégé | avec_canal | `add 2026-10-04T20:00 obsidian rappel idées pour l'article` |
| envoie-moi un SMS chaque lundi 7h45 pour le sport | familier | avec_canal | `add "45 7 * * 1" sms rappel le sport` |
| Pouvez-vous me prévenir sur Telegram dans 3 jours pour renouveler l'ordonnance ? | poli | avec_canal | `add 3d telegram rappel renouveler l'ordonnance` |
| le week-end à 11h rappelle-moi d'arroser | familier | rappel_recurrent | `add "0 11 * * 6,0" rappel arroser` |
| dans 10 minutes : thé | télégraphique | rappel_relatif | `add 10m rappel thé` |
| rappel 24 décembre 18h cadeaux à emballer | abrégé | rappel_ponctuel_date | `add 2026-12-24T18:00 rappel cadeaux à emballer` |
| ce soir 22h, couper le chauffage | abrégé | rappel_ponctuel_date | `add 2026-10-03T22:00 rappel couper le chauffage` |
| Je souhaiterais un rappel tous les mardis à 13h pour la réunion d'équipe. | formel | rappel_recurrent | `add "0 13 * * 2" rappel réunion d'équipe` |
| programme /tags tous les mois le 28 à 8h | télégraphique | commande_recurrente | `add "0 8 28 * *" commande /tags` |
| dans 1h30 rappelle moi de rappeler Paul | abrégé | rappel_relatif | `add 90m rappel rappeler Paul` |
| demain 6h50 réveil train | télégraphique | rappel_ponctuel_date | `add 2026-10-04T06:50 rappel train` |
| tous les jeudis soir à 20h, résume ce que j'ai lu dans la semaine | familier | consigne_recurrente | `add "0 20 * * 4" consigne résume ce que j'ai lu dans la semaine` |
| Merci de m'envoyer chaque soir à 21h par SMS la liste des tâches du lendemain. | poli | avec_canal | `add "0 21 * * *" sms consigne liste des tâches du lendemain` |
| rappelle-moi le 2 janvier à 9h de faire les vœux | familier | rappel_ponctuel_date | `add 2027-01-02T09:00 rappel faire les vœux` |

- [ ] **Step 4: Vérifier** — `… pytest gen_corpus/tests -q` → PASS ; `python3 -c "import json; json.load(open('gen_corpus/seed.json'))"`.

- [ ] **Step 5: Commit** — `git commit -m "feat(corpus): 30 exemples de départ cron variés"`

---

### Task 5: exécution réelle et contrôle (**accord de l'utilisateur requis**, appels DeepSeek)

- [ ] **Step 1: [accord] GEPA** — `cd gen_corpus && .venv/bin/python promptGenGEPA.py --valset-size 40`. Dans `gepa_llm_calls.log`, vérifier que la réflexion reçoit des retours écrits (« Infidèle… », « Trop long… »), et noter les scores de base et final. `gepa_program.json` doit exister.
- [ ] **Step 2: [accord] Génération cron** — fichier d'intentions réduit à `cron_schedule`, puis `--count 250 --output corpus_cron.jsonl`. Lire le rapport de diversité.
- [ ] **Step 3: Contrôle** — appliquer les critères d'acceptation du haut de ce plan, et relire 20 exemples tirés au hasard. Hors critères : analyser, corriger la grille ou le prompt, puis relancer seulement l'étape 2, après accord.
- [ ] **Step 4: Décision avec l'utilisateur** — régénérer ou non tout le corpus (3 200 exemples produits par l'ancienne chaîne, coût DeepSeek plus élevé) avant le réentraînement de Laya. Reprendre ensuite le plan cron, tâche 13 à partir de l'étape 3.
