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


def test_seed_cron_varie():
    import json, re
    seed = json.loads((Path(__file__).parent.parent / "seed.json").read_text(encoding="utf-8"))
    cron = [e["text"].lower() for e in seed if e["intention"] == "cron_schedule"]
    assert len(cron) >= 30
    for marqueur in ["lundi", "demain", "dans "]:
        assert sum(marqueur in t for t in cron) <= len(cron) * 0.2, marqueur
    assert sum(bool(re.search(r"\b(sms|whatsapp|telegram|obsidian)\b", t)) for t in cron) >= 6


def test_intentions_sans_argument_sans_sujet():
    # 2026-10-05 : un sujet imposé à /kbupdate donnait « maj la kb pr mon rdv médical »
    # (21/24 exemples infidèles).
    from chaine import SANS_ARGUMENT, tirer_contraintes
    assert {"wiki_kb_update", "wiki_ingest", "wiki_status", "wiki_tags", "wiki_reread",
            "gog_connect", "gog_inbox"} <= SANS_ARGUMENT
    for intention in SANS_ARGUMENT:
        c = tirer_contraintes(intention, random.Random(0))
        assert "sujet :" not in c and "longueur :" in c
    assert "sujet :" in tirer_contraintes("wiki_query", random.Random(0))
