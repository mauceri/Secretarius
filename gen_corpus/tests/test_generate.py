from pathlib import Path
from unittest.mock import MagicMock
import sys
import json
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_generate_one_structure():
    # command est désormais imposé par l'appelant (intentions.json), pas
    # deviné par le modèle — voir la note dans generate_one(). result.command
    # n'est donc plus lu ; seuls result.text et result.args le sont.
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "garde cet article https://example.com"
    mock_result.args = "https://example.com"
    mock_predict = MagicMock(return_value=mock_result)
    always_ok = lambda text, intention: True

    entry, accepted = generate_one(mock_predict, always_ok, "wiki_capture", "familier", "url_seule", "/c")

    assert accepted is True
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
    mock_result.args = ""
    mock_predict = MagicMock(return_value=mock_result)
    always_ok = lambda text, intention: True

    entry, accepted = generate_one(mock_predict, always_ok, "out_of_scope", "familier", "action_impossible", None)
    assert entry["action"]["command"] is None


def test_generate_one_normalizes_literal_empty_quotes():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "état wiki ?"
    mock_result.args = '""'
    mock_predict = MagicMock(return_value=mock_result)
    always_ok = lambda text, intention: True

    entry, accepted = generate_one(mock_predict, always_ok, "wiki_status", "abrégé", "sans_args", "/wikistatus")
    assert entry["action"]["args"] == ""


def test_generate_one_retries_until_fidelity_ok():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "lis le mail 123"
    mock_result.args = "123"
    mock_predict = MagicMock(return_value=mock_result)
    calls = {"n": 0}

    def flaky_check(text, intention):
        calls["n"] += 1
        return calls["n"] >= 2  # échoue une fois, réussit à la 2e tentative

    entry, accepted = generate_one(mock_predict, flaky_check, "gog_get", "familier",
                                    "avec_id_explicite", "/lire", max_attempts=3)
    assert accepted is True
    assert calls["n"] == 2
    assert mock_predict.call_count == 2


def test_generate_one_keeps_entry_after_max_attempts_exhausted():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "texte toujours rejeté"
    mock_result.args = ""
    mock_predict = MagicMock(return_value=mock_result)
    never_ok = lambda text, intention: False

    entry, accepted = generate_one(mock_predict, never_ok, "gog_get", "familier",
                                    "avec_id_explicite", "/lire", max_attempts=3)
    assert accepted is False
    assert entry["text"] == "texte toujours rejeté"
    assert mock_predict.call_count == 3


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
