import json
from collections import Counter

import pytest

from gen_corpus.to_laya_format import _to_laya_row, reformat


def _ligne(text: str, command) -> str:
    return json.dumps({"messages": [
        {"role": "system", "content": "ignoré"},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"command": command, "args": ""})},
    ]})


def test_ligne_wiki_porte_deux_questions():
    r = _to_laya_row("c koi SPLADE ?", "/q")
    assert set(r["questions"]) == {"domaine", "commande_wiki"}
    assert r["answers"] == {"domaine": "wiki", "commande_wiki": "/q"}


def test_ligne_google_porte_deux_questions():
    r = _to_laya_row("lis le mail 18ab3f2", "/lire")
    assert r["answers"] == {"domaine": "google", "commande_google": "/lire"}


def test_ligne_planification_web_et_aucune_une_seule_question():
    assert _to_laya_row("rappel 8h", "/cron")["answers"] == {"domaine": "planification"}
    assert _to_laya_row("lis https://x.fr", "/source")["answers"] == {"domaine": "web"}
    assert _to_laya_row("allume la lumière", "aucune")["answers"] == {"domaine": "aucune"}


def test_commande_hors_mapping_rejetee():
    with pytest.raises(KeyError):
        _to_laya_row("cherche X", "/r")


def test_commande_null_devient_aucune(tmp_path):
    src = tmp_path / "c.jsonl"
    src.write_text(_ligne("il fait beau aujourd'hui", None) + "\n", encoding="utf-8")
    reformat(src, tmp_path / "t.jsonl", tmp_path / "v.jsonl")
    rows = [json.loads(l) for f in ("t.jsonl", "v.jsonl") for l in open(tmp_path / f) if l.strip()]
    assert rows[0]["answers"] == {"domaine": "aucune"}


def test_deduplication_avant_decoupage(tmp_path):
    src = tmp_path / "c.jsonl"
    src.write_text((_ligne("cherche transformers dans le wiki", "/q") + "\n") * 2, encoding="utf-8")
    n_train, n_val = reformat(src, tmp_path / "t.jsonl", tmp_path / "v.jsonl")
    assert n_train + n_val == 1


def test_validation_stratifiee_et_deterministe(tmp_path):
    src = tmp_path / "c.jsonl"
    lignes = [_ligne(f"{cmd} msg {i}", cmd) for cmd, n in (("/q", 200), ("/cron", 100), ("/lire", 20))
              for i in range(n)]
    src.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    reformat(src, tmp_path / "t.jsonl", tmp_path / "v.jsonl")
    val = Counter(json.loads(l)["state"].split()[0] for l in open(tmp_path / "v.jsonl"))
    assert val == {"/q": 40, "/cron": 25, "/lire": 5}
    reformat(src, tmp_path / "t2.jsonl", tmp_path / "v2.jsonl")
    assert (tmp_path / "v.jsonl").read_text() == (tmp_path / "v2.jsonl").read_text()
