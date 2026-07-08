import os
import torch
from router_service.faq import parse_faq, FaqIndex
from router_service import server as router_server


def _stub_embed(texts):
    # embeddings one-hot par mot-clé (déjà normalisés)
    vecs = []
    for t in texts:
        tl = t.lower()
        if "perroquet" in tl:
            vecs.append([1.0, 0.0, 0.0])
        elif "wiki" in tl:
            vecs.append([0.0, 1.0, 0.0])
        else:
            vecs.append([0.0, 0.0, 1.0])
    return torch.tensor(vecs)


def _ecrire(tmp_path, contenu):
    p = tmp_path / "faits.md"
    p.write_text(contenu, encoding="utf-8")
    return p


def test_entree_simple():
    e = parse_faq("## Question ?\nRéponse.")
    assert e == [{"questions": ["Question ?"], "answer": "Réponse."}]


def test_multi_formulations():
    e = parse_faq("## Q1 ?\n## Q2 ?\nUne réponse.")
    assert e == [{"questions": ["Q1 ?", "Q2 ?"], "answer": "Une réponse."}]


def test_corps_multiligne_et_deux_entrees():
    txt = "## A ?\nligne1\nligne2\n\n## B ?\nrb"
    e = parse_faq(txt)
    assert e == [
        {"questions": ["A ?"], "answer": "ligne1\nligne2"},
        {"questions": ["B ?"], "answer": "rb"},
    ]


def test_commentaires_et_titre_h1_ignores():
    txt = "# Faits\n# Format : ...\n## Q ?\nR."
    assert parse_faq(txt) == [{"questions": ["Q ?"], "answer": "R."}]


def test_fichier_vide():
    assert parse_faq("") == []


def test_entree_sans_corps_ignoree():
    assert parse_faq("## Q sans réponse ?") == []


def test_garde_fou_entree_trop_longue(capsys):
    from router_service.faq import FAQ_MAX_ENTREE
    long = "x" * (FAQ_MAX_ENTREE + 1)
    assert parse_faq(f"## Q ?\n{long}") == []
    assert "ignorée" in capsys.readouterr().out


def test_lookup_match(tmp_path):
    p = _ecrire(tmp_path, "## Le perroquet de Mme Michu ?\nCoco.")
    idx = FaqIndex(_stub_embed, path=p, seuil=0.6)
    assert idx.lookup("parle-moi du perroquet")["answer"] == "Coco."


def test_lookup_sous_seuil(tmp_path):
    p = _ecrire(tmp_path, "## Le perroquet ?\nCoco.")
    idx = FaqIndex(_stub_embed, path=p, seuil=0.6)
    assert idx.lookup("quelle météo aujourd'hui") is None


def test_lookup_fichier_absent(tmp_path):
    idx = FaqIndex(_stub_embed, path=tmp_path / "absent.md", seuil=0.6)
    assert idx.lookup("le perroquet") is None


def test_reload_sur_mtime(tmp_path):
    p = _ecrire(tmp_path, "## wiki ?\nancienne")
    idx = FaqIndex(_stub_embed, path=p, seuil=0.6)
    assert idx.lookup("le wiki")["answer"] == "ancienne"
    p.write_text("## wiki ?\nnouvelle", encoding="utf-8")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
    assert idx.lookup("le wiki")["answer"] == "nouvelle"


def _install_faq(tmp_path):
    p = tmp_path / "faits.md"
    p.write_text("## Le perroquet de Mme Michu ?\nCoco.", encoding="utf-8")
    router_server._faq = FaqIndex(_stub_embed, path=p, seuil=0.6)


def test_route_faq_dabord(tmp_path):
    _install_faq(tmp_path)
    r = router_server.route_message("parle-moi du perroquet")
    assert r == {"status": "answer", "reply": "Coco."}


def test_route_slash_court_circuite_faq(tmp_path):
    _install_faq(tmp_path)
    # commence par '/' -> FAQ ignorée ; l'adaptateur (8998) est injoignable en
    # test -> call_adapter lève -> no_match. On vérifie surtout : jamais "answer".
    r = router_server.route_message("/perroquet")
    assert r["status"] != "answer"


def test_route_sans_match_retombe_routage(tmp_path):
    _install_faq(tmp_path)
    r = router_server.route_message("cherche un truc inconnu xyz")
    assert r["status"] != "answer"
