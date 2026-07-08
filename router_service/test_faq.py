from router_service.faq import parse_faq


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
