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
