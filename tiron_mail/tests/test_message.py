# tiron_mail/tests/test_message.py
from email.message import EmailMessage

from tiron_mail.message import analyser, resume


def _brut(**kw):
    m = EmailMessage()
    m["From"] = kw.get("de", "Paul <paul@ex.fr>")
    m["To"] = "moi@gmail.com"
    m["Subject"] = kw.get("objet", "Chantier")
    m["Date"] = "Tue, 06 Oct 2026 09:12:00 +0200"
    m["Message-ID"] = "<abc@ex.fr>"
    if "html" in kw:
        m.set_content(kw["html"], subtype="html")
    else:
        m.set_content(kw.get("texte", "Bonjour,\nle chantier avance."))
    if kw.get("pj"):
        m.add_attachment(b"%PDF", maintype="application", subtype="pdf", filename="devis.pdf")
    return m.as_bytes()


def test_message_texte():
    m = analyser(_brut())
    assert m["expediteur"] == "Paul <paul@ex.fr>" and m["objet"] == "Chantier"
    assert "le chantier avance" in m["texte"] and m["message_id"] == "<abc@ex.fr>"


def test_objet_encode_decode():
    m = analyser(_brut(objet="Réunion de lundi"))
    assert m["objet"] == "Réunion de lundi"


def test_html_converti_en_texte_sans_script():
    m = analyser(_brut(html="<html><body><p>Lettre <b>n°5</b></p><script>alert(1)</script></body></html>"))
    assert "Lettre n°5" in m["texte"] and "alert" not in m["texte"] and "<" not in m["texte"]


def test_pieces_jointes_listees_pas_ouvertes():
    m = analyser(_brut(pj=True))
    assert m["pieces_jointes"] == ["devis.pdf"] and "%PDF" not in m["texte"]


def test_resume():
    assert resume(analyser(_brut()), "g:42") == "g:42  06/10 09:12  Paul <paul@ex.fr>  Chantier"
