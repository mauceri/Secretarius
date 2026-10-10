# tiron_mail/tests/test_releve.py
import email
import email.policy

from tiron_mail.releve import authentifier, commandes

PROPRIO = ("christian@mauceri.me", "cmauceri@gmail.com")


def msg(entetes: list[tuple[str, str]]):
    brut = "".join(f"{k}: {v}\r\n" for k, v in entetes) + "\r\ncorps\r\n"
    return email.message_from_string(brut, policy=email.policy.default)


INTERNE = [("From", "C M <christian@mauceri.me>"), ("Return-Path", "<christian@mauceri.me>"),
           ("X-Pm-Origin", "internal"), ("X-Pm-Content-Encryption", "end-to-end")]
EXTERNE = [("From", "C M <cmauceri@gmail.com>"), ("Return-Path", "<cmauceri@gmail.com>"),
           ("X-Pm-Origin", "external"), ("X-Pm-Content-Encryption", "on-delivery"),
           ("Authentication-Results", "mail.protonmail.ch; dmarc=pass (p=none dis=none) header.from=gmail.com"),
           ("Authentication-Results", "mail.protonmail.ch; spf=pass smtp.mailfrom=gmail.com")]


def test_interne_du_proprietaire_accepte():
    assert authentifier(msg(INTERNE), PROPRIO) == ""


def test_externe_dmarc_pass_accepte():
    assert authentifier(msg(EXTERNE), PROPRIO) == ""


def test_expediteur_inconnu_refuse():
    m = msg([("From", "x@ex.fr"), ("Return-Path", "<x@ex.fr>")] + INTERNE[2:])
    assert "non autorisé" in authentifier(m, PROPRIO)


def test_return_path_different_refuse():
    m = msg([INTERNE[0], ("Return-Path", "<x@ex.fr>")] + INTERNE[2:])
    assert "Return-Path" in authentifier(m, PROPRIO)


def test_interne_non_chiffre_refuse():
    m = msg(INTERNE[:3] + [("X-Pm-Content-Encryption", "on-delivery")])
    assert authentifier(m, PROPRIO)


def test_externe_dmarc_fail_refuse_meme_avec_un_pass_ajoute():
    m = msg(EXTERNE + [("Authentication-Results", "mail.protonmail.ch; dmarc=fail header.from=gmail.com")])
    assert "DMARC" in authentifier(m, PROPRIO)


def test_externe_dmarc_d_un_autre_serveur_ignore():
    m = msg(EXTERNE[:4] + [("Authentication-Results", "mx.pirate.ex; dmarc=pass")])
    assert "DMARC" in authentifier(m, PROPRIO)


def test_origine_absente_refusee():
    assert authentifier(msg(INTERNE[:2]), PROPRIO)


def test_deux_from_refuses():
    m = msg([("From", "christian@mauceri.me, x@ex.fr")] + INTERNE[1:])
    assert authentifier(m, PROPRIO)


def test_commandes_objet_et_lignes_du_corps():
    assert commandes("/wikistatus", "Bonjour\n/inbox 3\n> /ecrire cité\n  /r chiens") == \
        ["/wikistatus", "/inbox 3", "/r chiens"]


def test_sans_commande():
    assert commandes("wikistatus", "pas de barre") == []
