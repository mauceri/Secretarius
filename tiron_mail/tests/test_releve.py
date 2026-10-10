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


from email.message import EmailMessage

import pytest

from tiron_mail.config import Config
from tiron_mail.releve import relever
from tiron_mail.tests.conftest import FauxIMAP


def brut_releve(n, objet, entetes, corps="Bonjour"):
    m = EmailMessage()
    for k, v in entetes:
        m[k] = v
    m["Subject"] = objet
    m["Message-ID"] = f"<r{n}@ex.fr>"
    m.set_content(corps)
    return m.as_bytes()


@pytest.fixture
def cfg(compte_gmail):
    return Config({"tiron": compte_gmail}, "tiron", PROPRIO, releve="tiron")


def lancer(cfg, serveur, tmp_path, executer=None):
    faits, avis = [], []
    texte = relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, executer or faits.append, avis.append,
                    tmp_path / "etat.json", connecter=lambda c: serveur)
    return faits, avis, texte


def test_commande_executee_puis_rangee(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE)})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == ["/wikistatus"] and avis == []
    assert serveur.ranges == [(1, "Folders/Traite")]


def test_refus_avis_sans_le_corps(cfg, tmp_path):
    entetes = [("From", "x@ex.fr"), ("Return-Path", "<x@ex.fr>")] + INTERNE[2:]
    serveur = FauxIMAP({1: brut_releve(1, "/inbox", entetes, corps="SECRET-DU-CORPS")})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == [] and serveur.ranges == [(1, "Folders/Refuse")]
    assert "x@ex.fr" in avis[0] and "SECRET-DU-CORPS" not in avis[0]


def test_jamais_deux_fois(cfg, tmp_path):
    lancer(cfg, FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE)}), tmp_path)
    serveur = FauxIMAP({7: brut_releve(1, "/wikistatus", INTERNE)})  # même Message-ID
    faits, _, _ = lancer(cfg, serveur, tmp_path)
    assert faits == [] and serveur.ranges == [(7, "Folders/Traite")]


def test_sans_commande_avis(cfg, tmp_path):
    faits, avis, _ = lancer(cfg, FauxIMAP({1: brut_releve(1, "wikistatus", INTERNE)}), tmp_path)
    assert faits == [] and "/" in avis[0]


def test_commande_en_echec_avis_et_suite(cfg, tmp_path):
    def executer(c):
        if c == "/a":
            raise OSError("openclaw arrêté")
    serveur = FauxIMAP({1: brut_releve(1, "/a", INTERNE, corps="/b")})
    _, avis, _ = lancer(cfg, serveur, tmp_path, executer=executer)
    assert "/a" in avis[0] and serveur.ranges == [(1, "Folders/Traite")]


def test_sans_message_id_refuse(cfg, tmp_path):
    m = EmailMessage()
    for k, v in INTERNE:
        m[k] = v
    m["Subject"] = "/wikistatus"
    m.set_content("x")
    serveur = FauxIMAP({1: m.as_bytes()})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == [] and "Message-ID" in avis[0]


def test_dix_echecs_un_seul_avis(cfg, tmp_path):
    def refus(c):
        raise OSError("Bridge arrêté")
    avis = []
    for _ in range(12):
        with pytest.raises(Exception):
            relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, lambda c: None, avis.append,
                    tmp_path / "etat.json", connecter=refus)
    assert len(avis) == 1
