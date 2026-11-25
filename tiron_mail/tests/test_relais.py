# tiron_mail/tests/test_relais.py
import email
import email.policy

import pytest

from tiron_mail.relais import authentifier_relaye, selecteur_relaye

R = ("yc28mybv@mail.grokbot.com",)


def test_regle_adresse():
    assert selecteur_relaye("yc28mybv@mail.grokbot.com", R) == ""
    assert selecteur_relaye("YC28MYBV+Rector@Mail.GrokBot.com", R) == "rector"
    for faux in ("yc28mybvx@mail.grokbot.com", "yc28mybv@x.mail.grokbot.com", "yc28mybv+a b@mail.grokbot.com",
                 "yc28mybv+@mail.grokbot.com", "autre@mail.grokbot.com", "yc28mybv@mail.grokbot.com.evil.fr", ""):
        assert selecteur_relaye(faux, R) is None, faux
    assert selecteur_relaye("yc28mybv@mail.grokbot.com", ()) is None


# En-têtes du courriel d'essai de Rector (2026-11-25 16:42 UTC).
AUTHENTIQUE = """From: yc28mybv@mail.grokbot.com
Return-Path: <010001a2-000000@mail.mail.grokbot.com>
To: tiron@secretarius.co
Subject: Test
Message-ID: <010001a2-000000@email.amazonses.com>
X-Pm-Origin: external
Authentication-Results: mail.protonmail.ch; dmarc=pass (p=reject dis=none) header.from=mail.grokbot.com
Authentication-Results: mail.protonmail.ch; spf=pass smtp.mailfrom=mail.mail.grokbot.com
Authentication-Results: mail.protonmail.ch; dkim=pass (2048-bit key) header.d=mail.grokbot.com header.i=@mail.grokbot.com header.b="Nj"; dkim=pass (1024-bit key) header.d=amazonses.com header.i=@amazonses.com header.b="SR"
Content-Type: text/plain; charset="utf-8"

Bonjour Tiron, ici Rector de Grok Bot, ceci est un test.
"""


def _m(texte):
    return email.message_from_string(texte, policy=email.policy.default)


def test_relaye_authentique():
    assert authentifier_relaye(_m(AUTHENTIQUE)) == ""


def test_relaye_dmarc_fail_refuse():
    assert "DMARC" in authentifier_relaye(_m(AUTHENTIQUE.replace("dmarc=pass", "dmarc=fail")))


def test_dmarc_autre_domaine_refuse():
    assert authentifier_relaye(_m(AUTHENTIQUE.replace("header.from=mail.grokbot.com", "header.from=evil.fr")))


def test_dkim_autre_domaine_refuse():
    m = _m(AUTHENTIQUE.replace('dkim=pass (2048-bit key) header.d=mail.grokbot.com header.i=@mail.grokbot.com header.b="Nj"; ', ""))
    assert "DKIM" in authentifier_relaye(m)


def test_auth_results_hors_proton_ignores():
    faux = AUTHENTIQUE.replace("X-Pm-Origin: external\n",
                               "X-Pm-Origin: external\nAuthentication-Results: evil.fr; dkim=pass header.d=mail.grokbot.com\n")
    faux = faux.replace("mail.protonmail.ch; dkim=pass", "mail.protonmail.ch; dkim=fail")
    assert authentifier_relaye(_m(faux))


def test_origine_interne_refusee():
    assert authentifier_relaye(_m(AUTHENTIQUE.replace("X-Pm-Origin: external", "X-Pm-Origin: internal")))


def test_sans_message_id_refuse():
    assert authentifier_relaye(_m(AUTHENTIQUE.replace("Message-ID: <010001a2-000000@email.amazonses.com>\n", "")))


def test_deux_from_refuse():
    assert authentifier_relaye(_m("From: autre@evil.fr\n" + AUTHENTIQUE))
