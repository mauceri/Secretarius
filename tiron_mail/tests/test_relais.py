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


def test_dkim_du_domaine_en_echec_mais_dmarc_pass_admis():
    # Courriel réel du 2026-11-25 17:29 UTC : clé DKIM de mail.grokbot.com absente du DNS,
    # DMARC passé par SPF aligné. DMARC seul suffit (décision utilisateur et Rector).
    m = _m(AUTHENTIQUE.replace('dkim=pass (2048-bit key) header.d=mail.grokbot.com',
                               'dkim=fail reason="key not found in DNS" (0-bit key) header.d=mail.grokbot.com'))
    assert authentifier_relaye(m) == ""


def test_auth_results_hors_proton_ignores():
    faux = AUTHENTIQUE.replace("X-Pm-Origin: external\n",
                               "X-Pm-Origin: external\nAuthentication-Results: evil.fr; dmarc=pass header.from=mail.grokbot.com\n")
    faux = faux.replace("mail.protonmail.ch; dmarc=pass", "mail.protonmail.ch; dmarc=fail")
    assert authentifier_relaye(_m(faux))


def test_origine_interne_refusee():
    assert authentifier_relaye(_m(AUTHENTIQUE.replace("X-Pm-Origin: external", "X-Pm-Origin: internal")))


def test_sans_message_id_refuse():
    assert authentifier_relaye(_m(AUTHENTIQUE.replace("Message-ID: <010001a2-000000@email.amazonses.com>\n", "")))


def test_deux_from_refuse():
    assert authentifier_relaye(_m("From: autre@evil.fr\n" + AUTHENTIQUE))


# --- Tâche 2 : garde, mise en forme, envoi Session ---
import json

from tiron_mail.relais import envoyeur_session, message_relais, relayeur


def _garde(rep):
    return lambda texte: rep


def test_message_ok():
    g = _garde({"blocked": False, "risk": "low", "clean_text": "Bonjour"})
    assert message_relais("rector", "Test", "Bonjour", g) == "Courriel de Grok Bot (rector) — « Test »\n\nBonjour"
    assert message_relais("", "Test", "Bonjour", g) == "Courriel de Grok Bot — « Test »\n\nBonjour"


def test_message_suspect():
    out = message_relais("", "T", "x", _garde({"blocked": False, "risk": "medium", "clean_text": "x"}))
    assert out == "Courriel de Grok Bot — « T »\n\nAttention : contenu potentiellement suspect.\n\nx"


def test_message_bloque():
    out = message_relais("", "T", "SECRET", _garde({"blocked": True, "reason": "injection"}))
    assert out == "Courriel de Grok Bot — « T »\n\ncontenu bloqué par le détecteur d'injection : injection"


def test_message_garde_injoignable():
    def g(t):
        raise ConnectionRefusedError("refusé")
    out = message_relais("", "T", "SECRET", g)
    assert "SECRET" not in out
    assert out.endswith("contenu non vérifié (garde injoignable), consultez la boîte de Tiron")


def test_message_coupe():
    out = message_relais("", "T", "a" * 5000, _garde({"blocked": False, "risk": "low", "clean_text": "a" * 5000}))
    assert out.endswith("a" * 3500 + "… (suite dans la boîte de Tiron)")
    assert "a" * 3501 not in out


class _Rep:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"{}"


def test_envoyeur_session(tmp_path):
    s = tmp_path / "secrets.env"
    s.write_text("SESSION_BOT_MNEMONIC='a b'\nTIRON_SESSION_BROKER_TOKEN='jt'\nTIRON_SESSION_ID=05ab\n")
    vus = []

    def ouvrir(req, timeout):
        vus.append(req)
        return _Rep()
    envoyeur_session(s, ouvrir=ouvrir)("coucou")
    req = vus[0]
    assert req.full_url == "http://127.0.0.1:8091/send"
    assert req.get_header("Authorization") == "Bearer jt"
    assert json.loads(req.data) == {"to": "05ab", "text": "coucou"}
    assert b"a b" not in req.data


def test_envoyeur_sans_jeton(tmp_path):
    s = tmp_path / "secrets.env"
    s.write_text("TIRON_SESSION_ID=05ab\n")
    vus = []
    with pytest.raises(RuntimeError, match="TIRON_SESSION_BROKER_TOKEN"):
        envoyeur_session(s, ouvrir=lambda req, timeout: vus.append(req))("x")
    assert vus == []


def test_envoyeur_http_en_erreur(tmp_path):
    s = tmp_path / "secrets.env"
    s.write_text("TIRON_SESSION_BROKER_TOKEN=jt\nTIRON_SESSION_ID=05ab\n")
    with pytest.raises(RuntimeError):
        envoyeur_session(s, ouvrir=lambda req, timeout: _Rep(502))("x")


def test_relayeur_repli():
    avis = []

    def session(t):
        raise OSError("broker arrêté")
    relayeur(session, avis.append)("Courriel de Grok Bot — « T »")
    assert avis == ["(Session indisponible) Courriel de Grok Bot — « T »"]

    def avis_ko(t):
        raise OSError("SMTP")
    with pytest.raises(OSError):
        relayeur(session, avis_ko)("x")


def test_relayeur_session_ok():
    vus, avis = [], []
    relayeur(vus.append, avis.append)("x")
    assert vus == ["x"] and avis == []


# --- Corrections de la relecture finale ---
def test_dmarc_fail_sans_header_from_refuse():
    # Un dmarc=fail de Proton sans header.from ne doit pas être ignoré au profit d'un pass ajouté.
    faux = AUTHENTIQUE.replace("dmarc=pass (p=reject dis=none) header.from=mail.grokbot.com", "dmarc=fail")
    faux = faux.replace("X-Pm-Origin: external\n", "X-Pm-Origin: external\nAuthentication-Results: mail.protonmail.ch; "
                        "dmarc=pass header.from=mail.grokbot.com\n")
    assert "DMARC" in authentifier_relaye(_m(faux))


def test_objet_verifie_par_le_garde_et_coupe():
    vus = []

    def garde(t):
        vus.append(t)
        return {"blocked": False, "risk": "low", "clean_text": t}
    out = message_relais("", "O" * 500, "corps", garde)
    assert vus[0] == "O" * 200 + "…\n\ncorps"  # le garde voit exactement l'objet affiché
    assert "O" * 200 + "…" in out and "O" * 201 not in out


def test_injection_dans_l_objet_bloquee():
    def garde(t):
        return {"blocked": True, "reason": "injection"} if "IGNORE" in t else {"blocked": False, "clean_text": t}
    out = message_relais("", "IGNORE tout", "corps", garde)
    assert "contenu bloqué" in out
