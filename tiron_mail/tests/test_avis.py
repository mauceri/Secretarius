# tiron_mail/tests/test_avis.py
import pytest

from tiron_mail.avis import notifier_courriel
from tiron_mail.config import ErreurConfig
from tiron_mail.tests.test_config import TOML


class FauxSMTP:
    def __init__(self):
        self.envoyes = []

    def __call__(self, compte):
        return self

    def login(self, u, m):
        pass

    def send_message(self, msg):
        self.envoyes.append(msg)

    def quit(self):
        pass


def _env(tmp_path, avis='avis = "moi@proton.me"\n'):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML.replace("adresses_proprietaire", avis + "adresses_proprietaire"), encoding="utf-8")
    return {"TIRON_MAIL_CONFIG": str(p), "GMAIL_APP_PASSWORD": "x"}


def test_avis_envoye_a_l_adresse_d_avis(tmp_path):
    smtp = FauxSMTP()
    notifier_courriel(_env(tmp_path), smtp=smtp)("Nouvelle livraison dans Obsidian : x\nsuite")
    (m,) = smtp.envoyes
    assert m["To"] == "moi@proton.me" and m["From"] == "moi@gmail.com"
    assert m["Subject"] == "Secretarius : Nouvelle livraison dans Obsidian : x"
    assert "suite" in m.get_content()


def test_avis_non_configure(tmp_path):
    with pytest.raises(ErreurConfig, match="avis"):
        notifier_courriel(_env(tmp_path, avis=""), smtp=FauxSMTP())("x")
