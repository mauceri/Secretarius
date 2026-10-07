# tiron_mail/tests/test_config.py
import pytest

from tiron_mail.config import ErreurConfig, charger, compte_de_prefixe

TOML = """
[envoi]
defaut = "gmail"
adresses_proprietaire = ["moi@gmail.com", "moi@proton.me"]

[comptes.gmail]
prefixe = "g"
adresse = "moi@gmail.com"
utilisateur = "moi@gmail.com"
hote_imap = "imap.gmail.com"
port_imap = 993
securite_imap = "ssl"
hote_smtp = "smtp.gmail.com"
port_smtp = 465
securite_smtp = "ssl"
secret_env = "GMAIL_APP_PASSWORD"
"""


def test_charger(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML, encoding="utf-8")
    cfg = charger(p)
    assert cfg.envoi_defaut == "gmail"
    assert cfg.adresses_proprietaire == ("moi@gmail.com", "moi@proton.me")
    g = cfg.comptes["gmail"]
    assert g.prefixe == "g" and g.port_imap == 993 and g.secret_env == "GMAIL_APP_PASSWORD"


def test_prefixe_inconnu(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML, encoding="utf-8")
    with pytest.raises(ErreurConfig, match="aucune boîte configurée pour « p »"):
        compte_de_prefixe(charger(p), "p")


def test_envoi_defaut_doit_exister(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML.replace('defaut = "gmail"', 'defaut = "proton"'), encoding="utf-8")
    with pytest.raises(ErreurConfig, match="envoi.defaut"):
        charger(p)


def test_fichier_absent(tmp_path):
    with pytest.raises(ErreurConfig, match="introuvable"):
        charger(tmp_path / "absent.toml")


def test_modele_du_depot_valide():
    from pathlib import Path
    cfg = charger(Path(__file__).parent.parent / "comptes.toml.modele")
    assert set(cfg.comptes) == {"gmail", "proton", "tiron"}
