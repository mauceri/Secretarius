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


def test_avis_lu(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML.replace('adresses_proprietaire', 'avis = "moi@proton.me"\nadresses_proprietaire'), encoding="utf-8")
    assert charger(p).avis == "moi@proton.me"


def test_avis_hors_proprietaire_refuse(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML.replace('adresses_proprietaire', 'avis = "autre@ex.fr"\nadresses_proprietaire'), encoding="utf-8")
    with pytest.raises(ErreurConfig, match="avis"):
        charger(p)


def test_relayes_lus(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML + '\n[releve]\nrelayes = ["YC28MYBV@Mail.GrokBot.com"]\n', encoding="utf-8")
    assert charger(p).relayes == ("yc28mybv@mail.grokbot.com",)


def test_relaye_proprietaire_refuse(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML + '\n[releve]\nrelayes = ["moi@proton.me"]\n', encoding="utf-8")
    with pytest.raises(ErreurConfig, match="relayes"):
        charger(p)


@pytest.mark.parametrize("entree", ["pas-une-adresse", "a+b@x.fr", "a@b", "a b@x.fr", "@x.fr"])
def test_relaye_invalide_refuse(tmp_path, entree):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML + f'\n[releve]\nrelayes = ["{entree}"]\n', encoding="utf-8")
    with pytest.raises(ErreurConfig, match="relayes"):
        charger(p)


def test_relaye_variante_du_proprietaire_refuse(tmp_path):
    p = tmp_path / "comptes.toml"
    toml = TOML.replace('"moi@proton.me"]', '"moi+tiron@proton.me"]')
    p.write_text(toml + '\n[releve]\nrelayes = ["moi@proton.me"]\n', encoding="utf-8")
    with pytest.raises(ErreurConfig, match="relayes"):
        charger(p)
