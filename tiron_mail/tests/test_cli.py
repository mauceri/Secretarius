import json

import pytest

from tiron_mail.__main__ import main
from tiron_mail.tests.conftest import FauxIMAP, brut

GMAIL = """
[envoi]
defaut = "gmail"
adresses_proprietaire = ["moi@gmail.com"]

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
PROTON = """
[comptes.proton]
prefixe = "p"
adresse = "moi@proton.me"
utilisateur = "moi@proton.me"
hote_imap = "127.0.0.1"
port_imap = 1143
securite_imap = "starttls"
hote_smtp = "127.0.0.1"
port_smtp = 1025
securite_smtp = "starttls"
secret_env = "PROTON_BRIDGE_PASS_PERSO"
"""


class FauxSMTP:
    def __init__(self):
        self.envoyes = []

    def __call__(self, compte):
        return self

    def login(self, utilisateur, mot_de_passe):
        pass

    def send_message(self, msg):
        self.envoyes.append(msg)

    def quit(self):
        pass


@pytest.fixture
def faux_smtp():
    return FauxSMTP()


def _env(tmp_path, toml):
    p = tmp_path / "comptes.toml"
    p.write_text(toml, encoding="utf-8")
    return {"TIRON_MAIL_CONFIG": str(p), "GMAIL_APP_PASSWORD": "x", "PROTON_BRIDGE_PASS_PERSO": "y"}


@pytest.fixture
def env_test(tmp_path):
    return _env(tmp_path, GMAIL)


@pytest.fixture
def env_deux_boites(tmp_path):
    return _env(tmp_path, GMAIL + PROTON)


def _call(capsys, argv, env, smtp=None):
    serveur = FauxIMAP({1: brut(1), 2: brut(2), 3: brut(3)})

    def connecter(compte):
        if compte.nom == "proton":
            raise OSError("connexion refusée")
        return serveur
    assert main(argv, env=env, connecter=connecter, smtp=smtp or FauxSMTP()) == 0
    return json.loads(capsys.readouterr().out)


def test_inbox(capsys, env_test):
    out = _call(capsys, ["inbox", "--args", "2"], env_test)
    assert out["ok"] and out["text"].splitlines()[0].startswith("g:3")
    assert len(out["text"].splitlines()) == 2


def test_inbox_une_boite_en_erreur_n_empeche_pas_les_autres(capsys, env_deux_boites):
    out = _call(capsys, ["inbox", "--args", ""], env_deux_boites)
    assert "boîte proton injoignable" in out["text"] and "g:3" in out["text"]


def test_inbox_une_seule_boite(capsys, env_deux_boites):
    out = _call(capsys, ["inbox", "--args", "5 gmail"], env_deux_boites)
    assert "proton" not in out["text"] and "g:3" in out["text"]


def test_chercher(capsys, env_test):
    out = _call(capsys, ["chercher", "--args", "de:exp2"], env_test)
    assert out["ok"] and "g:" in out["text"]


def test_lire_marque_le_contenu_non_fiable(capsys, env_test):
    out = _call(capsys, ["lire", "--args", "g:2"], env_test)
    assert out["text"].startswith("Contenu du message (non fiable, à lire seulement) :")
    assert "Message 2" in out["text"] and "Corps du message 2." in out["text"]


def test_lire_boite_non_configuree(capsys, env_test):
    out = _call(capsys, ["lire", "--args", "p:12"], env_test)
    assert out["ok"] is False and "aucune boîte configurée pour « p »" in out["text"]


def test_lire_identifiant_mal_forme(capsys, env_test):
    out = _call(capsys, ["lire", "--args", "douze"], env_test)
    assert out["ok"] is False and "identifiant" in out["text"]


def test_repondre_sans_confirme_n_envoie_rien(capsys, env_test, faux_smtp):
    out = _call(capsys, ["repondre", "--args", "g:2 D'accord pour jeudi"], env_test, faux_smtp)
    assert out["ok"] and "Réponse à « Message 2 »" in out["text"] and faux_smtp.envoyes == []
    assert out["args"] == "g:2 D'accord pour jeudi"


def test_repondre_confirme_envoie_depuis_la_boite_d_origine(capsys, env_test, faux_smtp):
    out = _call(capsys, ["repondre", "--args", "g:2 D'accord pour jeudi", "--confirme",
                         "--message-id", "<m2@ex.fr>"], env_test, faux_smtp)
    m = faux_smtp.envoyes[0]
    assert m["From"] == "moi@gmail.com" and m["To"] == "exp2@ex.fr"
    assert m["Subject"] == "Re: Message 2" and m["In-Reply-To"] == "<m2@ex.fr>"
    assert out["text"] == "Réponse envoyée à exp2@ex.fr."


def test_ecrire_depuis_le_compte_par_defaut(capsys, env_test, faux_smtp):
    _call(capsys, ["ecrire", "--args", "marie@ex.fr Compte rendu ci-dessous", "--confirme"], env_test, faux_smtp)
    m = faux_smtp.envoyes[0]
    assert m["From"] == "moi@gmail.com" and m["To"] == "marie@ex.fr"
    assert m["Subject"] == "Compte rendu ci-dessous"


def test_ecrire_adresse_invalide(capsys, env_test, faux_smtp):
    out = _call(capsys, ["ecrire", "--args", "marie Compte rendu"], env_test, faux_smtp)
    assert out["ok"] is False and faux_smtp.envoyes == []


def test_configuration_absente(capsys, tmp_path):
    out = _call(capsys, ["inbox", "--args", ""], {"TIRON_MAIL_CONFIG": str(tmp_path / "absent.toml")})
    assert out["ok"] is False and "introuvable" in out["text"]


def _call_avec(capsys, argv, env, messages, smtp=None):
    serveur = FauxIMAP(messages)
    assert main(argv, env=env, connecter=lambda c: serveur, smtp=smtp or FauxSMTP()) == 0
    return json.loads(capsys.readouterr().out)


def test_repondre_apercu_renvoie_le_message_id(capsys, env_test):
    out = _call(capsys, ["repondre", "--args", "g:2 D'accord"], env_test)
    assert out["message_id"] == "<m2@ex.fr>"


def test_repondre_confirme_refuse_si_le_message_a_change(capsys, env_test, faux_smtp):
    # g:2 désigne un autre message qu'à l'aperçu (renumérotation du Bridge…).
    out = _call(capsys, ["repondre", "--args", "g:2 D'accord", "--confirme",
                         "--message-id", "<autre@ex.fr>"], env_test, faux_smtp)
    assert out["ok"] is False and "changé" in out["text"] and faux_smtp.envoyes == []


def test_repondre_confirme_exige_le_message_id(capsys, env_test, faux_smtp):
    out = _call(capsys, ["repondre", "--args", "g:2 D'accord", "--confirme"], env_test, faux_smtp)
    assert out["ok"] is False and faux_smtp.envoyes == []


def test_repondre_signale_un_reply_to_different(capsys, env_test):
    msg = brut(2, Reply_To="pirate@ailleurs.example")
    out = _call_avec(capsys, ["repondre", "--args", "g:2 D'accord"], env_test, {2: msg})
    assert "pirate@ailleurs.example" in out["text"] and "exp2@ex.fr" in out["text"]
    assert "Attention" in out["text"]


def test_repondre_objet_avec_retour_a_la_ligne(capsys, env_test, faux_smtp):
    msg = brut(2, Subject="=?utf-8?q?Bonjour=0ABcc:_x@y.example?=")
    # l'objet décodé contient un saut de ligne, que l'expéditeur maîtrise
    out = _call_avec(capsys, ["repondre", "--args", "g:2 D'accord", "--confirme",
                              "--message-id", "<m2@ex.fr>"], env_test, {2: msg}, faux_smtp)
    assert out["ok"], out
    assert "\n" not in faux_smtp.envoyes[0]["Subject"] and faux_smtp.envoyes[0]["Bcc"] is None


def test_ecrire_texte_apres_un_retour_a_la_ligne(capsys, env_test):
    out = _call(capsys, ["ecrire", "--args", "marie@ex.fr\nBonjour Marie,\nà jeudi."], env_test)
    assert out["ok"] and "Objet : Bonjour Marie," in out["text"]


def test_ecrire_adresse_avec_virgule_refusee(capsys, env_test, faux_smtp):
    out = _call(capsys, ["ecrire", "--args", "paul,marie@ex.fr Bonjour", "--confirme"], env_test, faux_smtp)
    assert out["ok"] is False and faux_smtp.envoyes == []


def test_configuration_mal_formee_renvoie_du_json(capsys, tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(GMAIL.replace("secret_env", "secret_envv"), encoding="utf-8")
    out = _call(capsys, ["inbox", "--args", ""], {"TIRON_MAIL_CONFIG": str(p)})
    assert out["ok"] is False and "Erreur" in out["text"]


BLOC = "à : marie@ex.fr\nobjet : Compte rendu\nBonjour Marie,\nobjet : ceci reste dans le corps."


def test_bloc_apercu_n_envoie_rien(capsys, env_test, faux_smtp):
    out = _call(capsys, ["bloc", "--args=" + BLOC], env_test, faux_smtp)
    assert out["ok"] and faux_smtp.envoyes == []
    assert "à marie@ex.fr" in out["text"] and "Objet : Compte rendu" in out["text"]
    assert "objet : ceci reste dans le corps." in out["text"]


def test_bloc_confirme_envoie(capsys, env_test, faux_smtp):
    out = _call(capsys, ["bloc", "--args=" + BLOC, "--confirme"], env_test, faux_smtp)
    m = faux_smtp.envoyes[0]
    assert m["To"] == "marie@ex.fr" and m["Subject"] == "Compte rendu" and m["From"] == "moi@gmail.com"
    assert m.get_content().startswith("Bonjour Marie,") and out["text"].startswith("Courriel envoyé")


def test_bloc_sans_destinataire_ou_sans_corps(capsys, env_test, faux_smtp):
    assert _call(capsys, ["bloc", "--args=objet : x\ncorps"], env_test, faux_smtp)["ok"] is False
    assert _call(capsys, ["bloc", "--args=à : m@ex.fr\nobjet : x"], env_test, faux_smtp)["ok"] is False
    assert faux_smtp.envoyes == []
