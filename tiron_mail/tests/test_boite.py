# tiron_mail/tests/test_boite.py
from dataclasses import replace
from datetime import date

import imaplib
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import pytest

from tiron_mail.boite import Boite, ErreurBoite, criteres_imap
from tiron_mail.tests.conftest import FauxIMAP, brut


def test_criteres_imap():
    j = date(2026, 10, 7)
    assert criteres_imap("de:paul chantier", j) == ["FROM", '"paul"', "TEXT", '"chantier"']
    assert criteres_imap("devis toiture", j) == ["TEXT", '"devis"', "TEXT", '"toiture"']
    assert criteres_imap("objet:devis depuis:7j", j) == ["SUBJECT", '"devis"', "SINCE", "30-Sep-2026"]
    assert criteres_imap("", j) == ["ALL"]


def test_lister_les_plus_recents_d_abord(faux_imap, compte_gmail):
    b = Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=faux_imap)
    ids = [m["id"] for m in b.lister(2)]
    assert ids == ["g:3", "g:2"]


def test_lister_trie_par_date_et_non_par_uid(compte_gmail):
    # Proton Bridge numérote dans l'ordre de sa synchronisation (du plus récent
    # au plus ancien) : l'UID le plus haut n'est pas le message le plus récent.
    serveur = FauxIMAP({1: brut(3), 2: brut(2), 3: brut(1)})
    compte = replace(compte_gmail, tri_par_date=True)
    b = Boite(compte, {"GMAIL_APP_PASSWORD": "x"}, connecter=lambda c: serveur)
    assert [m["id"] for m in b.lister(2)] == ["g:1", "g:2"]


def test_lire(faux_imap, compte_gmail):
    m = Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=faux_imap).lire("2")
    assert m["objet"] == "Message 2"


def test_mot_de_passe_absent(faux_imap, compte_gmail):
    with pytest.raises(ErreurBoite, match="GMAIL_APP_PASSWORD"):
        Boite(compte_gmail, {}, connecter=faux_imap).lister()


def test_boite_injoignable(compte_gmail):
    def refus(compte):
        raise OSError("connexion refusée")
    with pytest.raises(ErreurBoite, match="injoignable"):
        Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=refus).lister()


def test_uid_introuvable(faux_imap, compte_gmail):
    with pytest.raises(ErreurBoite, match="introuvable"):
        Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=faux_imap).lire("99")


def test_phrase_naturelle_reduite_aux_mots_utiles():
    j = date(2026, 10, 7)
    assert criteres_imap("Ai-je reçu un mail de Paul sur le chantier ?", j) == \
        ["TEXT", '"Paul"', "TEXT", '"chantier"']
    assert criteres_imap("cherchez les mails de la banque", j) == ["TEXT", '"banque"']


def _boite(serveur, compte):
    return Boite(compte, {"GMAIL_APP_PASSWORD": "x"}, connecter=lambda c: serveur)


def test_recherche_accentuee_serveur_sans_utf8(compte_gmail):
    # Proton Bridge : ni UTF8=ACCEPT ni recherche non ASCII ; on garde la plus
    # longue partie ASCII du mot (« réunion » → « union », recherche par sous-chaîne).
    serveur = FauxIMAP({1: brut(1)})
    _boite(serveur, compte_gmail).chercher("réunion de:hélène chantier")
    assert serveur.recherches[-1] == ("TEXT", '"union"', "TEXT", '"chantier"')


def test_recherche_accentuee_serveur_utf8(compte_gmail):
    serveur = FauxIMAP({1: brut(1)}, utf8=True)
    _boite(serveur, compte_gmail).chercher("réunion")
    assert serveur.utf8_actif and serveur.recherches[-1] == ("TEXT", '"réunion"')


def test_recherche_entierement_accentuee_impossible(compte_gmail):
    with pytest.raises(ErreurBoite, match="accents"):
        _boite(FauxIMAP({1: brut(1)}), compte_gmail).chercher("été")


def test_tri_par_date_ne_date_que_les_extremites(compte_gmail):
    # 400 messages numérotés à rebours (le n° 1 est le plus récent), comme après
    # une synchronisation du Bridge : seuls les deux bouts sont datés.
    t0 = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    serveur = FauxIMAP({u: brut(1, quand=format_datetime(t0 - timedelta(hours=u))) for u in range(1, 401)})
    ids = [m["id"] for m in _boite(serveur, replace(compte_gmail, tri_par_date=True)).lister(3)]
    assert ids == ["g:1", "g:2", "g:3"]
    assert len(serveur.datees) <= 100


def test_lire_erreur_serveur_devient_erreur_boite(compte_gmail):
    class Refus(FauxIMAP):
        def uid(self, commande, *args):
            raise imaplib.IMAP4.error("BAD invalid uid")
    with pytest.raises(ErreurBoite, match="injoignable"):
        _boite(Refus({}), compte_gmail).lire("-5")


def test_depuis_demesure_ne_plante_pas():
    assert criteres_imap("depuis:99999999j", date(2026, 10, 7))[0] == "SINCE"
