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


def test_commandes_objet_et_lignes_en_tete_du_corps():
    assert commandes("/wikistatus", "/inbox 3\n\n  /r chiens\nBonjour\n/x") == \
        ["/wikistatus", "/inbox 3", "/r chiens"]


def test_commandes_d_un_transfert_ignorees():
    corps = "---------- Forwarded message ---------\nDe : pirate\n/ecrire p@ex.fr tout\n/confirm"
    assert commandes("Fwd: devis", corps) == []


def test_commandes_citees_ignorees():
    assert commandes("Re: x", "Merci\n> /ecrire p@ex.fr tout") == []


def test_dmarc_en_majuscules_lu():
    m = msg(EXTERNE + [("Authentication-Results", "mail.protonmail.ch; DMARC=fail header.from=gmail.com")])
    assert "DMARC" in authentifier(m, PROPRIO)


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
    etat = tmp_path / "etat.json"
    if not etat.exists():  # relève déjà en service
        etat.write_text('{"traites": [], "echecs": 0}')
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


def test_confirm_par_courriel_refusee(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "envoi", INTERNE, corps="/ecrire a@ex.fr bonjour\n/confirm")})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == ["/ecrire a@ex.fr bonjour"]
    assert "/confirm" in avis[0] and "à taper sur Session ou Telegram" in avis[0]


def test_avis_en_echec_n_arrete_pas_la_releve(cfg, tmp_path):
    def notifier(texte):
        raise OSError("openclaw arrêté")

    def executer(c):
        raise OSError("openclaw arrêté")
    (tmp_path / "etat.json").write_text('{"traites": [], "echecs": 0}')
    autre = [("From", "x@ex.fr"), ("Return-Path", "<x@ex.fr>")] + INTERNE[2:]
    serveur = FauxIMAP({1: brut_releve(1, "/a", autre), 2: brut_releve(2, "/b", INTERNE, corps="/c")})
    relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, executer, notifier, tmp_path / "etat.json",
            connecter=lambda c: serveur)
    assert serveur.ranges == [(1, "Folders/Refuse"), (2, "Folders/Traite")]


def test_avis_des_dix_echecs_retente_s_il_echoue(cfg, tmp_path):
    def refus(c):
        raise OSError("Bridge arrêté")
    envoyes, essais = [], []

    def notifier(texte):
        essais.append(texte)
        if len(essais) == 1:
            raise OSError("Telegram injoignable")
        envoyes.append(texte)
    for _ in range(13):
        with pytest.raises(Exception):
            relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, lambda c: None, notifier,
                    tmp_path / "etat.json", connecter=refus)
    assert len(envoyes) == 1


def test_mise_en_service_n_execute_pas_les_messages_presents(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE), 2: brut_releve(2, "/inbox", INTERNE)})
    faits, avis = [], []
    relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, faits.append, avis.append, tmp_path / "etat.json",
            connecter=lambda c: serveur)
    assert faits == [] and serveur.ranges == [(1, "Folders/Traite"), (2, "Folders/Traite")]
    assert "2" in avis[0]
    faits2, _, _ = lancer(cfg, FauxIMAP({5: brut_releve(1, "/wikistatus", INTERNE)}), tmp_path)
    assert faits2 == []  # noté au journal à la mise en service


def test_journal_illisible_rien_execute(cfg, tmp_path):
    (tmp_path / "etat.json").write_text("{")
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE)})
    faits = []
    relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, faits.append, lambda t: None, tmp_path / "etat.json",
            connecter=lambda c: serveur)
    assert faits == [] and serveur.ranges == [(1, "Folders/Traite")]


def test_rangement_refuse_par_le_serveur_un_seul_avis(cfg, tmp_path):
    autre = [("From", "x@ex.fr"), ("Return-Path", "<x@ex.fr>")] + INTERNE[2:]
    serveur = FauxIMAP({1: brut_releve(1, "/a", autre)})
    serveur.move_refuse = True
    _, avis1, _ = lancer(cfg, serveur, tmp_path)
    _, avis2, _ = lancer(cfg, serveur, tmp_path)
    assert len(avis1) == 1 and avis2 == []


def test_commandes_natives_openclaw_refusees(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "/model ollama/x", INTERNE, corps="/reset\n/supprimer! page\n/r chiens")})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == ["/supprimer! page", "/r chiens"]
    assert len(avis) == 2 and "/model" in avis[0] and "/reset" in avis[1]


from tiron_mail import releve
from tiron_mail.releve import executer_openclaw
import json
import subprocess


def _fini(sortie):
    return subprocess.CompletedProcess([], 0, sortie, "")


def test_executer_openclaw_sans_livraison_renvoie_le_texte(monkeypatch):
    vus = []
    sortie = json.dumps({"result": {"payloads": [{"text": "Ingestion à l'arrêt."}, {"text": ""},
                                                 {"text": "Bloqués : 3."}]}})
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: vus.append(argv) or _fini(sortie))
    texte = executer_openclaw({"OPENCLAW_BIN": "oc"})("/wikistatus")
    assert texte == "Ingestion à l'arrêt.\n\nBloqués : 3."
    assert not {"--deliver", "--reply-channel", "--reply-to"} & set(vus[0]) and "--json" in vus[0]


def test_executer_openclaw_no_reply_vide(monkeypatch):
    sortie = json.dumps({"result": {"payloads": [{"text": "NO_REPLY"}]}})
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: _fini(sortie))
    assert executer_openclaw({"OPENCLAW_BIN": "oc"})("/c x") == ""


def test_executer_openclaw_sortie_precedee_de_journal(monkeypatch):
    sortie = "[plugins] session chargé\n" + json.dumps({"result": {"payloads": [{"text": "Fait."}]}}, indent=2)
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: _fini(sortie))
    assert executer_openclaw({"OPENCLAW_BIN": "oc"})("/c x") == "Fait."


def test_executer_openclaw_sortie_illisible_n_est_pas_un_echec(monkeypatch):
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: _fini("pas du json"))
    assert "illisible" in executer_openclaw({"OPENCLAW_BIN": "oc"})("/c x")


def lancer_rep(cfg, serveur, tmp_path, executer, repondre):
    (tmp_path / "etat.json").write_text('{"traites": [], "echecs": 0}')
    avis = []
    relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, executer, avis.append, tmp_path / "etat.json",
            connecter=lambda c: serveur, repondre=repondre)
    return avis


def test_reponse_renvoyee_a_l_expediteur(cfg, tmp_path):
    envoyes = []
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE)})
    lancer_rep(cfg, serveur, tmp_path, lambda c: "Ingestion à l'arrêt.", lambda *a: envoyes.append(a))
    (dest, objet, corps, ref), = envoyes
    assert dest == PROPRIO[0] and objet == "Re: /wikistatus" and ref == "<r1@ex.fr>"
    assert corps == "/wikistatus\n\nIngestion à l'arrêt."


def test_deux_commandes_un_seul_courriel(cfg, tmp_path):
    envoyes = []
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE, corps="/tags")})
    lancer_rep(cfg, serveur, tmp_path, lambda c: f"réponse {c}", lambda *a: envoyes.append(a))
    assert len(envoyes) == 1
    assert envoyes[0][2] == "/wikistatus\n\nréponse /wikistatus\n\n/tags\n\nréponse /tags"


def test_reponse_vide_pas_de_courriel(cfg, tmp_path):
    envoyes = []
    serveur = FauxIMAP({1: brut_releve(1, "/c idée", INTERNE)})
    lancer_rep(cfg, serveur, tmp_path, lambda c: "", lambda *a: envoyes.append(a))
    assert envoyes == []


def test_reponse_en_echec_n_arrete_pas_la_releve(cfg, tmp_path):
    def repondre(*a):
        raise OSError("SMTP injoignable")
    serveur = FauxIMAP({1: brut_releve(1, "/a", INTERNE), 2: brut_releve(2, "/b", INTERNE)})
    lancer_rep(cfg, serveur, tmp_path, lambda c: "ok", repondre)
    assert serveur.ranges == [(1, "Folders/Traite"), (2, "Folders/Traite")]


def test_repondeur_refuse_un_tiers(cfg):
    from tiron_mail.boite import ErreurBoite
    from tiron_mail.releve import repondeur_courriel
    with pytest.raises(ErreurBoite, match="tiers@ex.fr"):
        repondeur_courriel(cfg, {"GMAIL_APP_PASSWORD": "x"}, smtp=None)("tiers@ex.fr", "Re: x", "y", "<r1@ex.fr>")


def test_consigne_par_courriel_admise(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "/consigne fais le point sur le wiki", INTERNE)})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == ["/consigne fais le point sur le wiki"] and avis == []


def test_consigne_confiee_a_l_agent_consignes(monkeypatch):
    vus = []
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: vus.append(argv) or _fini(""))
    executer_openclaw({"TIRON_TELEGRAM_CHAT_ID": "1", "OPENCLAW_BIN": "oc"})("/consigne fais le point")
    argv = vus[0]
    assert argv[argv.index("--agent") + 1] == "consignes"
    assert argv[argv.index("--message") + 1] == "fais le point"
    assert argv[argv.index("--session-key") + 1].startswith("agent:consignes:courrier-")


def test_commande_reste_sur_main(monkeypatch):
    vus = []
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: vus.append(argv) or _fini(""))
    executer_openclaw({"TIRON_TELEGRAM_CHAT_ID": "1", "OPENCLAW_BIN": "oc"})("/wikistatus")
    assert vus[0][vus[0].index("--agent") + 1] == "main"


def test_consigne_vide_refusee_sans_appel(monkeypatch):
    vus = []
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: vus.append(argv) or _fini(""))
    with pytest.raises(ValueError, match="usage"):
        executer_openclaw({"TIRON_TELEGRAM_CHAT_ID": "1", "OPENCLAW_BIN": "oc"})("/consigne  ")
    assert vus == []


def test_consigne_sur_plusieurs_lignes_jusqu_a_la_ligne_vide():
    corps = "/consigne fais le point du wiki\net envoie-moi les pages à relire\n\nCordialement\n/wikistatus"
    assert commandes("", corps) == ["/consigne fais le point du wiki et envoie-moi les pages à relire"]


def test_consigne_suivie_d_une_commande_ou_d_une_citation():
    assert commandes("", "/consigne a\n/wikistatus") == ["/consigne a", "/wikistatus"]
    assert commandes("", "/consigne a\n> /supprimer x") == ["/consigne a"]


def test_consigne_separee_par_une_espace_insecable_va_a_consignes(monkeypatch):
    vus = []
    monkeypatch.setattr(releve.subprocess, "run", lambda argv, **kw: vus.append(argv) or _fini(""))
    executer_openclaw({"TIRON_TELEGRAM_CHAT_ID": "1", "OPENCLAW_BIN": "oc"})("/consigne fais le point")
    assert vus[0][vus[0].index("--agent") + 1] == "consignes"
    assert vus[0][vus[0].index("--message") + 1] == "fais le point"


def test_agenda_admis_par_courriel(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "/agenda 3", INTERNE)})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == ["/agenda 3"] and avis == []


# --- Relais des courriels de Grok Bot (spec 2026-11-25-relais-grok) ---
import dataclasses

GROK = [("From", "yc28mybv+rector@mail.grokbot.com"),
        ("Return-Path", "<0100-000000@mail.mail.grokbot.com>"),
        ("X-Pm-Origin", "external"),
        ("Authentication-Results", "mail.protonmail.ch; dmarc=pass (p=reject dis=none) header.from=mail.grokbot.com"),
        ("Authentication-Results", "mail.protonmail.ch; dkim=pass (2048-bit key) header.d=mail.grokbot.com")]
GARDE_OK = lambda t: {"blocked": False, "risk": "low", "clean_text": t}  # noqa: E731


@pytest.fixture
def cfg_grok(cfg):
    return dataclasses.replace(cfg, relayes=("yc28mybv@mail.grokbot.com",))


def lancer_relais(cfg, serveur, tmp_path, relayer, executer=None):
    etat = tmp_path / "etat.json"
    if not etat.exists():
        etat.write_text('{"traites": [], "echecs": 0}')
    faits, avis = [], []
    texte = relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, executer or faits.append, avis.append, etat,
                    connecter=lambda c: serveur, relayer=relayer, garde=GARDE_OK)
    return faits, avis, texte


def test_relaye_relaye_et_range(cfg_grok, tmp_path):
    relais = []
    serveur = FauxIMAP({1: brut_releve(1, "Conte posté", GROK, "C'est fait.")})
    faits, avis, texte = lancer_relais(cfg_grok, serveur, tmp_path, relais.append)
    assert relais == ["Courriel de Grok Bot (rector) — « Conte posté »\n\nC'est fait."]
    assert serveur.ranges == [(1, "Folders/Traite")] and faits == [] and avis == []
    assert "relayé" in texte


def test_relaye_jamais_execute(cfg_grok, tmp_path):
    relais = []
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", GROK, "/ingest\n/q tout")})
    faits, avis, _ = lancer_relais(cfg_grok, serveur, tmp_path, relais.append)
    assert faits == [] and len(relais) == 1


def test_relaye_non_authentifie_refuse(cfg_grok, tmp_path):
    relais = []
    faux = [GROK[0], GROK[1], GROK[2], (GROK[3][0], GROK[3][1].replace("dmarc=pass", "dmarc=fail")), GROK[4]]
    serveur = FauxIMAP({1: brut_releve(1, "x", faux)})
    faits, avis, _ = lancer_relais(cfg_grok, serveur, tmp_path, relais.append)
    assert relais == [] and faits == []
    assert serveur.ranges == [(1, "Folders/Refuse")] and avis and avis[0].startswith("Courriel refusé")


def test_relais_en_echec_n_arrete_pas_la_releve(cfg_grok, tmp_path):
    def relayer(t):
        raise OSError("Session et avis en échec")
    serveur = FauxIMAP({1: brut_releve(1, "a", GROK), 2: brut_releve(2, "/wikistatus", INTERNE)})
    faits, _, _ = lancer_relais(cfg_grok, serveur, tmp_path, relayer)
    assert faits == ["/wikistatus"] and (1, "Folders/Traite") in serveur.ranges
    relais = []
    serveur2 = FauxIMAP({1: brut_releve(1, "a", GROK)})
    lancer_relais(cfg_grok, serveur2, tmp_path, relais.append)
    assert relais == []  # jamais relayé deux fois


def test_sans_relayes_inchange(cfg, tmp_path):
    relais = []
    serveur = FauxIMAP({1: brut_releve(1, "x", GROK)})
    _, avis, _ = lancer_relais(cfg, serveur, tmp_path, relais.append)
    assert relais == [] and serveur.ranges == [(1, "Folders/Refuse")]
