# tiron_mail/tests/test_veille_session.py
from datetime import datetime, timedelta, timezone

from tiron_mail.veille_session import lire_jeton, veiller

T0 = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)


def ok(at=T0):
    return lambda: {"ok": True, "lastPollOkAt": at.isoformat()}


def panne():
    raise ConnectionRefusedError("refusé")


def test_ok_rien(tmp_path):
    avis = []
    veiller(ok(), avis.append, T0, tmp_path / "v.json")
    assert avis == []


def test_panne_une_seule_alerte(tmp_path):
    avis = []
    veiller(panne, avis.append, T0, tmp_path / "v.json")
    veiller(panne, avis.append, T0 + timedelta(minutes=5), tmp_path / "v.json")
    assert len(avis) == 1 and avis[0].startswith("Session : Tiron ne reçoit plus depuis ")


def test_retablissement(tmp_path):
    avis = []
    veiller(panne, avis.append, T0, tmp_path / "v.json")
    veiller(ok(T0 + timedelta(minutes=10)), avis.append, T0 + timedelta(minutes=10), tmp_path / "v.json")
    veiller(ok(T0 + timedelta(minutes=15)), avis.append, T0 + timedelta(minutes=15), tmp_path / "v.json")
    assert len(avis) == 2 and avis[1].startswith("Session : Tiron reçoit de nouveau (panne depuis ")


def test_poll_trop_ancien(tmp_path):
    avis = []
    veiller(ok(T0 - timedelta(minutes=16)), avis.append, T0, tmp_path / "v.json")
    assert len(avis) == 1


def test_poll_jamais_fait(tmp_path):
    avis = []
    veiller(lambda: {"ok": True, "lastPollOkAt": None}, avis.append, T0, tmp_path / "v.json")
    assert len(avis) == 1


def test_alerte_reessayee_si_envoi_echoue(tmp_path):
    avis = []

    def notifier(t):
        if not avis:
            avis.append("échec")
            raise OSError("SMTP")
        avis.append(t)
    veiller(panne, notifier, T0, tmp_path / "v.json")
    veiller(panne, notifier, T0 + timedelta(minutes=5), tmp_path / "v.json")
    assert len(avis) == 2 and avis[1].startswith("Session : Tiron ne reçoit plus")


def test_etat_illisible(tmp_path):
    (tmp_path / "v.json").write_text("{pas du json")
    avis = []
    veiller(panne, avis.append, T0, tmp_path / "v.json")
    assert len(avis) == 1


def test_jeton_absent(tmp_path):
    from tiron_mail.veille_session import sonde_broker
    avis = []
    veiller(sonde_broker(""), avis.append, T0, tmp_path / "v.json")
    assert len(avis) == 1 and "jeton absent" in avis[0]


def test_lire_jeton_n_extrait_que_le_jeton(tmp_path):
    f = tmp_path / "secrets.env"
    f.write_text("SESSION_BOT_MNEMONIC='mot1 mot2'\nTIRON_SESSION_BROKER_TOKEN='abc123'\nAUTRE=x\n")
    assert lire_jeton(f) == "abc123"
    f.write_text("AUTRE=x\n")
    assert lire_jeton(f) == ""
