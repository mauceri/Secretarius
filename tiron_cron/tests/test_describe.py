# tiron_cron/tests/test_describe.py
from tiron_cron.describe import describe
from tiron_cron.syntax import Job


def test_hebdomadaire():
    job = Job("cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")
    assert describe(job, "sms", "Europe/Paris") == \
        "Chaque lundi à 08:00 (Europe/Paris), sms, rappel : payer le loyer"


def test_quotidien():
    job = Job("cron", "30 7 * * *", None, "consigne", "résume le wiki")
    assert describe(job, "telegram", "Europe/Paris").startswith("Chaque jour à 07:30 (Europe/Paris), telegram")


def test_dimanche_0_et_7():
    for dow in ("0", "7"):
        assert describe(Job("cron", f"0 9 * * {dow}", None, "rappel", "x"), "telegram", "UTC") \
            .startswith("Chaque dimanche à 09:00")


def test_cron_non_humanisable():
    job = Job("cron", "*/15 * * * *", None, "rappel", "x")
    assert describe(job, "telegram", "Europe/Paris").startswith("Selon « */15 * * * * »")


def test_iso():
    job = Job("at", "2026-10-02T09:00", None, "rappel", "x")
    assert describe(job, "telegram", "Europe/Paris").startswith("Le 02/10/2026 à 09:00 (Europe/Paris)")


def test_duree():
    assert describe(Job("at", "20m", None, "rappel", "x"), "sms", "UTC").startswith("Dans 20 min (UTC)")
    assert describe(Job("at", "2h", None, "rappel", "x"), "sms", "UTC").startswith("Dans 2 h")
    assert describe(Job("at", "1d", None, "rappel", "x"), "sms", "UTC").startswith("Dans 1 j")
