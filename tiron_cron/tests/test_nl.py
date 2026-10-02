# tiron_cron/tests/test_nl.py
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tiron_cron.nl import parse_nl
from tiron_cron.syntax import Job

NOW = datetime(2026, 10, 2, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))  # vendredi


@pytest.mark.parametrize("text,expected", [
    ("rappelle-moi chaque lundi à 8h de payer le loyer par SMS",
     Job("cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")),
    ("Tiron, tous les jours à 7h30 résume les nouveautés du wiki",
     Job("cron", "30 7 * * *", None, "consigne", "résume les nouveautés du wiki")),
    ("rappelle-moi demain à 9h d'appeler le garagiste",
     Job("at", "2026-10-03T09:00", None, "rappel", "appeler le garagiste")),
    ("dans 20 minutes rappelle-moi de sortir le gâteau",
     Job("at", "20m", None, "rappel", "sortir le gâteau")),
    ("le 15 octobre à 14h rappelle-moi la réunion budget sur WhatsApp",
     Job("at", "2026-10-15T14:00", "whatsapp", "rappel", "la réunion budget")),
    ("le 1 mars à 9h rappelle-moi de renouveler l'assurance",
     Job("at", "2027-03-01T09:00", None, "rappel", "renouveler l'assurance")),
    ("tous les dimanches à 18h /wikistatus",
     Job("cron", "0 18 * * 0", None, "commande", "/wikistatus")),
    ("dans 2 heures envoie-moi par texto : vérifier le four",
     Job("at", "2h", "sms", "consigne", "envoie-moi : vérifier le four")),
])
def test_formes_reconnues(text, expected):
    assert parse_nl(text, NOW) == expected


@pytest.mark.parametrize("text", [
    "rappelle-moi de payer le loyer",                 # pas de moment
    "aujourd'hui à 8h rappelle-moi de partir",        # déjà passé
    "chaque lundi à 25h rappelle-moi x",              # heure invalide
    "le 31 février à 9h rappelle-moi x",              # date invalide
    "chaque lundi à 8h",                              # contenu vide
    "rappelle-moi lundi prochain vers 8 heures de x", # forme non couverte → repli
])
def test_non_compris(text):
    assert parse_nl(text, NOW) is None
