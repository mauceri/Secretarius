"""Faux serveurs IMAP et SMTP pour les tests de tiron_mail (aucun réseau)."""
import email
import imaplib
from email.message import EmailMessage
from email.utils import parsedate_to_datetime

import pytest

from tiron_mail.config import Compte


def brut(n: int) -> bytes:
    m = EmailMessage()
    m["From"] = f"Expéditeur {n} <exp{n}@ex.fr>"
    m["To"] = "moi@gmail.com"
    m["Subject"] = f"Message {n}"
    m["Date"] = f"Tue, 0{n} Oct 2026 09:00:00 +0200"
    m["Message-ID"] = f"<m{n}@ex.fr>"
    m.set_content(f"Corps du message {n}.")
    return m.as_bytes()


class FauxIMAP:
    def __init__(self, messages: dict):
        self.messages = messages
        self.recherches: list = []

    def login(self, utilisateur, mot_de_passe):
        return "OK", [b"connecte"]

    def select(self, boite="INBOX", readonly=False):
        return "OK", [str(len(self.messages)).encode()]

    def uid(self, commande, *args):
        if commande == "search":
            self.recherches.append(args[1:])
            return "OK", [" ".join(str(u) for u in sorted(self.messages)).encode()]
        if commande == "fetch" and args[1] == "(INTERNALDATE)":
            lignes = []
            for u in (int(x) for x in args[0].split(",")):
                quand = parsedate_to_datetime(email.message_from_bytes(self.messages[u])["Date"])
                lignes.append(f"{u} (UID {u} INTERNALDATE {imaplib.Time2Internaldate(quand)})".encode())
            return "OK", lignes
        if commande == "fetch":
            u = int(args[0])
            if u not in self.messages:
                return "OK", [None]
            return "OK", [(f"{u} (UID {u} RFC822 {{1}}".encode(), self.messages[u]), b")"]
        raise AssertionError(commande)

    def logout(self):
        return "BYE", [b""]


@pytest.fixture
def faux_imap():
    serveur = FauxIMAP({1: brut(1), 2: brut(2), 3: brut(3)})

    def connecter(compte):
        return serveur
    connecter.serveur = serveur
    return connecter


@pytest.fixture
def compte_gmail():
    return Compte(nom="gmail", prefixe="g", adresse="moi@gmail.com", utilisateur="moi@gmail.com",
                  hote_imap="imap.gmail.com", port_imap=993, securite_imap="ssl",
                  hote_smtp="smtp.gmail.com", port_smtp=465, securite_smtp="ssl",
                  secret_env="GMAIL_APP_PASSWORD")
