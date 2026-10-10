"""Faux serveurs IMAP et SMTP pour les tests de tiron_mail (aucun réseau)."""
import email
import imaplib
from email.message import EmailMessage
from email.utils import parsedate_to_datetime

import pytest

from tiron_mail.config import Compte


def brut(n: int, quand: str | None = None, **entetes) -> bytes:
    m = EmailMessage()
    m["From"] = f"Expéditeur {n} <exp{n}@ex.fr>"
    m["To"] = "moi@gmail.com"
    m["Subject"] = f"Message {n}"
    m["Date"] = quand or f"Tue, 0{n} Oct 2026 09:00:00 +0200"
    for cle, val in entetes.items():
        del m[cle.replace("_", "-")]
        m[cle.replace("_", "-")] = val
    m["Message-ID"] = f"<m{n}@ex.fr>"
    m.set_content(f"Corps du message {n}.")
    return m.as_bytes()


class FauxIMAP:
    """Comme imaplib, encode les arguments en ASCII, sauf après ENABLE UTF8=ACCEPT
    (que seul un serveur qui l'annonce, utf8=True, accepte)."""

    def __init__(self, messages: dict, utf8: bool = False):
        self.messages = messages
        self.utf8 = utf8
        self.utf8_actif = False
        self.recherches: list = []
        self.datees: list = []
        self.ranges: list = []
        self.capabilities = ("IMAP4REV1",)

    def capability(self):
        return "OK", [b"IMAP4rev1 UTF8=ACCEPT" if self.utf8 else b"IMAP4rev1"]

    def enable(self, capacite):
        assert capacite in self.capabilities, "imaplib refuse ENABLE si la capacité n'est pas connue"
        self.utf8_actif = True
        return "OK", [b""]

    def create(self, dossier):
        return "OK", [b""]

    def login(self, utilisateur, mot_de_passe):
        return "OK", [b"connecte"]

    def select(self, boite="INBOX", readonly=False):
        return "OK", [str(len(self.messages)).encode()]

    def uid(self, commande, *args):
        for a in args:
            if isinstance(a, str):
                a.encode("utf-8" if self.utf8_actif else "ascii")
        if commande == "search":
            self.recherches.append(args[1:])
            return "OK", [" ".join(str(u) for u in sorted(self.messages)).encode()]
        if commande == "fetch" and args[1] == "(INTERNALDATE)":
            lignes = []
            for u in (int(x) for x in args[0].split(",")):
                self.datees.append(u)
                quand = parsedate_to_datetime(email.message_from_bytes(self.messages[u])["Date"])
                lignes.append(f"{u} (UID {u} INTERNALDATE {imaplib.Time2Internaldate(quand)})".encode())
            return "OK", lignes
        if commande == "fetch":
            u = int(args[0])
            if u not in self.messages:
                return "OK", [None]
            return "OK", [(f"{u} (UID {u} RFC822 {{1}}".encode(), self.messages[u]), b")"]
        if commande == "move":
            self.ranges.append((int(args[0]), args[1]))
            self.messages.pop(int(args[0]))
            return "OK", [b""]
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
