"""Accès IMAP à une boîte : lister, chercher, lire (lecture seule, INBOX)."""
from __future__ import annotations

import imaplib
import re
import ssl
import time
from collections.abc import Mapping
from datetime import date, timedelta

from tiron_mail.config import Compte
from tiron_mail.message import analyser, resume

_MOIS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class ErreurBoite(RuntimeError):
    pass


def _contexte_tls(hote: str) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if hote in ("127.0.0.1", "localhost"):
        # Proton Bridge écoute en local avec un certificat auto-signé : la
        # vérification est levée pour l'hôte local SEULEMENT.
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _connecter_imap(compte: Compte):
    if compte.securite_imap == "ssl":
        return imaplib.IMAP4_SSL(compte.hote_imap, compte.port_imap,
                                 ssl_context=_contexte_tls(compte.hote_imap), timeout=30)
    conn = imaplib.IMAP4(compte.hote_imap, compte.port_imap, timeout=30)
    conn.starttls(ssl_context=_contexte_tls(compte.hote_imap))
    return conn


def _date_imap(d: date) -> str:
    return f"{d.day:02d}-{_MOIS[d.month - 1]}-{d.year}"


# Mots sans valeur de recherche, retirés quand la demande arrive en phrase
# (« Ai-je reçu un mail de Paul sur le chantier ? » → Paul, chantier).
_MOTS_VIDES = {
    "ai", "je", "j", "reçu", "recu", "un", "une", "de", "du", "des", "le", "la", "les", "l",
    "sur", "à", "a", "au", "aux", "pour", "par", "avec", "et", "ou", "en", "dans",
    "mail", "mails", "email", "emails", "courriel", "courriels", "message", "messages",
    "cherche", "cherchez", "chercher", "retrouve", "retrouvez", "trouve", "trouvez",
    "moi", "me", "mes", "mon", "ma", "vous", "tu", "t", "il", "y", "est", "ce", "que", "qui",
    "quoi", "stp", "svp", "merci", "est-ce", "ai-je", "as-tu", "avez-vous",
}


def criteres_imap(texte: str, aujourd_hui: date) -> list[str]:
    """« de:paul objet:devis depuis:7j chantier » → critères IMAP SEARCH."""
    crit: list[str] = []
    libres: list[str] = []
    for mot in texte.split():
        cle, _, val = mot.partition(":")
        val = val.replace('"', "")
        if cle == "de" and val:
            crit += ["FROM", f'"{val}"']
        elif cle == "objet" and val:
            crit += ["SUBJECT", f'"{val}"']
        elif cle == "depuis" and val[:-1].isdigit() and val.endswith("j"):
            crit += ["SINCE", _date_imap(aujourd_hui - timedelta(days=int(val[:-1])))]
        else:
            propre = mot.strip("?!.,;:«»()'\"")
            if propre and propre.lower() not in _MOTS_VIDES:
                libres.append(propre)
    for mot in libres:  # chaque mot doit figurer (ET), pas la phrase entière
        crit += ["TEXT", f'"{mot}"']
    return crit or ["ALL"]


class Boite:
    def __init__(self, compte: Compte, env: Mapping[str, str], connecter=_connecter_imap) -> None:
        self.compte = compte
        self.env = env
        self._connecter = connecter

    def _ouvrir(self):
        mdp = self.env.get(self.compte.secret_env, "")
        if not mdp:
            raise ErreurBoite(f"boîte {self.compte.nom} : mot de passe absent ({self.compte.secret_env})")
        try:
            conn = self._connecter(self.compte)
            conn.login(self.compte.utilisateur, mdp)
            conn.select("INBOX", readonly=True)
            return conn
        except (OSError, imaplib.IMAP4.error) as e:
            raise ErreurBoite(f"boîte {self.compte.nom} injoignable : {e}") from e

    def _fetch(self, conn, uid: int, spec: str) -> bytes:
        typ, data = conn.uid("fetch", str(uid), spec)
        if typ != "OK" or not data or data[0] is None:
            raise ErreurBoite(f"message introuvable : {self.compte.prefixe}:{uid}")
        return data[0][1]

    def _resumes(self, conn, uids: list[int]) -> list[dict]:
        out = []
        for u in uids:
            ident = f"{self.compte.prefixe}:{u}"
            out.append({"id": ident, "resume": resume(analyser(self._fetch(conn, u, "(RFC822.HEADER)")), ident)})
        return out

    def _par_date(self, conn, uids: list[int]) -> list[int]:
        """UIDs triés du plus ancien au plus récent selon la date de réception.
        Pour les boîtes dont les UIDs ne suivent pas les dates (tri_par_date)."""
        if not uids:
            return []
        typ, data = conn.uid("fetch", ",".join(map(str, uids)), "(INTERNALDATE)")
        dates: dict[int, float] = {}
        for ligne in data if typ == "OK" else []:
            ligne = ligne[0] if isinstance(ligne, tuple) else ligne
            u = re.search(rb"UID (\d+)", ligne or b"")
            t = imaplib.Internaldate2tuple(ligne or b"")
            if u and t:
                dates[int(u.group(1))] = time.mktime(t)
        return sorted(uids, key=lambda u: (dates.get(u, 0.0), u))

    def _chercher(self, crit: list[str], n: int) -> list[dict]:
        conn = self._ouvrir()
        try:
            typ, data = conn.uid("search", None, *crit)
            uids = [int(x) for x in (data[0] or b"").split()] if typ == "OK" else []
            ordre = self._par_date(conn, uids) if self.compte.tri_par_date else sorted(uids)
            return self._resumes(conn, ordre[-n:][::-1])
        except (OSError, imaplib.IMAP4.error) as e:
            raise ErreurBoite(f"boîte {self.compte.nom} injoignable : {e}") from e
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    def lister(self, n: int = 10) -> list[dict]:
        return self._chercher(["ALL"], n)

    def chercher(self, criteres: str, n: int = 20, aujourd_hui: date | None = None) -> list[dict]:
        return self._chercher(criteres_imap(criteres, aujourd_hui or date.today()), n)

    def lire(self, uid: str) -> dict:
        conn = self._ouvrir()
        try:
            return analyser(self._fetch(conn, int(uid), "(RFC822)"))
        except ValueError as e:
            raise ErreurBoite(f"identifiant invalide : {uid}") from e
        finally:
            try:
                conn.logout()
            except Exception:
                pass
