"""Accès IMAP à une boîte : lister, chercher, lire (lecture seule, INBOX)."""
from __future__ import annotations

import hashlib
import hmac
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


def verifier_empreinte(sock, compte: Compte) -> None:
    """Le Bridge a un certificat auto-signé : sans vérification par autorité, on
    compare son empreinte à celle notée dans la configuration."""
    if not compte.empreinte_tls:
        return
    vue = hashlib.sha256(sock.getpeercert(binary_form=True) or b"").hexdigest()
    attendue = compte.empreinte_tls.replace(":", "").lower()
    if not hmac.compare_digest(vue, attendue):
        raise ErreurBoite(f"boîte {compte.nom} : certificat inattendu ({vue[:16]}…), connexion refusée")


def _connecter_imap(compte: Compte):
    if compte.securite_imap == "ssl":
        conn = imaplib.IMAP4_SSL(compte.hote_imap, compte.port_imap,
                                 ssl_context=_contexte_tls(compte.hote_imap), timeout=30)
    else:
        conn = imaplib.IMAP4(compte.hote_imap, compte.port_imap, timeout=30)
        conn.starttls(ssl_context=_contexte_tls(compte.hote_imap))
    verifier_empreinte(conn.sock, compte)
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


def _activer_utf8(conn) -> bool:
    """Active UTF8=ACCEPT (RFC 6855) si le serveur l'annonce : Gmail oui, Proton
    Bridge non. imaplib ne relit pas les capacités après login, d'où la relecture."""
    typ, data = conn.capability()
    caps = tuple((data[0] or b"").decode().upper().split()) if typ == "OK" and data else ()
    if "UTF8=ACCEPT" not in caps:
        return False
    conn.capabilities = caps
    conn.enable("UTF8=ACCEPT")
    return True


def _partie_ascii(valeur: str) -> str:
    """« "réunion" » → « "union" » : plus longue partie ASCII d'au moins 3
    caractères (la recherche IMAP est par sous-chaîne), "" s'il n'y en a pas."""
    morceaux = re.findall(r"[\x20-\x7e]+", valeur.strip('"'))
    meilleur = max(morceaux, key=len, default="").strip()
    return f'"{meilleur}"' if len(meilleur) >= 3 else ""


def sans_accents(crit: list[str]) -> list[str]:
    """Critères pour un serveur sans UTF-8 : chaque valeur non ASCII est réduite à
    sa partie ASCII, ou retirée avec sa clé. ErreurBoite s'il ne reste rien."""
    out: list[str] = []
    for i in range(0, len(crit) - 1, 2):
        cle, val = crit[i], crit[i + 1]
        if not val.isascii():
            val = _partie_ascii(val)
            if not val:
                continue
        out += [cle, val]
    if crit != ["ALL"] and not out:
        raise ErreurBoite("recherche impossible : ce serveur ne gère pas les accents, "
                          "reformulez avec un mot sans accent")
    return out or crit


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
            jours = min(int(val[:-1]), 36500)  # au-delà, timedelta déborde
            crit += ["SINCE", _date_imap(aujourd_hui - timedelta(days=jours))]
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
        self._utf8 = False

    def _ouvrir(self):
        mdp = self.env.get(self.compte.secret_env, "")
        if not mdp:
            raise ErreurBoite(f"boîte {self.compte.nom} : mot de passe absent ({self.compte.secret_env})")
        try:
            conn = self._connecter(self.compte)
            conn.login(self.compte.utilisateur, mdp)
            self._utf8 = _activer_utf8(conn)
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

    def _par_date(self, conn, uids: list[int], n: int) -> list[int]:
        """UIDs triés du plus ancien au plus récent selon la date de réception.
        Pour les boîtes dont les UIDs ne suivent pas les dates (tri_par_date).
        Seuls les deux bouts sont datés : le Bridge numérote sa synchronisation
        à rebours (récents en bas), puis les arrivées à la suite (récents en
        haut) ; dater toute une grosse boîte prendrait des dizaines de secondes."""
        if not uids:
            return []
        k = max(5 * n, 50)
        uids = sorted(uids)
        if len(uids) > 2 * k:
            uids = uids[:k] + uids[-k:]
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
            if not self._utf8:
                crit = sans_accents(crit)
            typ, data = conn.uid("search", None, *crit)
            uids = [int(x) for x in (data[0] or b"").split()] if typ == "OK" else []
            ordre = self._par_date(conn, uids, n) if self.compte.tri_par_date else sorted(uids)
            return self._resumes(conn, ordre[-n:][::-1])
        except (OSError, UnicodeError, imaplib.IMAP4.error) as e:
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
        except (OSError, imaplib.IMAP4.error) as e:
            raise ErreurBoite(f"boîte {self.compte.nom} injoignable : {e}") from e
        finally:
            try:
                conn.logout()
            except Exception:
                pass
