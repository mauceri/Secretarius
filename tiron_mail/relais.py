"""Relais sur Session des courriels d'expéditeurs listés dans [releve] relayes (spec
2026-11-25-relais-grok). Information seulement : rien n'en est jamais exécuté."""
from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from email.message import Message
from email.utils import getaddresses
from pathlib import Path

from tiron_mail.veille_session import lire_cle

_DMARC = re.compile(r"\bdmarc=(\w+)", re.IGNORECASE)
_DMARC_FROM = re.compile(r"\bdmarc=\w+[^;]*?\bheader\.from=([^\s;]+)", re.IGNORECASE)
_DKIM = re.compile(r"\bdkim=(\w+)[^;]*?\bheader\.d=([^\s;]+)", re.IGNORECASE)


def selecteur_relaye(adresse: str, relayes: tuple) -> str | None:
    """None si l'adresse n'est pas relayée, "" sans sélecteur, sinon le sélecteur."""
    adresse = adresse.strip().lower()
    for entree in relayes:
        local, _, domaine = entree.lower().partition("@")
        m = re.fullmatch(re.escape(local) + r"(?:\+([a-z0-9-]+))?@" + re.escape(domaine), adresse)
        if local and domaine and m:
            return m.group(1) or ""
    return None


def authentifier_relaye(m: Message) -> str:
    """"" si Proton a authentifié le domaine de l'expéditeur (DMARC + DKIM alignée), sinon le motif."""
    froms = getaddresses([str(v) for v in m.get_all("From", [])])
    if len(froms) != 1:
        return "expéditeur absent ou multiple"
    domaine = froms[0][1].lower().rpartition("@")[2]
    if [str(v).strip() for v in m.get_all("X-Pm-Origin", [])] != ["external"]:
        return "origine Proton inattendue"
    if not str(m.get("Message-ID", "")).strip():
        return "message sans Message-ID"
    proton = [" ".join(str(v).split()) for v in m.get_all("Authentication-Results", [])
              if str(v).strip().startswith("mail.protonmail.ch;")]
    resultats = [r.lower() for v in proton for r in _DMARC.findall(v)]  # tous, même sans header.from
    domaines = [d.lower() for v in proton for d in _DMARC_FROM.findall(v)]
    if not resultats or any(r != "pass" for r in resultats) or not domaines or any(d != domaine for d in domaines):
        return f"DMARC non validé par Proton pour {domaine}"
    if not any(res.lower() == "pass" and dom.lower() == domaine for v in proton for res, dom in _DKIM.findall(v)):
        return f"aucune signature DKIM valide de {domaine}"
    return ""


# --- Garde d'injection, mise en forme, envoi sur Session ---
GARDE = "http://127.0.0.1:8990/check"
BROKER = "http://127.0.0.1:8091/send"
_MAX = 3500


def garde_injection(url: str = GARDE, ouvrir=urllib.request.urlopen) -> Callable[[str], dict]:
    """Même garde que check_email : regex + DeBERTa, aucun LLM."""
    def verifier(texte: str) -> dict:
        req = urllib.request.Request(url, data=json.dumps({"type": "text", "content": texte}).encode(),
                                     headers={"Content-Type": "application/json"})
        with ouvrir(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    return verifier


def message_relais(selecteur: str, objet: str, texte: str, garde: Callable[[str], dict]) -> str:
    objet = objet.strip()
    objet = objet[:200] + "…" if len(objet) > 200 else objet
    tete = f"Courriel de Grok Bot{f' ({selecteur})' if selecteur else ''} — « {objet} »"
    try:
        v = garde(f"{objet}\n\n{texte}")  # l'objet aussi passe par le garde
    except Exception:  # garde injoignable : jamais de texte non vérifié sur Session
        return f"{tete}\n\ncontenu non vérifié (garde injoignable), consultez la boîte de Tiron"
    if v.get("blocked"):
        return f"{tete}\n\ncontenu bloqué par le détecteur d'injection : {v.get('reason', 'contenu suspect')}"
    corps = texte.strip()  # verdict rendu sur objet + texte ; en type text le garde ne réécrit pas
    if len(corps) > _MAX:
        corps = corps[:_MAX] + "… (suite dans la boîte de Tiron)"
    alerte = "Attention : contenu potentiellement suspect.\n\n" if v.get("risk") == "medium" else ""
    return f"{tete}\n\n{alerte}{corps}"


def envoyeur_session(secrets: Path, url: str = BROKER, ouvrir=urllib.request.urlopen) -> Callable[[str], None]:
    """Envoi direct au broker Session ; ne lit dans secrets.env que le jeton et l'identifiant."""
    def envoyer(texte: str) -> None:
        jeton = lire_cle(secrets, "TIRON_SESSION_BROKER_TOKEN")
        dest = lire_cle(secrets, "TIRON_SESSION_ID")
        if not jeton or not dest:
            raise RuntimeError("TIRON_SESSION_BROKER_TOKEN ou TIRON_SESSION_ID absent de secrets.env")
        req = urllib.request.Request(url, data=json.dumps({"to": dest, "text": texte}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {jeton}"})
        with ouvrir(req, timeout=30) as r:
            if r.status != 200:
                raise RuntimeError(f"broker Session : HTTP {r.status}")
    return envoyer


def relayeur(session: Callable[[str], None], avis: Callable[[str], None]) -> Callable[[str], None]:
    """Session d'abord ; sinon courriel d'avis ; lève si les deux échouent."""
    def relayer(texte: str) -> None:
        try:
            session(texte)
        except Exception:
            avis("(Session indisponible) " + texte)
    return relayer
