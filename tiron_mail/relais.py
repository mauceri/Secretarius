"""Relais sur Session des courriels d'expéditeurs listés dans [releve] relayes (spec
2026-11-25-relais-grok). Information seulement : rien n'en est jamais exécuté."""
from __future__ import annotations

import re
from email.message import Message
from email.utils import getaddresses

_DMARC = re.compile(r"\bdmarc=(\w+)[^;]*?\bheader\.from=([^\s;]+)", re.IGNORECASE)
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
    dmarc = [r for v in proton for r in _DMARC.findall(v)]
    if not dmarc or any(res.lower() != "pass" or dom.lower() != domaine for res, dom in dmarc):
        return f"DMARC non validé par Proton pour {domaine}"
    if not any(res.lower() == "pass" and dom.lower() == domaine for v in proton for res, dom in _DKIM.findall(v)):
        return f"aucune signature DKIM valide de {domaine}"
    return ""
