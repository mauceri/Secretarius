"""Relève de tiron@secretarius.co : n'exécute que les commandes des messages de
l'utilisateur authentifiés par Proton (spec courrier, § Relève). Le corps d'un
message refusé n'est jamais transmis."""
from __future__ import annotations

import re
from email.message import Message
from email.utils import getaddresses, parseaddr

_DMARC = re.compile(r"\bdmarc=(\w+)")


def authentifier(m: Message, proprietaires: tuple) -> str:
    """"" si le message vient de l'utilisateur et que Proton l'authentifie, sinon le motif.
    Proton remplace les X-Pm-* fournis par l'expéditeur (vérifié le 2026-10-10)."""
    froms = getaddresses([str(v) for v in m.get_all("From", [])])
    if len(froms) != 1:
        return "expéditeur absent ou multiple"
    exp = froms[0][1].lower()
    if exp not in proprietaires:
        return f"expéditeur non autorisé ({exp})"
    if parseaddr(str(m.get("Return-Path", "")))[1].lower() != exp:
        return "Return-Path différent de l'expéditeur"
    origine = [str(v).strip() for v in m.get_all("X-Pm-Origin", [])]
    chiffrement = [str(v).strip() for v in m.get_all("X-Pm-Content-Encryption", [])]
    if origine == ["internal"]:
        return "" if chiffrement == ["end-to-end"] else "message interne non chiffré de bout en bout"
    if origine != ["external"]:
        return "origine Proton inconnue"
    dmarc = [r.group(1) for v in m.get_all("Authentication-Results", [])
             if str(v).strip().startswith("mail.protonmail.ch;") and (r := _DMARC.search(str(v)))]
    if not dmarc or any(d != "pass" for d in dmarc):
        return "DMARC non validé par Proton"
    return ""


def commandes(objet: str, texte: str) -> list[str]:
    """Objet et lignes du corps qui commencent par « / » (les lignes citées commencent par « > »)."""
    return [l.strip() for l in [objet, *texte.splitlines()] if l.strip().startswith("/")]
