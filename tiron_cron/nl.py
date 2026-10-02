# tiron_cron/nl.py
"""Analyse déterministe des demandes de planification en français courant.
Renvoie None dès qu'un doute subsiste : le repli (fallback.py) prend le relais,
puis la reformulation + /confirm laisse le dernier mot à l'utilisateur."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from tiron_cron.syntax import COMMANDES_PLANIFIABLES, Job

_JOURS = {"lundi": 1, "mardi": 2, "mercredi": 3, "jeudi": 4, "vendredi": 5, "samedi": 6, "dimanche": 0}
_MOIS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
         "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
         "décembre": 12, "decembre": 12}
_H = r"(\d{1,2})\s*h\s*(\d{2})?"
_I = re.IGNORECASE
_CANAL = re.compile(r"\b(?:par|en|sur|via)\s+(sms|texto|whatsapp|telegram|obsidian)\b", _I)
_RELATIF = re.compile(r"\bdans\s+(\d+)\s*(minutes?|min|heures?|h|jours?)\b", _I)
_HEBDO = re.compile(r"\b(?:chaque|tous\s+les)\s+(" + "|".join(_JOURS) + r")s?\s+à\s+" + _H, _I)
_QUOTIDIEN = re.compile(r"\b(?:chaque\s+jour|tous\s+les\s+jours|chaque\s+matin|tous\s+les\s+matins|"
                        r"quotidiennement)\s+à\s+" + _H, _I)
_PROCHE = re.compile(r"\b(aujourd'hui|demain|après-demain)\s+à\s+" + _H, _I)
_DATE = re.compile(r"\ble\s+(\d{1,2})\s+(" + "|".join(_MOIS) + r")(?:\s+(\d{4}))?\s+à\s+" + _H, _I)
_TIRON = re.compile(r"^\s*tiron\s*[,:]?\s*", _I)
_RAPPEL = re.compile(r"^\s*(?:rappelle[- ]moi|rappelle[- ]nous|rappelez[- ]moi)\s*(?:de\s+|d'|que\s+|qu'|:\s*)?", _I)
_CMD = re.compile(r"(?<!\S)(/[a-z]+)\b")


def _hm(h: str, m: str | None) -> tuple[int, int] | None:
    hh, mm = int(h), int(m or 0)
    return (hh, mm) if 0 <= hh <= 23 and 0 <= mm <= 59 else None


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def _quand(text: str, now: datetime) -> tuple[str, str, re.Match] | None:
    if m := _RELATIF.search(text):
        unite = m.group(2).lower()[0]
        return "at", f"{int(m.group(1))}{'d' if unite == 'j' else unite}", m
    if m := _HEBDO.search(text):
        hm = _hm(m.group(2), m.group(3))
        return ("cron", f"{hm[1]} {hm[0]} * * {_JOURS[m.group(1).lower()]}", m) if hm else None
    if m := _QUOTIDIEN.search(text):
        hm = _hm(m.group(1), m.group(2))
        return ("cron", f"{hm[1]} {hm[0]} * * *", m) if hm else None
    if m := _PROCHE.search(text):
        hm = _hm(m.group(2), m.group(3))
        if not hm:
            return None
        decalage = {"aujourd'hui": 0, "demain": 1, "après-demain": 2}[m.group(1).lower()]
        dt = (now + timedelta(days=decalage)).replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
        return ("at", _iso(dt), m) if dt > now else None
    if m := _DATE.search(text):
        hm = _hm(m.group(4), m.group(5))
        if not hm:
            return None
        annee = int(m.group(3)) if m.group(3) else now.year
        try:
            dt = now.replace(year=annee, month=_MOIS[m.group(2).lower()], day=int(m.group(1)),
                             hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if dt <= now and not m.group(3):
                dt = dt.replace(year=annee + 1)
        except ValueError:
            return None
        return ("at", _iso(dt), m) if dt > now else None
    return None


def _nettoie(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ,.;:!")


def parse_nl(text: str, now: datetime) -> Job | None:
    canal = None
    if m := _CANAL.search(text):
        canal = {"texto": "sms"}.get(m.group(1).lower(), m.group(1).lower())
        text = text[:m.start()] + " " + text[m.end():]
    q = _quand(text, now)
    if q is None:
        return None
    kind, quand, m = q
    reste = _nettoie(_TIRON.sub("", _nettoie(text[:m.start()] + " " + text[m.end():])))
    if (c := _CMD.search(reste)) and c.group(1) in COMMANDES_PLANIFIABLES:
        type_, contenu = "commande", reste[c.start():]
    elif r := _RAPPEL.match(reste):
        type_, contenu = "rappel", reste[r.end():]
    else:
        type_, contenu = "consigne", reste
    contenu = _nettoie(contenu)
    return Job(kind, quand, canal, type_, contenu) if contenu else None
