# tiron_cron/describe.py
"""Reformulation française d'une tâche, montrée avant /confirm."""
from __future__ import annotations

from datetime import datetime

from tiron_cron.syntax import Job

_JOURS = {"0": "dimanche", "1": "lundi", "2": "mardi", "3": "mercredi",
          "4": "jeudi", "5": "vendredi", "6": "samedi", "7": "dimanche"}
_UNITES = {"m": "min", "h": "h", "d": "j"}


def _quand(job: Job) -> str:
    if job.quand_kind == "cron":
        m, h, dom, mon, dow = job.quand.split()
        if m.isdigit() and h.isdigit() and dom == "*" and mon == "*":
            heure = f"{int(h):02d}:{int(m):02d}"
            if dow == "*":
                return f"chaque jour à {heure}"
            if dow in _JOURS:
                return f"chaque {_JOURS[dow]} à {heure}"
        return f"selon « {job.quand} »"
    if "T" in job.quand:
        return datetime.fromisoformat(job.quand).strftime("le %d/%m/%Y à %H:%M")
    return f"dans {job.quand[:-1]} {_UNITES[job.quand[-1]]}"


def describe(job: Job, canal: str, tz: str) -> str:
    q = _quand(job)
    return f"{q[0].upper()}{q[1:]} ({tz}), {canal}, {job.type} : {job.contenu}"
