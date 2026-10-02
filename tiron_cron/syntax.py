# tiron_cron/syntax.py
"""Syntaxe normalisée de /cron (cf. docs/superpowers/specs/2026-10-01-cron-tiron-design.md).
Le contenu n'est jamais découpé par shlex : il est pris tel quel (apostrophes)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

CANAUX = ("telegram", "whatsapp", "sms", "obsidian")
TYPES = ("rappel", "commande", "consigne")
# Lecture ou écriture sans risque de perte ; /inbox s'ajoutera avec l'agent courrier.
COMMANDES_PLANIFIABLES = frozenset({"/wikistatus", "/tags", "/relire", "/q", "/r", "/ingest", "/kbupdate"})

_CRON_FIELD = re.compile(r"^[0-9*/,\-]+$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_DUREE = re.compile(r"^(\d+)[mhd]$")
USAGE_ADD = "usage : /cron add <quand> [<canal>] <type> <contenu>"


class SyntaxeInvalide(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    quand_kind: str  # "cron" | "at"
    quand: str
    canal: str | None
    type: str
    contenu: str


@dataclass(frozen=True)
class Gestion:
    action: str  # "list" | "rm" | "pause" | "reprendre"
    id: str | None


def _split_first(s: str) -> tuple[str, str]:
    s = s.lstrip()
    if not s:
        return "", ""
    if s[0] in "\"'":
        end = s.find(s[0], 1)
        if end < 0:
            raise SyntaxeInvalide("guillemet non fermé")
        return s[1:end], s[end + 1:]
    parts = s.split(None, 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def parse_quand(q: str) -> tuple[str, str]:
    fields = q.split()
    if len(fields) == 5 and all(_CRON_FIELD.match(f) for f in fields):
        return "cron", " ".join(fields)
    if _ISO.match(q):
        try:
            datetime.fromisoformat(q)
        except ValueError:
            pass
        else:
            return "at", q
    m = _DUREE.match(q)
    if m and int(m.group(1)) > 0:
        return "at", q
    raise SyntaxeInvalide(
        f"<quand> invalide : {q!r} (attendu : \"0 8 * * 1\", 2026-10-02T09:00 ou 20m)")


def parse_command(args: str) -> Job | Gestion:
    action, rest = _split_first(args)
    action = action.lower()
    if action == "list":
        return Gestion("list", None)
    if action in ("rm", "pause", "reprendre"):
        id_, _ = _split_first(rest)
        if not id_:
            raise SyntaxeInvalide(f"usage : /cron {action} <id>")
        return Gestion(action, id_)
    if action != "add":
        raise SyntaxeInvalide(f"action inconnue : {action!r} (add, list, rm, pause, reprendre)")
    quand_raw, rest = _split_first(rest)
    if not quand_raw:
        raise SyntaxeInvalide(USAGE_ADD)
    kind, quand = parse_quand(quand_raw)
    tok, rest = _split_first(rest)
    canal = None
    if tok.lower() in CANAUX:
        canal = tok.lower()
        tok, rest = _split_first(rest)
    type_ = tok.lower()
    if type_ not in TYPES:
        raise SyntaxeInvalide(f"type inconnu : {tok!r} (rappel, commande, consigne)")
    contenu = rest.strip()
    if not contenu:
        raise SyntaxeInvalide("contenu vide")
    # Une consigne qui commence par / serait exécutée comme une commande au
    # déclenchement (derisk-deleg) : même liste blanche que le type commande.
    if (type_ == "commande" or contenu.startswith("/")) and contenu.split()[0] not in COMMANDES_PLANIFIABLES:
        raise SyntaxeInvalide(
            f"commande non planifiable : {contenu.split()[0]} "
            f"(autorisées : {', '.join(sorted(COMMANDES_PLANIFIABLES))})")
    return Job(kind, quand, canal, type_, contenu)


def to_args(job: Job) -> str:
    quand = f'"{job.quand}"' if job.quand_kind == "cron" else job.quand
    parts = ["add", quand] + ([job.canal] if job.canal else []) + [job.type, job.contenu]
    return " ".join(parts)
