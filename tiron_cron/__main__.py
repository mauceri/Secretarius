# tiron_cron/__main__.py
"""CLI JSON de tiron_cron — seul point d'entrée pour derisk-deleg et wiki-lm-server.
Toujours code 0 et une ligne JSON {"ok": bool, "text": str, ...}."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from tiron_cron import fallback
from tiron_cron.deliver_obsidian import deliver
from tiron_cron.describe import describe
from tiron_cron.nl import parse_nl
from tiron_cron.openclaw_cli import OpenClawError, add_job, build_add_argv, manage
from tiron_cron.syntax import CANAUX, USAGE_ADD, Gestion, Job, SyntaxeInvalide, parse_command, to_args


def _job_from_args(args: str) -> Job:
    r = parse_command(args)
    if not isinstance(r, Job):
        raise SyntaxeInvalide(USAGE_ADD)
    return r


def _origine(origin: str) -> str | None:
    o = (origin or "").lower()
    return o if o in CANAUX else None


def _apercu(job: Job, origin: str, tz: str, env, job_key: str, now: datetime) -> tuple[str, list[str]]:
    # Jamais une tâche dans le passé, quelle que soit la source (syntaxe tapée,
    # analyseur, repli) ; revérifié à la création (/confirm jusqu'à 10 min après).
    if job.quand_kind == "at" and "T" in job.quand and \
            datetime.fromisoformat(job.quand).replace(tzinfo=ZoneInfo(tz)) <= now:
        raise SyntaxeInvalide(f"date déjà passée : {job.quand}")
    canal = job.canal or _origine(origin)
    argv = build_add_argv(job, canal, tz, env, job_key)  # valide canal + destinataire dès l'aperçu
    return describe(job, canal, tz), argv


def main(argv=None, runner=subprocess.run, env=None, now=None, http=None, job_key=None) -> int:
    env = os.environ if env is None else env
    tz = env.get("TIRON_TZ", "Europe/Paris")
    now = now or datetime.now(ZoneInfo(tz))
    job_key = job_key or f"tiron:{uuid.uuid4().hex[:8]}"
    p = argparse.ArgumentParser(prog="tiron_cron")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "add"):
        s = sub.add_parser(name)
        s.add_argument("--origin", default="")
        s.add_argument("--args", required=True)
    s = sub.add_parser("nl")
    s.add_argument("--origin", default="")
    s.add_argument("--text", required=True)
    sub.add_parser("manage").add_argument("--args", required=True)
    sub.add_parser("deliver-obsidian").add_argument("--text", required=True)
    a = p.parse_args(argv)

    try:
        if a.cmd == "preview":
            job = _job_from_args(a.args)
            text, _ = _apercu(job, a.origin, tz, env, job_key, now)
            out = {"ok": True, "text": text, "args": to_args(job)}
        elif a.cmd == "add":
            job = _job_from_args(a.args)
            text, cmd = _apercu(job, a.origin, tz, env, job_key, now)
            try:
                full = add_job(cmd, job_key, runner)
            except OpenClawError as e:
                out = {"ok": False, "text": f"Échec de la création : {e}"}
            else:
                out = {"ok": True, "id": full, "text": f"Tâche créée ({full[:8]}) : {text}"}
        elif a.cmd == "nl":
            job, source = parse_nl(a.text, now), "regles"
            if job is None:
                kw = {"http": http} if http else {}
                job, source = fallback.normalize_with_model(a.text, now, env, **kw), "modele"
            if job is None:
                out = {"ok": False, "text": "Planification non comprise. Syntaxe : /cron " + USAGE_ADD.split("/cron ", 1)[1]}
            else:
                job = _job_from_args(to_args(job))  # même validation que la syntaxe tapée
                text, _ = _apercu(job, a.origin, tz, env, job_key, now)
                out = {"ok": True, "text": text, "args": to_args(job), "source": source}
        elif a.cmd == "manage":
            g = parse_command(a.args)
            if not isinstance(g, Gestion):
                raise SyntaxeInvalide("manage : list, rm, pause ou reprendre")
            try:
                out = {"ok": True, "text": manage(g, runner)}
            except OpenClawError as e:
                out = {"ok": False, "text": f"Échec : {e}"}
        else:
            out = {"ok": True, "text": str(deliver(a.text, now, env))}
    except SyntaxeInvalide as e:
        out = {"ok": False, "text": str(e)}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
