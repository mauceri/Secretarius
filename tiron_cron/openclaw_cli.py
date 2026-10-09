# tiron_cron/openclaw_cli.py
"""Traduction d'un Job en appel `openclaw cron`, et gestion des tâches tiron:*.
Les tâches système (ex. skill-collection-review) ne sont jamais visibles ni
modifiables d'ici : toute résolution d'id se fait parmi les tâches tiron:*."""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from tiron_cron.syntax import Gestion, Job, SyntaxeInvalide

DECL_PREFIX = "tiron:"
# Canal Tiron -> identifiant de canal OpenClaw (vérifié par l'essai préalable).
CANAUX_OPENCLAW = {"telegram": "telegram", "whatsapp": "whatsapp", "sms": "sms"}
DESTINATAIRES_ENV = {"telegram": "TIRON_TELEGRAM_CHAT_ID", "whatsapp": "TIRON_WHATSAPP_TO",
                     "sms": "TIRON_SMS_TO"}


class OpenClawError(RuntimeError):
    pass


def secretarius_dir(env: Mapping[str, str]) -> str:
    return env.get("SECRETARIUS_DIR") or str(Path.home() / "Secretarius")


def build_add_argv(job: Job, canal: str | None, tz: str, env: Mapping[str, str],
                   job_key: str) -> list[str]:
    if canal is None:
        raise SyntaxeInvalide("canal non précisé et canal d'origine inconnu (telegram, whatsapp, sms, obsidian)")
    argv = ["openclaw", "cron", "add", "--json", "--name", job_key.replace(":", "-"),
            "--display-name", f"{job.type} : {job.contenu}"[:60], "--declaration-key", job_key]
    if job.quand_kind == "cron":
        argv += ["--cron", job.quand, "--tz", tz, "--exact"]
    else:
        argv += ["--at", job.quand, "--tz", tz, "--delete-after-run"]
    if canal == "obsidian":
        if job.type == "rappel":
            return argv + ["--command-argv", json.dumps(
                ["python3", "-m", "tiron_cron", "deliver-obsidian", "--text", job.contenu]),
                "--command-cwd", secretarius_dir(env), "--no-deliver"]
        # Consigne ou commande : le résultat part en webhook vers wiki-lm-server,
        # qui écrit la note dans Réception/ et prévient sur Telegram (spec courrier).
        url = env.get("TIRON_LIVRAISON_URL") or "http://127.0.0.1:5051/livraison"
        return argv + ["--agent", "main", "--session", "isolated", "--light-context",
                       "--thinking", "off", "--message", job.contenu, "--webhook", url]
    var = DESTINATAIRES_ENV[canal]
    dest = env.get(var)
    if not dest:
        raise SyntaxeInvalide(f"destinataire {canal} non configuré ({var})")
    delivery = ["--announce", "--channel", CANAUX_OPENCLAW[canal], "--to", dest]
    if job.type == "rappel":
        # « Rappel : » pour qu'on le reconnaisse au premier coup d'œil (2026-10-06).
        return argv + ["--command-argv", json.dumps(["printf", "%s\n", f"Rappel : {job.contenu}"])] + delivery
    # Contexte allégé et raisonnement coupé : ~20 s gagnées avec qwen3:8b sur le
    # processeur (essai du 2026-10-07). Sans effet pour une commande, que le hook
    # de derisk-deleg traite avant le modèle.
    return argv + ["--agent", "main", "--session", "isolated", "--light-context",
                   "--thinking", "off", "--message", job.contenu] + delivery


def _run(argv: list[str], runner) -> str:
    # Hors du PATH (service systemd : wiki-lm-server, blocs cron! d'Obsidian),
    # passer par l'enveloppe ~/.local/bin/openclaw, qui charge le jeton du gateway.
    if argv[0] == "openclaw" and shutil.which("openclaw") is None:
        argv = [str(Path.home() / ".local" / "bin" / "openclaw"), *argv[1:]]
    try:
        p = runner(argv, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:
        raise OpenClawError(f"openclaw introuvable ({e})") from e
    except subprocess.TimeoutExpired as e:
        raise OpenClawError("openclaw ne répond pas (délai dépassé) ; vérifiez /cron list avant de recommencer") from e
    if p.returncode != 0:
        raise OpenClawError((p.stderr or p.stdout or "échec openclaw").strip())
    return p.stdout


def list_jobs(runner) -> list[dict]:
    data = json.loads(_run(["openclaw", "cron", "list", "--json"], runner) or "{}")
    return [j for j in data.get("jobs", []) if str(j.get("declarationKey") or "").startswith(DECL_PREFIX)]


def add_job(argv: list[str], job_key: str, runner) -> str:
    _run(argv, runner)
    for j in list_jobs(runner):
        if j.get("declarationKey") == job_key:
            return j["id"]
    raise OpenClawError(f"tâche créée mais introuvable ({job_key})")


def resolve_id(prefix: str, jobs: list[dict]) -> str:
    matches = [j["id"] for j in jobs if j["id"].startswith(prefix)]
    if not matches:
        raise SyntaxeInvalide(f"tâche introuvable : {prefix}")
    if len(matches) > 1:
        raise SyntaxeInvalide(f"identifiant ambigu : {prefix} ({len(matches)} tâches)")
    return matches[0]


def _ligne(j: dict) -> str:
    etat = "actif" if j.get("enabled") else "en pause"
    return f"{j['id'][:8]}  {etat}  {j.get('displayName', '')}  (dernier : {j.get('lastRunStatus') or '—'})"


def manage(g: Gestion, runner) -> str:
    jobs = list_jobs(runner)
    if g.action == "list":
        return "\n".join(_ligne(j) for j in jobs) if jobs else "Aucune tâche planifiée."
    full = resolve_id(g.id or "", jobs)
    verbe = {"rm": "rm", "pause": "disable", "reprendre": "enable"}[g.action]
    _run(["openclaw", "cron", verbe, full], runner)
    return {"rm": "Tâche {} supprimée.", "pause": "Tâche {} en pause.",
            "reprendre": "Tâche {} reprise."}[g.action].format(full[:8])
