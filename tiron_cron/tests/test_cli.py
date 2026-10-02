# tiron_cron/tests/test_cli.py
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from tiron_cron.__main__ import main
from tiron_cron.tests.test_openclaw_cli import ENV, JOBS, _runner

NOW = datetime(2026, 10, 2, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))


def _call(capsys, argv, **kw):
    kw.setdefault("env", ENV)
    kw.setdefault("now", NOW)
    assert main(argv, **kw) == 0
    return json.loads(capsys.readouterr().out)


def test_preview(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", 'add "0 8 * * 1" rappel payer le loyer'])
    assert out == {"ok": True, "args": 'add "0 8 * * 1" rappel payer le loyer',
                   "text": "Chaque lundi à 08:00 (Europe/Paris), telegram, rappel : payer le loyer"}


def test_preview_origine_inconnue_sans_canal(capsys):
    out = _call(capsys, ["preview", "--origin", "webchat", "--args", "add 20m rappel x"])
    assert out["ok"] is False and "canal" in out["text"]


def test_preview_refuse_la_gestion(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "list"])
    assert out["ok"] is False


def test_add(capsys):
    run = _runner(["{}", json.dumps({"jobs": [{"id": "cccc3333-q", "declarationKey": "tiron:FIXE"}]})])
    out = _call(capsys, ["add", "--origin", "sms", "--args", "add 20m rappel x"],
                runner=run, job_key="tiron:FIXE")
    assert out["ok"] and out["id"] == "cccc3333-q" and out["text"].startswith("Tâche créée (cccc3333)")
    assert run.calls[0][:3] == ["openclaw", "cron", "add"]


def test_add_erreur_openclaw(capsys):
    out = _call(capsys, ["add", "--origin", "sms", "--args", "add 20m rappel x"],
                runner=_runner([RuntimeError("gateway arrêté")]))
    assert out == {"ok": False, "text": "Échec de la création : gateway arrêté"}


def test_nl_regles(capsys):
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "dans 20 minutes rappelle-moi de x"])
    assert out["ok"] and out["source"] == "regles" and out["args"] == "add 20m rappel x"


def test_nl_repli_modele(capsys):
    http = lambda url, body, timeout: {"message": {"content": "add 1d rappel x"}}
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "lundi prochain vers 8 heures x"], http=http)
    assert out["ok"] and out["source"] == "modele" and out["args"] == "add 1d rappel x"


def test_nl_non_compris(capsys):
    http = lambda url, body, timeout: {"message": {"content": "AUCUNE"}}
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "bonjour"], http=http)
    assert out["ok"] is False and "/cron add" in out["text"]


def test_manage(capsys):
    out = _call(capsys, ["manage", "--args", "list"], runner=_runner([json.dumps(JOBS)]))
    assert out["ok"] and "aaaa1111" in out["text"]


def test_manage_refuse_add(capsys):
    assert _call(capsys, ["manage", "--args", "add 20m rappel x"])["ok"] is False


def test_deliver_obsidian(capsys, tmp_path):
    out = _call(capsys, ["deliver-obsidian", "--text", "payer"], env={"TIRON_RAPPELS_DIR": str(tmp_path)})
    assert out["ok"] and out["text"].endswith("payer.md")
