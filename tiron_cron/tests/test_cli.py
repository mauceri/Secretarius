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


def test_date_passee_refusee_en_syntaxe(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "add 2020-01-01T09:00 rappel x"])
    assert out["ok"] is False and "passé" in out["text"]


def test_date_passee_du_modele_refusee(capsys):
    http = lambda url, body, timeout: {"message": {"content": "add 2026-10-02T08:00 rappel appeler"}}
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "ce matin vers 8 heures x"], http=http)
    assert out["ok"] is False and "passé" in out["text"]


def test_nl_regles_revalidees_duree_nulle(capsys):
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "dans 0 minutes rappelle-moi de x"])
    assert out["ok"] is False


def test_openclaw_introuvable_reste_du_json(capsys):
    def run(argv, **kw):
        raise FileNotFoundError("openclaw")
    out = _call(capsys, ["add", "--origin", "sms", "--args", "add 20m rappel x"], runner=run)
    assert out["ok"] is False and out["text"].startswith("Échec de la création")


def test_delai_openclaw_depasse_reste_du_json(capsys):
    import subprocess

    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 60)
    out = _call(capsys, ["manage", "--args", "list"], runner=run)
    assert out["ok"] is False and out["text"].startswith("Échec")


def test_nl_consigne_du_modele_ramenee_a_un_rappel(capsys):
    http = lambda url, body, timeout: {"message": {"content": "add 1d consigne résume le wiki"}}
    out = _call(capsys, ["nl", "--origin", "telegram", "--text", "lundi prochain vers 8 heures x"], http=http)
    assert out["ok"] and out["args"] == "add 1d rappel résume le wiki"


def test_livrer_lit_l_evenement_sur_stdin(capsys, monkeypatch, tmp_path):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"jobId": "j", "status": "ok", "summary": "x"}'))
    env = dict(ENV, TIRON_RECEPTION_DIR=str(tmp_path))
    out = _call(capsys, ["livrer"], env=env, notifier=lambda t: None)
    assert out["ok"] and out["text"].endswith("j.md")


def test_consigne_va_dans_obsidian_par_defaut_meme_depuis_telegram(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "add 20m consigne fais le point"])
    assert out["ok"] and ", obsidian, consigne" in out["text"]


def test_consigne_telegram_explicite_garde_telegram(capsys):
    out = _call(capsys, ["preview", "--origin", "obsidian", "--args", "add 20m telegram consigne x"])
    assert ", telegram, consigne" in out["text"]


def test_rappel_garde_le_canal_d_origine(capsys):
    out = _call(capsys, ["preview", "--origin", "telegram", "--args", "add 20m rappel x"])
    assert ", telegram, rappel" in out["text"]


def test_livrer_signale_un_avis_non_envoye(capsys, monkeypatch, tmp_path):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"jobId": "j", "status": "ok", "summary": "x"}'))
    env = dict(ENV, TIRON_RECEPTION_DIR=str(tmp_path))

    def panne(_):
        raise OSError("openclaw : Node.js trop ancien")
    out = _call(capsys, ["livrer"], env=env, notifier=panne)
    assert out["ok"] and (tmp_path / "2026-10-02 1000 j.md").exists()
    assert "avis Telegram non envoyé : openclaw : Node.js trop ancien" in out["text"]
