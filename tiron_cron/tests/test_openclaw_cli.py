# tiron_cron/tests/test_openclaw_cli.py
import json
import subprocess

import pytest

from tiron_cron.openclaw_cli import (
    OpenClawError, add_job, build_add_argv, list_jobs, manage, resolve_id)
from tiron_cron.syntax import Gestion, Job, SyntaxeInvalide

ENV = {"TIRON_TELEGRAM_CHAT_ID": "123", "TIRON_SMS_TO": "+33600000000",
       "TIRON_WHATSAPP_TO": "+33600000001", "SECRETARIUS_DIR": "/srv/sec"}


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def test_rappel_cron_sms():
    job = Job("cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")
    argv = build_add_argv(job, "sms", "Europe/Paris", ENV, "tiron:abcd1234")
    assert argv[:3] == ["openclaw", "cron", "add"]
    assert _opt(argv, "--cron") == "0 8 * * 1"
    assert _opt(argv, "--tz") == "Europe/Paris"
    assert _opt(argv, "--declaration-key") == "tiron:abcd1234"
    assert json.loads(_opt(argv, "--command-argv")) == ["printf", "%s\n", "Rappel : payer le loyer"]
    assert "--announce" in argv and _opt(argv, "--channel") == "sms" and _opt(argv, "--to") == "+33600000000"
    assert "--delete-after-run" not in argv


def test_tache_unique_supprimee_apres_succes():
    argv = build_add_argv(Job("at", "20m", None, "rappel", "x"), "telegram", "UTC", ENV, "tiron:1")
    assert _opt(argv, "--at") == "20m" and "--delete-after-run" in argv


def test_consigne_va_a_main_en_session_isolee():
    argv = build_add_argv(Job("cron", "0 7 * * *", None, "consigne", "résume le wiki"),
                          "telegram", "UTC", ENV, "tiron:1")
    assert _opt(argv, "--agent") == "main" and _opt(argv, "--session") == "isolated"
    assert _opt(argv, "--message") == "résume le wiki" and "--command-argv" not in argv


def test_consigne_en_contexte_allege_sans_raisonnement():
    # Essai du 2026-10-07 : ~20 s gagnées avec qwen3:8b sur le processeur.
    argv = build_add_argv(Job("cron", "0 18 * * 5", None, "consigne", "résume le wiki"),
                          "telegram", "UTC", ENV, "tiron:1")
    assert "--light-context" in argv and _opt(argv, "--thinking") == "off"


def test_rappel_sans_options_d_agent():
    argv = build_add_argv(Job("at", "20m", None, "rappel", "x"), "telegram", "UTC", ENV, "tiron:1")
    assert "--light-context" not in argv and "--thinking" not in argv


def test_commande_va_a_main():
    argv = build_add_argv(Job("cron", "0 8 * * 1", None, "commande", "/wikistatus"),
                          "telegram", "UTC", ENV, "tiron:1")
    assert _opt(argv, "--message") == "/wikistatus"


def test_rappel_obsidian_lance_le_script_interne_sans_livraison():
    argv = build_add_argv(Job("at", "1h", "obsidian", "rappel", "payer"), "obsidian", "UTC", ENV, "tiron:1")
    assert json.loads(_opt(argv, "--command-argv")) == [
        "python3", "-m", "tiron_cron", "deliver-obsidian", "--text", "payer"]
    assert _opt(argv, "--command-cwd") == "/srv/sec" and "--no-deliver" in argv
    assert "--announce" not in argv


def test_consigne_et_commande_obsidian_par_webhook():
    for type_, contenu in (("consigne", "résume"), ("commande", "/tags")):
        argv = build_add_argv(Job("at", "1h", "obsidian", type_, contenu), "obsidian", "UTC", ENV, "tiron:1")
        assert _opt(argv, "--webhook") == "http://127.0.0.1:5051/livraison"
        assert _opt(argv, "--agent") == "main" and _opt(argv, "--message") == contenu
        for interdit in ("--announce", "--channel", "--to", "--no-deliver"):
            assert interdit not in argv


def test_url_de_livraison_configurable():
    env = dict(ENV, TIRON_LIVRAISON_URL="http://127.0.0.1:9/x")
    argv = build_add_argv(Job("at", "1h", "obsidian", "consigne", "r"), "obsidian", "UTC", env, "tiron:1")
    assert _opt(argv, "--webhook") == "http://127.0.0.1:9/x"


def test_destinataire_non_configure():
    with pytest.raises(SyntaxeInvalide, match="TIRON_SMS_TO"):
        build_add_argv(Job("at", "1h", "sms", "rappel", "x"), "sms", "UTC", {}, "tiron:1")


def test_canal_inconnu():
    with pytest.raises(SyntaxeInvalide, match="canal"):
        build_add_argv(Job("at", "1h", None, "rappel", "x"), None, "UTC", ENV, "tiron:1")


def _runner(outputs):
    """Faux subprocess.run : renvoie successivement les stdout donnés, enregistre les argv."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        out = outputs.pop(0)
        return subprocess.CompletedProcess(argv, 0 if not isinstance(out, Exception) else 1,
                                           stdout=out if isinstance(out, str) else "",
                                           stderr=str(out) if isinstance(out, Exception) else "")
    run.calls = calls
    return run


JOBS = {"jobs": [
    {"id": "aaaa1111-x", "declarationKey": "tiron:k1", "displayName": "rappel : loyer",
     "enabled": True, "lastRunStatus": "ok"},
    {"id": "aaaa2222-y", "declarationKey": "tiron:k2", "displayName": "consigne : wiki",
     "enabled": False, "lastRunStatus": None},
    {"id": "208f659e-z", "declarationKey": "skill-collection-review:main", "displayName": "système",
     "enabled": True, "lastRunStatus": "ok"},
]}


def test_list_ne_garde_que_tiron():
    jobs = list_jobs(_runner([json.dumps(JOBS)]))
    assert [j["id"] for j in jobs] == ["aaaa1111-x", "aaaa2222-y"]


def test_resolve_id_prefixe_unique_ambigu_systeme():
    jobs = list_jobs(_runner([json.dumps(JOBS)]))
    assert resolve_id("aaaa1", jobs) == "aaaa1111-x"
    with pytest.raises(SyntaxeInvalide, match="ambigu"):
        resolve_id("aaaa", jobs)
    with pytest.raises(SyntaxeInvalide, match="introuvable"):
        resolve_id("208f659e", jobs)  # tâche système : jamais atteignable


def test_add_job_retrouve_l_id_par_declaration_key():
    run = _runner(["{}", json.dumps(JOBS)])
    assert add_job(["openclaw", "cron", "add"], "tiron:k2", run) == "aaaa2222-y"


def test_add_job_erreur_openclaw():
    with pytest.raises(OpenClawError, match="boom"):
        add_job(["openclaw", "cron", "add"], "tiron:k9", _runner([RuntimeError("boom")]))


def test_manage_list_rm_pause_reprendre():
    assert "aaaa1111" in manage(Gestion("list", None), _runner([json.dumps(JOBS)]))
    run = _runner([json.dumps(JOBS), ""])
    assert "supprimée" in manage(Gestion("rm", "aaaa1"), run)
    assert run.calls[-1] == ["openclaw", "cron", "rm", "aaaa1111-x"]
    run = _runner([json.dumps(JOBS), ""])
    manage(Gestion("pause", "aaaa1"), run)
    assert run.calls[-1] == ["openclaw", "cron", "disable", "aaaa1111-x"]
    run = _runner([json.dumps(JOBS), ""])
    manage(Gestion("reprendre", "aaaa2"), run)
    assert run.calls[-1] == ["openclaw", "cron", "enable", "aaaa2222-y"]


def test_manage_list_vide():
    assert manage(Gestion("list", None), _runner([json.dumps({"jobs": []})])) == "Aucune tâche planifiée."
