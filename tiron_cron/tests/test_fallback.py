# tiron_cron/tests/test_fallback.py
from datetime import datetime
from zoneinfo import ZoneInfo

from tiron_cron.fallback import normalize_with_model
from tiron_cron.syntax import Job

NOW = datetime(2026, 10, 2, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))


def _http(content):
    calls = []

    def http(url, body, timeout):
        calls.append((url, body))
        if isinstance(content, Exception):
            raise content
        return {"message": {"content": content}}
    http.calls = calls
    return http


def test_ligne_valide():
    http = _http('add "0 8 * * 1" rappel payer le loyer')
    assert normalize_with_model("x", NOW, {}, http) == Job("cron", "0 8 * * 1", None, "rappel", "payer le loyer")
    url, body = http.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "qwen3:8b" and body["think"] is False and body["stream"] is False
    assert "2026-10-02T10:00" in body["messages"][0]["content"]


def test_prefixe_cron_et_bloc_de_code_toleres():
    assert normalize_with_model("x", NOW, {}, _http("```\n/cron add 20m rappel x\n```")) is not None


def test_env_surcharge_url_et_modele():
    http = _http("add 20m rappel x")
    normalize_with_model("x", NOW, {"OLLAMA_URL": "http://h:1", "TIRON_CRON_MODEL": "m"}, http)
    assert http.calls[0][0] == "http://h:1/api/chat" and http.calls[0][1]["model"] == "m"


def test_aucune_syntaxe_invalide_gestion_erreur_reseau():
    assert normalize_with_model("x", NOW, {}, _http("AUCUNE")) is None
    assert normalize_with_model("x", NOW, {}, _http("add demain rappel x")) is None
    assert normalize_with_model("x", NOW, {}, _http("rm 1234")) is None
    assert normalize_with_model("x", NOW, {}, _http(OSError("refused"))) is None


def test_prompt_contient_des_exemples_ponctuel_et_recurrent():
    http = _http("AUCUNE")
    normalize_with_model("x", NOW, {}, http)
    system = http.calls[0][1]["messages"][0]["content"]
    assert "lundi prochain" in system and "chaque lundi" in system
    assert "add 2026-10-05T08:00 rappel sortir les poubelles" in system
