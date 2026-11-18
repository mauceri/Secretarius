"""Ajout du canal Session à une configuration openclaw.json (phase 1, spec 2026-11-18)."""
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from canal_session import ajouter_canal_session  # noqa: E402

GABARIT = Path(__file__).parent / "openclaw.json.template"
ID = "05" + "a" * 64


def _config():
    return json.loads(re.sub(r"\$\{[A-Z_]+\}", "0", GABARIT.read_text()))


def test_canal_session_lie_a_main():
    avant = _config()
    c = ajouter_canal_session(copy.deepcopy(avant), ID, 8091)
    assert {"agentId": "main", "match": {"channel": "session", "accountId": "default"}} in c["bindings"]
    assert c["channels"]["session"] == {"enabled": True, "brokerUrl": "http://127.0.0.1:8091",
                                        "dmPolicy": "allowlist", "allowFrom": [ID]}
    assert f"session:{ID}" in c["commands"]["ownerAllowFrom"]
    assert c["plugins"]["entries"]["session"] == {"enabled": True}
    # Telegram inchangé
    assert c["channels"]["telegram"] == avant["channels"]["telegram"]
    assert [b for b in c["bindings"] if b["match"]["channel"] == "telegram"] == \
        [b for b in avant["bindings"] if b["match"]["channel"] == "telegram"]


def test_idempotent():
    c1 = ajouter_canal_session(_config(), ID, 8091)
    c2 = ajouter_canal_session(copy.deepcopy(c1), ID, 8091)
    assert c1 == c2


def test_refuse_identifiant_invalide():
    import pytest
    with pytest.raises(ValueError):
        ajouter_canal_session(_config(), "05abc", 8091)


def test_installateur_n_exporte_pas_la_phrase_secrete():
    # set -a exporte tout secrets.env aux sous-processus (docker build, openclaw…) :
    # la phrase secrète de Session doit être retirée aussitôt après le chargement.
    texte = (Path(__file__).parent.parent / "install.sh").read_text()
    i = texte.index('set -a; source "$ENV_FILE"; set +a')
    assert "unset SESSION_BOT_MNEMONIC" in texte[i:i + 200]
