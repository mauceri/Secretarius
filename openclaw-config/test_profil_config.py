"""Élagage d'openclaw.json par profil, sur le vrai gabarit (spec 2026-10-12)."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from profil_config import OUTILS_DE, appliquer_profil  # noqa: E402

GABARIT = Path(__file__).parent / "openclaw.json.template"
LEGER = {"wiki", "scout", "obfuscateur"}


def _config():
    texte = re.sub(r"\$\{[A-Z_]+\}", "0", GABARIT.read_text())
    return json.loads(texte)


def _toutes_listes(c):
    listes = [c["tools"]["sandbox"]["tools"].get(k, []) for k in ("allow", "deny")]
    for a in c["agents"]["entries"].values():
        t = a.get("tools", {})
        listes.append(t.get("deny", []) if isinstance(t.get("deny"), list) else [])
        st = t.get("sandbox", {}).get("tools", {})
        listes += [st.get("allow", []), st.get("deny", [])]
    return [x for liste in listes for x in liste]


def test_leger_sans_agents_absents():
    c = appliquer_profil(_config(), LEGER, "modal")
    assert set(c["agents"]["entries"]) == {"main", "wiki", "scout"}


def test_leger_aucun_outil_de_module_absent():
    c = appliquer_profil(_config(), LEGER, "modal")
    absents = set(OUTILS_DE["courrier"]) | set(OUTILS_DE["gog"])
    assert absents.isdisjoint(_toutes_listes(c))


def test_leger_cerveau_et_proxy_obfusques():
    c = appliquer_profil(_config(), LEGER, "modal")
    assert c["agents"]["entries"]["main"]["model"]["primary"] == "obfusque/qwen3-14b-h128-a1-h02"
    assert c["models"]["providers"]["obfusque"]["baseUrl"] == "http://172.17.0.1:8001/v1"


def test_leger_routeur_desactive():
    c = appliquer_profil(_config(), LEGER, "modal")
    assert c["plugins"]["entries"]["derisk-deleg"]["config"] == {"routeur": False}


def test_leger_wiki_sans_ollama():
    env = appliquer_profil(_config(), LEGER, "modal")["agents"]["entries"]["wiki"]["sandbox"]["docker"]["env"]
    assert env["WIKI_INGEST_LLM_BACKEND"] == "openai"
    assert env["OPENAI_BASE_URL"] == "http://172.17.0.1:8001/v1"
    assert not any(k.startswith("OLLAMA_") for k in env)


def test_leger_sans_montage_de_cle_infomaniak():
    c = appliquer_profil(_config(), LEGER, "modal")
    binds = c["agents"]["entries"]["wiki"]["sandbox"]["docker"]["binds"]
    assert not any("infomaniak-key" in b for b in binds)


def test_lourd_garde_tout():
    tous = {"wiki", "scout", "obfuscateur", "laya", "courrier", "cron", "gog"}
    c = appliquer_profil(_config(), tous, "ollama")
    assert set(c["agents"]["entries"]) == {"main", "wiki", "scout", "consignes", "gog"}
    assert c["plugins"]["entries"]["derisk-deleg"].get("config", {}).get("routeur", True)


def test_entree_non_modifiee():
    c = _config()
    avant = json.dumps(c, sort_keys=True)
    appliquer_profil(c, LEGER, "modal")
    assert json.dumps(c, sort_keys=True) == avant


@pytest.mark.xfail(strict=True, reason="Task 4 : identifiant Telegram en dur dans le gabarit")
def test_aucune_valeur_propre_a_sanroque():
    texte = json.dumps(appliquer_profil(_config(), LEGER, "modal"))
    assert "/home/mauceric" not in texte and "Arbath" not in texte and "220537738" not in texte
