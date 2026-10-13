"""Jetons des agents de la porte Grok."""
import stat

import pytest

from porte import jetons


def test_ajouter_stocke_empreinte_seulement(tmp_path):
    f = tmp_path / "porte" / "agents"
    jeton = jetons.ajouter(f, "veille")
    contenu = f.read_text()
    assert jeton not in contenu
    assert contenu == f"veille:{jetons.empreinte(jeton)}\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE(f.parent.stat().st_mode) == 0o700


def test_agent_pour_jeton_valide(tmp_path):
    f = tmp_path / "agents"
    jeton = jetons.ajouter(f, "veille")
    jetons.ajouter(f, "redaction")
    assert jetons.agent_pour(f, f"Bearer {jeton}") == "veille"


@pytest.mark.parametrize("en_tete", [None, "", "Bearer", "Bearer ", "bearer x", "Basic abc", "Bearer faux"])
def test_agent_pour_refuse(tmp_path, en_tete):
    f = tmp_path / "agents"
    jetons.ajouter(f, "veille")
    assert jetons.agent_pour(f, en_tete) is None


def test_fichier_absent_refuse_tout(tmp_path):
    assert jetons.agent_pour(tmp_path / "absent", "Bearer x") is None


def test_retrait_effectif_immediatement(tmp_path):
    f = tmp_path / "agents"
    jeton = jetons.ajouter(f, "veille")
    assert jetons.retirer(f, "veille") is True
    assert jetons.agent_pour(f, f"Bearer {jeton}") is None
    assert jetons.retirer(f, "veille") is False


def test_doublon_et_prefixe_invalide_refuses(tmp_path):
    f = tmp_path / "agents"
    jetons.ajouter(f, "veille")
    with pytest.raises(ValueError):
        jetons.ajouter(f, "veille")
    for mauvais in ["Veille", "1veille", "v", "veille_x", "a" * 40]:
        with pytest.raises(ValueError):
            jetons.ajouter(f, mauvais)


def test_lister(tmp_path):
    f = tmp_path / "agents"
    jetons.ajouter(f, "veille")
    jetons.ajouter(f, "redaction")
    assert jetons.lister(f) == ["veille", "redaction"]


def test_cli_ajouter_affiche_le_jeton_une_fois(tmp_path, capsys):
    f = tmp_path / "agents"
    assert jetons.main(["--fichier", str(f), "ajouter", "veille"]) == 0
    jeton = capsys.readouterr().out.strip().splitlines()[-1]
    assert jetons.agent_pour(f, f"Bearer {jeton}") == "veille"
    assert jetons.main(["--fichier", str(f), "ajouter", "veille"]) == 1
    assert jetons.main(["--fichier", str(f), "lister"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "veille"
    assert jetons.main(["--fichier", str(f), "retirer", "veille"]) == 0
    assert jetons.main(["--fichier", str(f), "retirer", "veille"]) == 1
