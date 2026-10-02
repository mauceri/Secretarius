# tiron_cron/tests/test_syntax.py
import pytest

from tiron_cron.syntax import Gestion, Job, SyntaxeInvalide, parse_command, to_args


def test_add_cron_avec_canal():
    assert parse_command('add "0 8 * * 1" sms rappel payer le loyer') == Job(
        "cron", "0 8 * * 1", "sms", "rappel", "payer le loyer")


def test_add_iso_sans_canal():
    assert parse_command("add 2026-10-02T09:00 rappel appeler le garagiste") == Job(
        "at", "2026-10-02T09:00", None, "rappel", "appeler le garagiste")


def test_add_duree():
    assert parse_command("add 20m telegram rappel sortir le gâteau").quand == "20m"


def test_contenu_avec_apostrophes_et_guillemets_conserve_tel_quel():
    job = parse_command("add 1h rappel d'appeler l'école \"avant midi\"")
    assert job.contenu == "d'appeler l'école \"avant midi\""


def test_commande_liste_blanche():
    assert parse_command('add "0 8 * * 1" commande /wikistatus').contenu == "/wikistatus"


def test_commande_hors_liste_blanche_refusee():
    with pytest.raises(SyntaxeInvalide, match="non planifiable"):
        parse_command('add "0 8 * * 1" commande /supprimer src-x')


def test_consigne():
    assert parse_command("add 2h consigne résume le wiki").type == "consigne"


@pytest.mark.parametrize("args", [
    'add "0 8 * *" rappel x',          # 4 champs
    "add 2026-13-02T09:00 rappel x",   # mois 13
    "add 0m rappel x",                 # durée nulle
    "add demain rappel x",             # forme libre
])
def test_quand_invalide(args):
    with pytest.raises(SyntaxeInvalide, match="quand"):
        parse_command(args)


def test_type_inconnu():
    with pytest.raises(SyntaxeInvalide, match="type inconnu"):
        parse_command("add 20m alerte x")


def test_contenu_vide():
    with pytest.raises(SyntaxeInvalide, match="contenu vide"):
        parse_command("add 20m rappel   ")


def test_guillemet_non_ferme():
    with pytest.raises(SyntaxeInvalide, match="guillemet"):
        parse_command('add "0 8 * * 1 rappel x')


def test_gestion():
    assert parse_command("list") == Gestion("list", None)
    assert parse_command("rm 1a2b3c4d") == Gestion("rm", "1a2b3c4d")
    assert parse_command("PAUSE 1a2b") == Gestion("pause", "1a2b")
    assert parse_command("reprendre 1a2b") == Gestion("reprendre", "1a2b")


def test_gestion_sans_id():
    with pytest.raises(SyntaxeInvalide, match="usage"):
        parse_command("rm")


def test_action_inconnue():
    with pytest.raises(SyntaxeInvalide, match="action inconnue"):
        parse_command("modifier 1a2b")


def test_to_args_aller_retour():
    for args in ['add "0 8 * * 1" sms rappel payer le loyer',
                 "add 2026-10-02T09:00 consigne résume le wiki",
                 "add 20m rappel d'appeler l'école"]:
        assert to_args(parse_command(args)) == args
