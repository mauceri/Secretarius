"""Confinement des fichiers du coffre pour la porte Grok."""
import pytest

from porte import coffre
from porte.coffre import Refus


@pytest.fixture
def racine(tmp_path):
    r = tmp_path / "Publication"
    (r / "Brouillons").mkdir(parents=True)
    (r / "Brouillons" / "veille-a.md").write_text("A")
    (r / "Notes.md").write_text("perso")
    (r / "Wiki_LM" / "raw").mkdir(parents=True)
    (r / "Wiki_LM" / "wiki.md").write_text("w")
    (r / ".obsidian").mkdir()
    (r / ".obsidian" / "app.md").write_text("cfg")
    (r / "image.png").write_bytes(b"x")
    return r


@pytest.mark.parametrize("chemin", [
    "/etc/passwd", "../dehors.md", "Brouillons/../../dehors.md", "",
    "Wiki_LM/wiki.md", "Wiki_LM/raw/veille-x.md", ".obsidian/app.md", "Brouillons/.cache/veille-x.md",
])
def test_resoudre_refuse(racine, chemin):
    with pytest.raises(Refus):
        coffre.resoudre(racine, chemin)


def test_lien_symbolique_sortant_refuse(racine, tmp_path):
    (tmp_path / "secret.md").write_text("secret")
    (racine / "lien.md").symlink_to(tmp_path / "secret.md")
    (tmp_path / "ailleurs").mkdir()
    (racine / "dossier-lien").symlink_to(tmp_path / "ailleurs", target_is_directory=True)
    with pytest.raises(Refus):
        coffre.lire(racine, "lien.md")
    with pytest.raises(Refus):
        coffre.ecrire(racine, "veille", "dossier-lien/veille-x.md", "x")
    assert "lien.md" not in coffre.lister(racine)
    assert not (tmp_path / "ailleurs" / "veille-x.md").exists()


def test_lister_hors_zones_interdites(racine):
    assert coffre.lister(racine) == ["Brouillons/veille-a.md", "Notes.md"]
    assert coffre.lister(racine, "Brouillons") == ["Brouillons/veille-a.md"]


def test_lister_dossier_inexistant_refuse(racine):
    with pytest.raises(Refus):
        coffre.lister(racine, "Absent")


def test_lire(racine):
    assert coffre.lire(racine, "Notes.md") == "perso"
    with pytest.raises(Refus):
        coffre.lire(racine, "image.png")
    with pytest.raises(Refus):
        coffre.lire(racine, "Absent.md")


def test_ecrire_prefixe_impose(racine):
    with pytest.raises(Refus, match="veille-"):
        coffre.ecrire(racine, "veille", "Notes.md", "écrasé")
    assert (racine / "Notes.md").read_text() == "perso"
    assert coffre.ecrire(racine, "veille", "Brouillons/Veille/veille-2026-10-13.md", "v1") \
        == "Brouillons/Veille/veille-2026-10-13.md"
    coffre.ecrire(racine, "veille", "Brouillons/Veille/veille-2026-10-13.md", "v2")
    assert (racine / "Brouillons/Veille/veille-2026-10-13.md").read_text() == "v2"


def test_ecrire_extension_et_taille(racine):
    with pytest.raises(Refus):
        coffre.ecrire(racine, "veille", "veille-x.txt", "x")
    with pytest.raises(Refus):
        coffre.ecrire(racine, "veille", "veille-x.md", "é" * (coffre.TAILLE_MAX // 2 + 1))
