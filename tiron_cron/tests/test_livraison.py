# tiron_cron/tests/test_livraison.py
from datetime import datetime

from tiron_cron.livraison import livrer

NOW = datetime(2026, 10, 9, 8, 0)


def _env(tmp_path):
    return {"TIRON_RECEPTION_DIR": str(tmp_path / "Réception")}


def test_resultat_ok_ecrit_une_note_et_previent(tmp_path):
    avis = []
    evt = {"jobId": "abc", "status": "ok", "summary": "3 pages à relire.",
           "job": {"displayName": "consigne : fais le point sur le wiki"}}
    p = livrer(evt, NOW, _env(tmp_path), avis.append)
    assert p == tmp_path / "Réception" / "2026-10-09 0800 fais le point sur le wiki.md"
    assert p.read_text(encoding="utf-8") == (
        "---\ntags: [livraison]\norigine: cron\ntache: fais le point sur le wiki\n"
        "date: 2026-10-09T08:00\nstatut: ok\n---\n\n3 pages à relire.\n")
    assert avis == ["Nouvelle livraison dans Obsidian : fais le point sur le wiki (Réception)"]


def test_echec_ecrit_une_note_d_echec(tmp_path):
    avis = []
    evt = {"jobId": "abc", "status": "error", "error": "model timeout",
           "job": {"displayName": "commande : /wikistatus"}}
    p = livrer(evt, NOW, _env(tmp_path), avis.append)
    assert "statut: erreur" in p.read_text(encoding="utf-8")
    assert "Échec : model timeout" in p.read_text(encoding="utf-8")
    assert avis == ["Échec de la tâche « wikistatus » : détail dans Obsidian (Réception)"]


def test_saute_ou_vide_ne_livre_rien(tmp_path):
    avis = []
    assert livrer({"jobId": "a", "status": "skipped"}, NOW, _env(tmp_path), avis.append) is None
    assert livrer({"jobId": "a", "status": "ok", "summary": "  "}, NOW, _env(tmp_path), avis.append) is None
    assert livrer({"jobId": "a", "status": "ok", "summary": "NO_REPLY"}, NOW, _env(tmp_path), avis.append) is None
    assert avis == [] and not (tmp_path / "Réception").exists()


def test_titre_de_repli_et_caracteres_interdits(tmp_path):
    p = livrer({"jobId": "j1", "status": "ok", "summary": "x", "job": {"name": "a/b:c"}},
               NOW, _env(tmp_path), lambda t: None)
    assert p.name == "2026-10-09 0800 abc.md"
    p2 = livrer({"jobId": "j1", "status": "ok", "summary": "x"}, NOW, _env(tmp_path), lambda t: None)
    assert p2.name == "2026-10-09 0800 j1.md"


def test_collision_de_nom(tmp_path):
    evt = {"jobId": "j", "status": "ok", "summary": "x", "job": {"displayName": "consigne : t"}}
    p1 = livrer(evt, NOW, _env(tmp_path), lambda t: None)
    p2 = livrer(evt, NOW, _env(tmp_path), lambda t: None)
    assert p2.name == "2026-10-09 0800 t 2.md" and p1 != p2


def test_avis_en_echec_n_empeche_pas_la_note(tmp_path):
    def panne(_):
        raise OSError("openclaw introuvable")
    p = livrer({"jobId": "j", "status": "ok", "summary": "x"}, NOW, _env(tmp_path), panne)
    assert p.exists()
