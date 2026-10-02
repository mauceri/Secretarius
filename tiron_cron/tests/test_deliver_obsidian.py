# tiron_cron/tests/test_deliver_obsidian.py
from datetime import datetime

from tiron_cron.deliver_obsidian import deliver

NOW = datetime(2026, 10, 6, 8, 0)


def test_ecrit_la_note(tmp_path):
    p = deliver("payer le loyer", NOW, {"TIRON_RAPPELS_DIR": str(tmp_path / "Rappels")})
    assert p == tmp_path / "Rappels" / "2026-10-06 0800 payer le loyer.md"
    assert p.read_text(encoding="utf-8") == \
        "---\ntags: [rappel]\ndate: 2026-10-06T08:00\n---\n\npayer le loyer\n"


def test_titre_sans_caracteres_interdits(tmp_path):
    p = deliver('a/b:c*?"<>|#^[x]', NOW, {"TIRON_RAPPELS_DIR": str(tmp_path)})
    assert p.name == "2026-10-06 0800 abcx.md"


def test_titre_vide_et_collision(tmp_path):
    env = {"TIRON_RAPPELS_DIR": str(tmp_path)}
    p1 = deliver("///", NOW, env)
    p2 = deliver("///", NOW, env)
    assert p1.name == "2026-10-06 0800 rappel.md" and p2.name == "2026-10-06 0800 rappel 2.md"
