import json, os, subprocess, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "switch-brain.sh"


def _setup(tmp, key_file_content=None):
    brains = tmp / "brains.env"
    lines = [
        "BRAIN_SANROQUE_URL=http://100.100.126.7:8998",
        "BRAIN_SANROQUE_KEY=",
        "BRAIN_MODAL_URL=https://ex--tiron.modal.run",
        f"BRAIN_MODAL_KEY_FILE={tmp}/key",
        f"WIKI_PATH={tmp}/wiki",
    ]
    brains.write_text("\n".join(lines) + "\n")
    if key_file_content is not None:
        (tmp / "key").write_text(key_file_content)
    oj = tmp / "openclaw.json"
    oj.write_text(json.dumps({"models": {"providers": {"tiron-llm": {
        "baseUrl": "http://127.0.0.1:8998/v1", "apiKey": "local"}}}}))
    renv = tmp / "router.env"
    renv.write_text("TIRON_LLAMA_BASE=http://127.0.0.1:8998\nTIRON_LLAMA_KEY=\nWIKI_PATH=/keep\n")
    return brains, oj, renv


def _run(tmp, name, brains, oj, renv):
    env = {**os.environ, "BRAINS_ENV": str(brains), "OPENCLAW_JSON": str(oj),
           "ROUTER_ENV": str(renv), "SWITCH_BRAIN_NO_RESTART": "1"}
    return subprocess.run(["bash", str(SCRIPT), name], env=env,
                          capture_output=True, text=True)


def test_switch_modal(tmp_path):
    brains, oj, renv = _setup(tmp_path, key_file_content="SECRET123")
    r = _run(tmp_path, "modal", brains, oj, renv)
    assert r.returncode == 0, r.stderr
    prov = json.loads(oj.read_text())["models"]["providers"]["tiron-llm"]
    assert prov["baseUrl"] == "https://ex--tiron.modal.run/v1"
    assert prov["apiKey"] == "SECRET123"
    env = renv.read_text()
    assert "TIRON_LLAMA_BASE=https://ex--tiron.modal.run" in env
    assert "TIRON_LLAMA_KEY=SECRET123" in env
    assert "WIKI_PATH=/keep" in env  # ligne préservée
    assert (oj.parent / "openclaw.json.bak").exists()


def test_switch_sanroque_no_key(tmp_path):
    brains, oj, renv = _setup(tmp_path)
    r = _run(tmp_path, "sanroque", brains, oj, renv)
    assert r.returncode == 0, r.stderr
    prov = json.loads(oj.read_text())["models"]["providers"]["tiron-llm"]
    assert prov["baseUrl"] == "http://100.100.126.7:8998/v1"
    assert prov["apiKey"] == "local"  # clé vide -> "local" (inerte)
    assert "TIRON_LLAMA_BASE=http://100.100.126.7:8998" in renv.read_text()


def test_unknown_brain_touches_nothing(tmp_path):
    brains, oj, renv = _setup(tmp_path)
    before = oj.read_text()
    r = _run(tmp_path, "inconnu", brains, oj, renv)
    assert r.returncode == 1
    assert oj.read_text() == before  # rien modifié


def test_modal_missing_key_fails(tmp_path):
    brains, oj, renv = _setup(tmp_path, key_file_content=None)  # pas de fichier clé
    before = oj.read_text()
    r = _run(tmp_path, "modal", brains, oj, renv)
    assert r.returncode == 1
    assert oj.read_text() == before
