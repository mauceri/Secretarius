"""Tests des endpoints /capture et /query de server.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def raw_path(tmp_path, monkeypatch):
    target = tmp_path / "raw"
    monkeypatch.setenv("WIKI_RAW_PATH", str(target))
    return target


class TestHandleCapture:
    def test_missing_text_returns_400(self, client, raw_path):
        response = client.post("/capture", json={"tags": ["ia"]})
        assert response.status_code == 400

    def test_writes_file_with_text_and_tags(self, client, raw_path):
        response = client.post("/capture", json={
            "text": "Note d'origine : Ma note (dossier/ma-note.md)\n\nContenu de test.",
            "tags": ["documentation"],
        })
        assert response.status_code == 200
        data = response.get_json()
        created = raw_path / data["filename"]
        assert created.exists()
        content = created.read_text(encoding="utf-8")
        assert "Contenu de test." in content
        assert "documentation" in content

    def test_returns_created_filename(self, client, raw_path):
        response = client.post("/capture", json={"text": "Contenu minimal."})
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["filename"].endswith(".md")

    def test_slug_uses_title_over_boilerplate_prefixed_text(self, client, raw_path):
        response = client.post("/capture", json={
            "text": "Note d'origine : Douleur (Ecriture/Exode/Douleur.md)\n\nLizzie Klein avait...",
            "title": "Douleur",
        })
        data = response.get_json()
        assert "douleur" in data["filename"]
        assert "lizzie" not in data["filename"]

    def test_forwards_vault_name(self, client, monkeypatch):
        import server
        calls = []

        def fake_capture_comment(text, raw, tags=None, title=None, vault_name=None):
            calls.append(vault_name)
            class _Path:
                name = "fake.md"
            return _Path()

        monkeypatch.setattr(server, "capture_comment", fake_capture_comment)
        client.post("/capture", json={"text": "note", "vault_name": "Arbath"})
        client.post("/capture", json={"text": "note"})

        assert calls == ["Arbath", None]


class TestHandleQuery:
    def test_returns_history_slug_and_brief(self, client, monkeypatch):
        class _Result:
            text = "Synthèse."
            references = ["c-test"]
            saved_slug = ""
            history_slug = "20260908-120000-question-test"
            brief = "Résumé bref."

        class _Q:
            mode = "hybrid"

            def query(self, question, top_k=5, save=False, vault_name=None):
                return _Result()

        import server
        monkeypatch.setattr(server, "_wq", _Q())

        response = client.post("/query", json={"question": "Question ?"})

        assert response.status_code == 200
        data = response.get_json()
        assert data["text"] == "Synthèse."
        assert data["references"] == ["c-test"]
        assert data["history_slug"] == "20260908-120000-question-test"
        assert data["brief"] == "Résumé bref."

    def test_forwards_vault_name_to_query(self, client, monkeypatch):
        class _Result:
            text = "Synthèse."
            references = []
            saved_slug = ""
            history_slug = "slug"
            brief = "Bref."

        calls = []

        class _Q:
            mode = "hybrid"

            def query(self, question, top_k=5, save=False, vault_name=None):
                calls.append(vault_name)
                return _Result()

        import server
        monkeypatch.setattr(server, "_wq", _Q())

        client.post("/query", json={"question": "Question ?", "vault_name": "Arbath"})
        client.post("/query", json={"question": "Question ?"})

        assert calls == ["Arbath", None]


class TestHandleRun:
    _COMMAND_TABLE = [
        ("/ingest", "op_ingest", ""),
        ("/wikistatus", "op_status", ""),
        ("/r", "op_search", "une recherche"),
        ("/tags", "op_tags", ""),
        ("/kbupdate", "op_kb_update", ""),
        ("/relire", "op_review", ""),
        ("/verifie", "op_verify", "src-test"),
        ("/supprimer?", "op_delete_preview", "src-test"),
        ("/supprimer!", "op_delete", "src-test"),
    ]

    @pytest.mark.parametrize("command,op_name,arg", _COMMAND_TABLE)
    def test_dispatches_each_command_to_its_op(self, client, monkeypatch, command, op_name, arg):
        import server
        calls = []

        def fake(*args):
            calls.append(args)
            return {"status": "ok"}

        monkeypatch.setattr(server, op_name, fake)
        response = client.post("/run", json={"command": command, "arg": arg})

        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
        assert calls == ([(arg,)] if arg else [()])

    def test_dispatches_c_with_vault_name(self, client, monkeypatch):
        import server
        calls = []
        monkeypatch.setattr(server, "op_capture", lambda *a: calls.append(a) or {"status": "ok"})
        client.post("/run", json={"command": "/c", "arg": "texte", "vault_name": "Arbath"})
        assert calls == [("texte", "Arbath")]

    def test_dispatches_c_without_vault_name(self, client, monkeypatch):
        import server
        calls = []
        monkeypatch.setattr(server, "op_capture", lambda *a: calls.append(a) or {"status": "ok"})
        client.post("/run", json={"command": "/c", "arg": "texte"})
        assert calls == [("texte", None)]

    def test_dispatches_q_with_vault_name(self, client, monkeypatch):
        import server
        calls = []
        monkeypatch.setattr(server, "op_query", lambda *a: calls.append(a) or {"status": "ok"})
        client.post("/run", json={"command": "/q", "arg": "question ?", "vault_name": "Arbath"})
        assert calls == [("question ?", "Arbath")]

    def test_unknown_command_returns_400(self, client):
        response = client.post("/run", json={"command": "/inconnue", "arg": ""})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_supprimer_is_refused_and_never_dispatched(self, client):
        response = client.post("/run", json={"command": "/supprimer", "arg": "src-test"})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_op_exception_returns_500(self, client, monkeypatch):
        import server

        def boom(*a):
            raise RuntimeError("panne simulée")

        monkeypatch.setattr(server, "op_capture", boom)
        response = client.post("/run", json={"command": "/c", "arg": "texte"})

        assert response.status_code == 500
        assert "panne simulée" in response.get_json()["error"]
