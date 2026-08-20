"""Tests de l'endpoint /capture de server.py."""

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
