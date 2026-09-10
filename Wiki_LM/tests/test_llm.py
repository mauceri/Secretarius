"""Tests de la façade LLM et de son mécanisme de repli.

Contexte du repli : l'ingestion tourne en régime autarcique sur Qwen3-8B
local (Ollama, CPU) — un texte un peu long peut dépasser le timeout de
_OllamaBackend (300 s) sans jamais lever d'erreur explicite côté modèle.
Le repli bascule alors sur un second backend (en production : le proxy
OpenAI local vers Qwen3-14B obfusqué servi sur Modal)."""

from __future__ import annotations

import pytest

from llm import LLM


class _StubBackend:
    """Backend factice : soit répond, soit lève, au choix du test."""

    def __init__(self, response: str = "", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def complete(self, messages, system="", max_tokens=2048):
        self.calls.append({"messages": messages, "system": system, "max_tokens": max_tokens})
        if self.error:
            raise self.error
        return self.response


def _llm_with_backend(backend) -> LLM:
    """Construit un LLM sans passer par _BACKENDS — injecte directement le stub."""
    obj = LLM.__new__(LLM)
    obj._backend = backend
    obj._fallback = None
    return obj


def test_complete_sans_fallback_retourne_la_reponse_du_backend():
    backend = _StubBackend(response="bonjour")
    llm = _llm_with_backend(backend)
    assert llm.complete("salut") == "bonjour"


def test_complete_sans_fallback_propage_lerreur():
    backend = _StubBackend(error=TimeoutError("trop lent"))
    llm = _llm_with_backend(backend)
    with pytest.raises(TimeoutError):
        llm.complete("salut")


def test_complete_bascule_sur_le_fallback_apres_echec():
    primaire = _StubBackend(error=TimeoutError("trop lent"))
    repli = _StubBackend(response="réponse du repli")
    llm = _llm_with_backend(primaire)
    llm._fallback = _llm_with_backend(repli)

    assert llm.complete("salut") == "réponse du repli"
    assert primaire.calls  # le primaire a bien été tenté d'abord
    assert repli.calls


def test_complete_ne_touche_pas_au_fallback_si_le_primaire_reussit():
    primaire = _StubBackend(response="ok")
    repli = _StubBackend(response="jamais utilisé")
    llm = _llm_with_backend(primaire)
    llm._fallback = _llm_with_backend(repli)

    assert llm.complete("salut") == "ok"
    assert not repli.calls


def test_complete_propage_lerreur_du_fallback_si_les_deux_echouent():
    primaire = _StubBackend(error=RuntimeError("primaire KO"))
    repli = _StubBackend(error=RuntimeError("repli KO aussi"))
    llm = _llm_with_backend(primaire)
    llm._fallback = _llm_with_backend(repli)

    with pytest.raises(RuntimeError, match="repli KO aussi"):
        llm.complete("salut")


def test_llm_init_accepte_un_fallback():
    """LLM(backend=..., fallback=...) construit bien avec un second LLM attaché."""
    primaire = LLM(backend="ollama")
    repli = LLM(backend="ollama")
    llm = LLM(backend="ollama", fallback=repli)
    assert llm._fallback is repli


def test_openai_backend_base_url_explicite(monkeypatch):
    """Le backend openai accepte un base_url explicite, distinct de $OPENAI_BASE_URL
    — nécessaire pour pointer le repli vers le proxy local sans perturber le
    backend openai "normal" (DeepSeek) utilisé ailleurs."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "clé-normale")
    llm = LLM(backend="openai", model="qwen3-14b-h128-a1-h02",
               base_url="http://127.0.0.1:8001/v1", api_key="local")
    assert llm._backend.model == "qwen3-14b-h128-a1-h02"
    assert str(llm._backend._client.base_url).rstrip("/") == "http://127.0.0.1:8001/v1"
