"""
Couche d'abstraction LLM pour Wiki_LM.

Backends supportés :
  - claude   : Anthropic API (ANTHROPIC_API_KEY)
  - ollama   : serveur Ollama local (OLLAMA_BASE_URL, défaut http://localhost:11434)
  - openai   : API OpenAI-compatible (OPENAI_BASE_URL + OPENAI_API_KEY)

Sélection du backend :
  Variable d'environnement WIKI_LLM_BACKEND  (claude | ollama | openai)
  ou argument backend= au constructeur.

Usage :
    from llm import LLM
    llm = LLM()                         # backend auto-détecté
    llm = LLM(backend="ollama", model="qwen2.5:7b")
    text = llm.complete("Résume ce texte : …")
    text = llm.complete(messages=[{"role": "user", "content": "…"}])
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _ClaudeBackend:
    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=_env("ANTHROPIC_API_KEY"))
        self.model = model or "claude-sonnet-4-6"

    def complete(self, messages: list[dict], system: str = "", max_tokens: int = 2048) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text


class _OllamaBackend:
    def __init__(self, model: str) -> None:
        self.base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or _env("OLLAMA_MODEL", "qwen2.5:7b")

    def complete(self, messages: list[dict], system: str = "", max_tokens: int = 2048) -> str:
        import urllib.request
        import json

        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        msg = data["message"]
        # Certains modèles (ex. qwen3.5) placent la réponse dans "thinking"
        # si le contenu textuel est vide.
        content = msg.get("content", "")
        if not content.strip():
            content = msg.get("thinking", "")
        return content


class _OpenAIBackend:
    def __init__(self, model: str) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self._client = openai.OpenAI(
            api_key=_env("OPENAI_API_KEY", "ollama"),
            base_url=_env("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )
        self.model = model or _env("OPENAI_MODEL", "qwen2.5:7b")

    def complete(self, messages: list[dict], system: str = "", max_tokens: int = 2048) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Façade publique
# ---------------------------------------------------------------------------

_BACKENDS = {
    "claude": _ClaudeBackend,
    "ollama": _OllamaBackend,
    "openai": _OpenAIBackend,
}


class LLM:
    """Façade unifiée pour tous les backends LLM.

    Params
    ------
    backend : str, optional
        "claude", "ollama" ou "openai". Défaut : $WIKI_LLM_BACKEND ou "ollama".
    model : str, optional
        Identifiant du modèle. Défaut : dépend du backend.
    """

    def __init__(self, backend: str = "", model: str = "") -> None:
        backend = backend or _env("WIKI_LLM_BACKEND", "ollama")
        if backend not in _BACKENDS:
            raise ValueError(f"Backend inconnu : {backend!r}. Valeurs possibles : {list(_BACKENDS)}")
        self._backend = _BACKENDS[backend](model)

    def complete(
        self,
        prompt: str = "",
        *,
        messages: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """Appelle le LLM et retourne la réponse texte.

        Params
        ------
        prompt : str
            Raccourci — transformé en message user unique.
        messages : list[dict], optional
            Liste de messages au format {role, content}. Prioritaire sur prompt.
        system : str, optional
            Instruction système (ignorée si le backend ne la supporte pas).
        max_tokens : int
            Limite de tokens en sortie.
        """
        if messages is None:
            if not prompt:
                raise ValueError("Fournir prompt= ou messages=")
            messages = [{"role": "user", "content": prompt}]
        return self._backend.complete(messages, system=system, max_tokens=max_tokens)
