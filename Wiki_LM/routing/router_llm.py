"""Routeur par LLM local (llama.cpp Phi-4-mini sur :8998)."""
from __future__ import annotations

import json
import re
import urllib.request

from router_base import Router, RouteResult

_ENDPOINT = "http://127.0.0.1:8998/v1/chat/completions"
_MODEL = "phi-4-mini-instruct"


def _default_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def build_prompt(agents: list[dict]) -> str:
    lines = [
        "Tu es un routeur. Choisis l'agent le plus adapté à la demande de l'utilisateur.",
        "Agents disponibles :",
    ]
    for a in agents:
        lines.append(f'- {a["name"]} : {a["description"]}')
    lines.append('Réponds UNIQUEMENT par un objet JSON {"agent": "<nom>"} sans aucun autre texte.')
    return "\n".join(lines)


def _parse_agent(content: str) -> str | None:
    m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    agent = obj.get("agent")
    return agent if isinstance(agent, str) else None


class LlmRouter(Router):
    def __init__(self, agents: list[dict], endpoint: str = _ENDPOINT,
                 model: str = _MODEL, post_fn=_default_post):
        self.system_prompt = build_prompt(agents)
        self.valid = {a["name"] for a in agents}
        self.endpoint = endpoint
        self.model = model
        self.post_fn = post_fn

    def route(self, message: str) -> RouteResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
        }
        try:
            data = self.post_fn(self.endpoint, payload)
            content = data["choices"][0]["message"]["content"]
            agent = _parse_agent(content)
        except Exception:
            return RouteResult("clarify", 0.0)
        if agent in self.valid:
            return RouteResult(agent, 1.0)
        return RouteResult("clarify", 0.0)
