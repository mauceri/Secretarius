"""Clients LLM minces, OpenAI-compatibles, renvoyant (text, usage).

Seul module qui touche le réseau. usage = {prompt_tokens, completion_tokens}.
"""
from __future__ import annotations

import os
import time

_last_mistral_call: float = 0.0


def _extract(resp) -> tuple[str, dict]:
    text = resp.choices[0].message.content
    u = resp.usage
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
    return text, usage


def deepseek_generate(prompt: str, temperature: float = 0.9) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return _extract(resp)


def mistral_critique(prompt: str, max_tokens: int = 8,
                     temperature: float = 0.0) -> tuple[str, dict]:
    """Appel Mistral/Euria avec throttle 1 req/s (limite Infomaniak : 60 req/min)."""
    global _last_mistral_call
    elapsed = time.time() - _last_mistral_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_mistral_call = time.time()

    from openai import OpenAI
    pid = os.environ["EURIA_PRODUCT_ID"]
    client = OpenAI(base_url=f"https://api.infomaniak.com/2/ai/{pid}/openai/v1",
                    api_key=os.environ["EURIA_API_KEY"])
    resp = client.chat.completions.create(
        model="mistralai/Mistral-Small-4-119B-2603",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract(resp)
