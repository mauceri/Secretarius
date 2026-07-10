#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génération de la page source via phi-4 nu (JSON contraint) + assemblage
déterministe du markdown au format wiki existant."""
import json
import os
import urllib.request

PHI4_BASE = os.environ.get("PHI4_BASE_URL", "http://127.0.0.1:8998")

_SYSTEM = ("Tu es l'assistant d'un wiki personnel. À partir UNIQUEMENT des passages "
           "fournis, produis un résumé fidèle et concis en français. N'invente rien "
           "qui ne figure pas dans les passages.")

_SCHEMA = {
    "type": "object",
    "properties": {
        "resume": {"type": "string"},
        "points_cles": {"type": "array", "items": {"type": "string"}},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "entites": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["resume", "points_cles", "concepts", "entites", "tags"],
}


def generate_page_content(passages: str, base_url: str = PHI4_BASE) -> dict:
    user = ("Passages :\n" + passages + "\n\n"
            "Produis un JSON : resume (3 à 5 phrases), points_cles (liste de points), "
            "concepts (concepts abstraits cités), entites (personnes/outils/organisations "
            "cités), tags (3 à 6 mots-clés).")
    body = {
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": 600, "temperature": 0,
        "lora": [{"id": 0, "scale": 0}],   # phi-4 nu, par-requête (ne touche pas l'état global)
        "json_schema": _SCHEMA,
    }
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return json.loads(d["choices"][0]["message"]["content"])
