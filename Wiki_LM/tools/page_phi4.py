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


def _bullets(items, prefix: str = "") -> str:
    return "\n".join(f"- {prefix}{x}" for x in items) if items else ""


def assemble_source_page(title: str, today: str, data: dict, tags: list[str]) -> str:
    all_tags = tags + data.get("tags", [])
    tags_uniq = list(dict.fromkeys(all_tags))              # dédup, ordre préservé
    tags_str = ", ".join(json.dumps(t, ensure_ascii=False) for t in tags_uniq)
    points = _bullets(data.get("points_cles", []))
    concepts = _bullets(data.get("concepts", []), "concept: ")
    entites = _bullets(data.get("entites", []), "entité: ")
    conc_ent = "\n".join(x for x in (concepts, entites) if x) or "Aucun"
    resume = (data.get("resume") or "").strip()
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        "category: source\n"
        f"tags: [{tags_str}]\n"
        f"created: {today}\n"
        "sources: []\n"
        "---\n\n"
        f"# {title}\n\n"
        "## Résumé\n\n"
        f"{resume}\n\n"
        "## Points clés\n\n"
        f"{points}\n\n"
        "## Concepts et entités mentionnés\n\n"
        f"{conc_ent}\n\n"
        "## Liens internes suggérés\n\n"
        "Aucun\n"
    )
