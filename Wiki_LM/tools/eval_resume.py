#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Éval du résumeur phi-4 : le juge DeepSeek note fidélité (candidat vs source) et
couverture (candidat vs page de référence Euria), sur 1..5."""
import json
import os
import urllib.request

from central_passages import select_central_passages
from page_phi4 import generate_page_content, assemble_source_page

_JUGE = ("Tu es un évaluateur strict. Compare un RÉSUMÉ CANDIDAT à sa SOURCE et à une "
         "page de RÉFÉRENCE. Note de 1 à 5 :\n"
         "- fidelite : le candidat n'affirme rien qui ne soit dans la SOURCE (5 = aucune "
         "invention, 1 = hallucinations).\n"
         "- couverture : le candidat couvre le contenu central de la RÉFÉRENCE (5 = tout "
         "l'essentiel, 1 = manque l'essentiel).\n"
         'Réponds UNIQUEMENT par un JSON {"fidelite": <int>, "couverture": <int>}.')


def judge_resume(source: str, reference: str, candidate: str) -> dict:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    user = f"SOURCE:\n{source}\n\nRÉFÉRENCE:\n{reference}\n\nCANDIDAT:\n{candidate}"
    body = {"model": "deepseek-chat",
            "messages": [{"role": "system", "content": _JUGE},
                         {"role": "user", "content": user}],
            "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    raw = d["choices"][0]["message"]["content"]
    start, end = raw.find("{"), raw.rfind("}")
    parsed = json.loads(raw[start:end + 1])
    return {"fidelite": int(parsed["fidelite"]), "couverture": int(parsed["couverture"])}


def run_eval(paires: list[dict]) -> list[dict]:
    """paires : [{"source": str, "reference": str, "titre": str}]"""
    resultats = []
    for p in paires:
        passages = select_central_passages(p["source"])
        data = generate_page_content(passages)
        candidate = assemble_source_page(p.get("titre", "Source"), "2026-07-09",
                                         data, data.get("tags", []))
        notes = judge_resume(passages, p["reference"], candidate)
        resultats.append({"titre": p.get("titre"), **notes})
    return resultats
