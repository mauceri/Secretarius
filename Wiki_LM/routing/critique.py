"""Critique d'un candidat de corpus par un modèle tiers (Mistral/Euria).

Le critique est DISTINCT du générateur (anti-circularité) : il confirme ou rejette
chaque exemple. Pas de ré-étiquetage — un candidat ambigu ou mal classé est rejeté.
"""
from __future__ import annotations


def build_critique_prompt(candidate: dict, agents: list[dict]) -> str:
    target = candidate["agent"]
    desc = next((a["description"] for a in agents if a["name"] == target), "")
    lines = [
        f'Un message a été étiqueté comme relevant de l\'agent "{target}".',
        f'Rôle de "{target}" : {desc}',
        "Autres agents :",
    ]
    for a in agents:
        if a["name"] != target:
            lines.append(f'- {a["name"]} : {a["description"]}')
    lines += [
        f'Message : "{candidate["message"]}"',
        "Ce message relève-t-il clairement et sans ambiguïté de cet agent, et d'aucun autre ?",
        "Réponds par un seul mot : GARDER ou REJETER.",
    ]
    return "\n".join(lines)


def parse_verdict(text: str) -> bool:
    """True (garder) seulement si GARDER présent ET REJETER absent."""
    up = text.upper()
    return "GARDER" in up and "REJETER" not in up


def critique_candidates(candidates: list[dict], agents: list[dict], critique_fn):
    """Applique le critique à chaque candidat. Retourne (gardés, usage_cumulé).

    critique_fn(prompt) -> (text, usage) ; usage = {prompt_tokens, completion_tokens}.
    """
    kept: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    for c in candidates:
        text, usage = critique_fn(build_critique_prompt(c, agents))
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
        if parse_verdict(text):
            kept.append(c)
    return kept, usage_total
