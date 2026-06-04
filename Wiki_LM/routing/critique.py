"""Critique d'un candidat de corpus par un modèle tiers (Mistral/Euria).

Le critique est DISTINCT du générateur (anti-circularité) : il confirme ou rejette
chaque exemple. Pas de ré-étiquetage — un candidat ambigu ou mal classé est rejeté.

Deux modes :
- individuel (critique_candidates) : 1 appel par candidat — simple, mais explose
  les rate limits (60 req/min Infomaniak) pour de grands corpus.
- batch (critique_batch) : 1 seul appel pour tous les candidats d'un agent — recommandé.
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
    """1 appel par candidat. critique_fn(prompt) -> (text, usage). Déconseillé sur gros corpus."""
    kept: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    for c in candidates:
        text, usage = critique_fn(build_critique_prompt(c, agents))
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
        if parse_verdict(text):
            kept.append(c)
    return kept, usage_total


# ── Mode batch ─────────────────────────────────────────────────────────────────

def build_batch_critique_prompt(candidates: list[dict], agents: list[dict]) -> str:
    """Prompt pour critiquer tous les candidats d'un même agent en un seul appel."""
    if not candidates:
        return ""
    target = candidates[0]["agent"]
    desc = next((a["description"] for a in agents if a["name"] == target), "")
    lines = [
        f'Ces messages ont été étiquetés pour l\'agent "{target}".',
        f'Rôle de "{target}" : {desc}',
        "Autres agents :",
    ]
    for a in agents:
        if a["name"] != target:
            lines.append(f'- {a["name"]} : {a["description"]}')
    lines += [
        "Pour chaque message ci-dessous, réponds GARDER ou REJETER "
        "(un mot par ligne, dans le même ordre, sans autre texte).",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f'{i}. "{c["message"]}"')
    return "\n".join(lines)


def parse_batch_verdicts(text: str, n: int) -> list[bool]:
    """Extrait n verdicts (bool) depuis la réponse batch. Manquant → False (rejeté)."""
    results: list[bool] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Ignorer les numéros éventuels : "1. GARDER" → "GARDER"
        if ". " in line:
            line = line.split(". ", 1)[1].strip()
        results.append(parse_verdict(line))
        if len(results) == n:
            break
    while len(results) < n:
        results.append(False)
    return results


def critique_batch(candidates: list[dict], agents: list[dict], critique_fn):
    """Critique tous les candidats en UN SEUL appel LLM. Retourne (gardés, usage).

    critique_fn(prompt, max_tokens) -> (text, usage).
    Tous les candidats doivent appartenir au même agent.
    """
    if not candidates:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = build_batch_critique_prompt(candidates, agents)
    max_tokens = max(16, len(candidates) * 8)
    text, usage = critique_fn(prompt, max_tokens)
    verdicts = parse_batch_verdicts(text, len(candidates))
    kept = [c for c, v in zip(candidates, verdicts) if v]
    return kept, usage
