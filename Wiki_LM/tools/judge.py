"""Juge de fidélité : compare un résumé généré à son texte source, signale
les substitutions/inventions factuelles. Premier jet écrit à la main
(2026-09-10), pas encore câblé dans le pipeline d'ingestion — outil de
validation avant de décider s'il vaut la peine d'être intégré, et avant un
éventuel passage GEPA (voir gen_corpus/promptGenGEPA.py pour le patron déjà
utilisé dans ce projet).

Usage :
    from judge import judge
    from llm import LLM
    result = judge(source_text, summary_text, LLM(backend="ollama", model="qwen3:8b"))
"""

from __future__ import annotations

_SYSTEM_JUDGE = """\
Tu es un vérificateur factuel. On te donne un texte source et un résumé \
censé en être fidèle. Ta seule tâche : repérer si le résumé contredit, \
substitue ou invente un fait par rapport à la source — jamais juger le \
style, la longueur, les omissions ou les choix de formulation.

Cherche spécifiquement :
- une substitution d'espèce, de nom propre, de lieu, de date ou de nombre
- une relation causale ou une conclusion absente de la source
- une affirmation présentée comme un fait sans appui dans la source

Ne signale PAS :
- une reformulation, une paraphrase, une simplification légitimes
- une omission de détails (le résumé peut être incomplet)
- un jugement de valeur ou une interprétation raisonnable du texte source

Format de réponse strict, rien d'autre :
VERDICT: fidèle ou douteux
PROBLEME: <citation exacte du résumé en cause, ou "aucun">
JUSTIFICATION: <une phrase, citant le passage correspondant de la source>"""

_PROMPT_JUDGE = """\
Texte source :
---
{source}
---

Résumé à vérifier :
---
{summary}
---"""


def judge(source: str, summary: str, llm) -> dict:
    """Retourne {verdict, probleme, justification, raw}. verdict vaut
    "fidèle", "douteux" ou "indéterminé" (réponse du LLM non parsable)."""
    prompt = _PROMPT_JUDGE.format(source=source, summary=summary)
    raw = llm.complete(prompt, system=_SYSTEM_JUDGE, max_tokens=400)

    verdict = "indéterminé"
    probleme = ""
    justification = ""
    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            v = line[len("VERDICT:"):].strip().lower()
            verdict = "douteux" if "douteux" in v else "fidèle" if "fidèle" in v else "indéterminé"
        elif line.startswith("PROBLEME:"):
            probleme = line[len("PROBLEME:"):].strip()
        elif line.startswith("JUSTIFICATION:"):
            justification = line[len("JUSTIFICATION:"):].strip()

    return {"verdict": verdict, "probleme": probleme, "justification": justification, "raw": raw}
