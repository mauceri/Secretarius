"""
Résumé de documents longs par Map-Reduce ou Refine.

Usage CLI :
    python tools/summarize.py document.txt
    python tools/summarize.py document.pdf --strategy refine
    python tools/summarize.py document.txt --max-words 300 --backend openai

Usage module :
    from summarize import summarize
    text = summarize(long_text, llm=llm, strategy="map-reduce")
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

from llm import LLM

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "Tu es un assistant de synthèse documentaire. "
    "Réponds uniquement en français, de manière concise et encyclopédique."
)

_PROMPT_MAP = """\
Résume ce passage en 3 à 5 phrases concises, en conservant les idées essentielles :

{chunk}"""

_PROMPT_REDUCE = """\
Voici plusieurs résumés partiels d'un même document.
Produis une synthèse cohérente et fluide en 6 à 10 phrases :

{summaries}"""

_PROMPT_REFINE = """\
Résumé en cours :
{current}

Nouveau passage à intégrer :
{chunk}

Affine le résumé pour intégrer les informations importantes de ce nouveau passage. \
Conserve la cohérence d'ensemble."""

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunks(text: str, max_words: int = 400) -> list[str]:
    """Découpe le texte en blocs de max_words mots avec léger chevauchement."""
    words = text.split()
    overlap = max_words // 10
    result = []
    i = 0
    while i < len(words):
        result.append(" ".join(words[i : i + max_words]))
        i += max_words - overlap
    return result

# ---------------------------------------------------------------------------
# Stratégies
# ---------------------------------------------------------------------------

def _map_reduce(
    text: str,
    llm: LLM,
    max_words: int,
    map_tokens: int,
    reduce_tokens: int,
) -> str:
    blocs = chunks(text, max_words)
    if len(blocs) == 1:
        return llm.complete(
            _PROMPT_MAP.format(chunk=blocs[0]),
            system=_SYSTEM,
            max_tokens=map_tokens,
        )
    # Map
    summaries = [
        llm.complete(
            _PROMPT_MAP.format(chunk=b),
            system=_SYSTEM,
            max_tokens=map_tokens,
        )
        for b in blocs
    ]
    # Reduce (récursif si les résumés sont encore trop nombreux)
    combined = "\n\n---\n\n".join(summaries)
    # Si combined est encore très long, on refait un niveau de map-reduce
    if len(combined.split()) > max_words * 2:
        return _map_reduce(combined, llm, max_words, map_tokens, reduce_tokens)
    return llm.complete(
        _PROMPT_REDUCE.format(summaries=combined),
        system=_SYSTEM,
        max_tokens=reduce_tokens,
    )


def _refine(
    text: str,
    llm: LLM,
    max_words: int,
    map_tokens: int,
    refine_tokens: int,
) -> str:
    blocs = chunks(text, max_words)
    current = llm.complete(
        _PROMPT_MAP.format(chunk=blocs[0]),
        system=_SYSTEM,
        max_tokens=map_tokens,
    )
    for bloc in blocs[1:]:
        current = llm.complete(
            _PROMPT_REFINE.format(current=current, chunk=bloc),
            system=_SYSTEM,
            max_tokens=refine_tokens,
        )
    return current

# ---------------------------------------------------------------------------
# Façade publique
# ---------------------------------------------------------------------------

def summarize(
    text: str,
    llm: LLM | None = None,
    strategy: Literal["map-reduce", "refine"] = "map-reduce",
    max_words: int = 400,
    map_tokens: int = 300,
    reduce_tokens: int = 600,
) -> str:
    """Résume un texte long.

    Params
    ------
    text        : texte source (peut être très long)
    llm         : instance LLM ; créée automatiquement si None
    strategy    : "map-reduce" (parallélisable) ou "refine" (plus cohérent)
    max_words   : taille des chunks en mots
    map_tokens  : tokens max par résumé de chunk
    reduce_tokens : tokens max pour la synthèse finale
    """
    if llm is None:
        llm = LLM()
    text = text.strip()
    if not text:
        return ""

    if strategy == "refine":
        return _refine(text, llm, max_words, map_tokens, reduce_tokens)
    return _map_reduce(text, llm, max_words, map_tokens, reduce_tokens)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Résumé de document par Map-Reduce ou Refine")
    parser.add_argument("source", help="Fichier texte ou PDF à résumer")
    parser.add_argument(
        "--strategy", choices=["map-reduce", "refine"], default="map-reduce",
        help="Stratégie de résumé (défaut : map-reduce)",
    )
    parser.add_argument("--max-words", type=int, default=400, help="Taille des chunks en mots")
    parser.add_argument("--backend", default="", help="Backend LLM")
    parser.add_argument("--model", default="", help="Modèle LLM")
    args = parser.parse_args()

    path = Path(args.source)
    if not path.exists():
        parser.error(f"Fichier introuvable : {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            parser.error("pip install pypdf")
    else:
        text = path.read_text(errors="replace")

    llm = LLM(backend=args.backend, model=args.model) if (args.backend or args.model) else LLM()

    n_chunks = len(chunks(text, args.max_words))
    print(f"[summarize] {n_chunks} chunk(s), stratégie : {args.strategy}")

    result = summarize(text, llm=llm, strategy=args.strategy, max_words=args.max_words)
    print("\n" + result)


if __name__ == "__main__":
    main()
