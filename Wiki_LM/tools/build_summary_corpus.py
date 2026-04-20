"""
Génère un corpus de distillation pour fine-tuner phi-4-mini sur la summarisation.

Pour chaque document dans raw/ (ou un fichier unique) :
  1. Chunker le texte
  2. Appeler DeepSeek sur chaque chunk → résumé Map
  3. Optionnellement générer les paires Refine (multi-étapes)
  4. Écrire au format JSONL ChatML (compatible phi-4-mini / transformers)

Les documents dépassant --large-threshold chunks sont signalés et ignorés
par défaut — utilisez --background pour les lancer en tâche de fond ou
--include-large pour les traiter en ligne.

Usage :
    python tools/build_summary_corpus.py
    python tools/build_summary_corpus.py --refine
    python tools/build_summary_corpus.py --file raw/doc.pdf
    python tools/build_summary_corpus.py --background   # grands docs en arrière-plan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from llm import LLM
from summarize import chunks, _SYSTEM, _PROMPT_MAP, _PROMPT_REFINE

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
DEFAULT_OUTPUT = CORPUS_DIR / "summary_corpus.jsonl"
LARGE_THRESHOLD = 30          # chunks — au-delà : document "large"
MAX_WORDS = 400
MAP_TOKENS = 300
REFINE_TOKENS = 500
SUPPORTED = (".url", ".md", ".pdf", ".txt", ".html")

# ---------------------------------------------------------------------------
# Lecture de documents
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            print(f"[corpus] pypdf manquant, ignoré : {path.name}", file=sys.stderr)
            return ""
    return path.read_text(errors="replace")


def _read_url_file(path: Path) -> str:
    """Lit l'URL dans un .url et retourne le contenu téléchargé."""
    import urllib.request
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("url:"):
            line = line[4:].strip()
        if line.startswith("http://") or line.startswith("https://"):
            try:
                req = urllib.request.Request(
                    line,
                    headers={"User-Agent": "WikiLM/1.0 (corpus builder; python-urllib)"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                print(f"[corpus] Impossible de télécharger {line} : {e}", file=sys.stderr)
                return ""
    return ""

# ---------------------------------------------------------------------------
# Génération des exemples
# ---------------------------------------------------------------------------

def _make_map_examples(blocs: list[str], llm: LLM) -> list[dict]:
    """Génère N exemples Map : (chunk → résumé)."""
    examples = []
    for i, chunk in enumerate(blocs, 1):
        print(f"  Map chunk {i}/{len(blocs)}…", end=" ", flush=True)
        try:
            summary = llm.complete(
                _PROMPT_MAP.format(chunk=chunk),
                system=_SYSTEM,
                max_tokens=MAP_TOKENS,
            )
            examples.append(_fmt_map(chunk, summary))
            print("OK")
        except Exception as e:
            print(f"ERREUR : {e}")
    return examples


def _make_refine_examples(blocs: list[str], llm: LLM) -> list[dict]:
    """Génère N exemples Refine : (résumé_courant + chunk → résumé_affiné)."""
    examples = []
    # Initialiser avec le premier chunk
    try:
        current = llm.complete(
            _PROMPT_MAP.format(chunk=blocs[0]),
            system=_SYSTEM,
            max_tokens=MAP_TOKENS,
        )
    except Exception as e:
        print(f"  Refine init ERREUR : {e}")
        return []

    for i, chunk in enumerate(blocs[1:], 2):
        print(f"  Refine chunk {i}/{len(blocs)}…", end=" ", flush=True)
        try:
            refined = llm.complete(
                _PROMPT_REFINE.format(current=current, chunk=chunk),
                system=_SYSTEM,
                max_tokens=REFINE_TOKENS,
            )
            examples.append(_fmt_refine(current, chunk, refined))
            current = refined
            print("OK")
        except Exception as e:
            print(f"ERREUR : {e}")

    return examples


def _fmt_map(chunk: str, summary: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _PROMPT_MAP.format(chunk=chunk)},
            {"role": "assistant", "content": summary},
        ]
    }


def _fmt_refine(current: str, chunk: str, refined: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _PROMPT_REFINE.format(current=current, chunk=chunk)},
            {"role": "assistant", "content": refined},
        ]
    }

# ---------------------------------------------------------------------------
# Traitement d'un fichier
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    llm: LLM,
    output: Path,
    refine: bool = False,
    large_threshold: int = LARGE_THRESHOLD,
) -> int:
    """Traite un fichier et ajoute les exemples à output. Retourne le nb d'exemples."""
    suffix = path.suffix.lower()
    text = _read_url_file(path) if suffix == ".url" else _read_file(path)
    if not text.strip():
        print(f"[corpus] Vide ou illisible, ignoré : {path.name}")
        return 0

    blocs = chunks(text, MAX_WORDS)
    n = len(blocs)
    print(f"[corpus] {path.name} → {n} chunk(s)")

    examples = _make_map_examples(blocs, llm)
    if refine and len(blocs) > 1:
        examples += _make_refine_examples(blocs, llm)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[corpus] → {len(examples)} exemple(s) ajouté(s) dans {output.name}")
    return len(examples)


def _launch_background(path: Path, args: argparse.Namespace, log_dir: Path) -> None:
    """Lance le traitement d'un fichier en arrière-plan via nohup."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"corpus_{path.stem}.log"
    cmd = [
        sys.executable, __file__,
        "--file", str(path),
        "--output", str(args.output),
        "--large-threshold", "99999",   # pas de limite en mode background
    ]
    if args.refine:
        cmd.append("--refine")

    proc = subprocess.Popen(
        ["nohup"] + cmd,
        stdout=log_file.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[corpus] Lancé en arrière-plan (PID {proc.pid}), log : {log_file}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un corpus de distillation pour phi-4-mini"
    )
    parser.add_argument(
        "--file", default="",
        help="Traiter un fichier unique (sinon : tous les fichiers de raw/)",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(Path.home() / "Secretarius/Wiki_LM/raw"),
        help="Répertoire raw/ à parcourir",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Fichier JSONL de sortie (défaut : {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="Générer aussi les paires Refine (coûteux : 2× les appels LLM)",
    )
    parser.add_argument(
        "--large-threshold", type=int, default=LARGE_THRESHOLD,
        help=f"Seuil de chunks pour considérer un doc comme 'large' (défaut : {LARGE_THRESHOLD})",
    )
    parser.add_argument(
        "--include-large", action="store_true",
        help="Traiter les grands documents en ligne (bloquant)",
    )
    parser.add_argument(
        "--background", action="store_true",
        help="Lancer les grands documents en tâche de fond (nohup)",
    )
    parser.add_argument("--backend", default="", help="Backend LLM")
    parser.add_argument("--model", default="", help="Modèle LLM")
    args = parser.parse_args()

    output = Path(args.output)
    log_dir = output.parent / "logs"
    llm = LLM(backend=args.backend, model=args.model) if (args.backend or args.model) else LLM()

    # Mode fichier unique
    if args.file:
        process_file(Path(args.file), llm, output, refine=args.refine,
                     large_threshold=args.large_threshold)
        return

    # Mode répertoire raw/
    raw_dir = Path(args.raw_dir).expanduser()
    files = sorted(
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED
    )
    print(f"[corpus] {len(files)} fichier(s) dans {raw_dir}")

    total = 0
    deferred = []

    for path in files:
        suffix = path.suffix.lower()
        # Estimer la taille sans lire tout le contenu pour les .url
        if suffix != ".url":
            text_sample = path.read_text(errors="replace")
            n_chunks = len(chunks(text_sample, MAX_WORDS))
        else:
            # Pour les .url on ne télécharge pas à l'avance — on traite directement
            n_chunks = 0

        if n_chunks > args.large_threshold:
            print(f"[corpus] ⚠ Grand document ({n_chunks} chunks) : {path.name}")
            if args.background:
                _launch_background(path, args, log_dir)
            elif args.include_large:
                total += process_file(path, llm, output, refine=args.refine,
                                      large_threshold=99999)
            else:
                deferred.append((path, n_chunks))
                print(f"[corpus]   → ignoré (utilisez --background ou --include-large)")
        else:
            total += process_file(path, llm, output, refine=args.refine,
                                  large_threshold=args.large_threshold)

    if deferred:
        print(f"\n[corpus] {len(deferred)} grand(s) document(s) différé(s) :")
        for p, n in deferred:
            print(f"  {p.name}  ({n} chunks)")
        print(
            f"\n  Pour les traiter :\n"
            f"  python tools/build_summary_corpus.py --background\n"
            f"  python tools/build_summary_corpus.py --include-large"
        )

    print(f"\n[corpus] Total : {total} exemple(s) → {output}")


if __name__ == "__main__":
    main()
