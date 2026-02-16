from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import (
    DocRecord,
    LlamaExtractorClient,
    SimpleSemanticChunker,
    index_document,
    records_to_contract_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Secretarius phase-1 extraction CLI")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Raw input text to process")
    input_group.add_argument("--input-file", help="Path to a UTF-8 text file to process")
    parser.add_argument("--base-url", default="http://localhost:8080", help="llama-server base URL")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--min-sentences", type=int, default=2, help="Minimum sentences per chunk")
    parser.add_argument("--max-sentences", type=int, default=5, help="Maximum sentences per chunk")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def _load_text(text: str | None, input_file: str | None) -> str:
    if text is not None:
        return text
    if input_file is not None:
        return Path(input_file).read_text(encoding="utf-8")
    raise ValueError("Either --text or --input-file is required")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_sentences < 1 or args.max_sentences < 1:
        parser.error("--min-sentences and --max-sentences must be >= 1")
    if args.min_sentences > args.max_sentences:
        parser.error("--min-sentences cannot be greater than --max-sentences")

    text = _load_text(args.text, args.input_file)
    chunker = SimpleSemanticChunker(
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
    )
    extractor = LlamaExtractorClient(base_url=args.base_url, timeout_s=args.timeout_s)
    doc = DocRecord(id_doc="cli-input", source="cli", titre="cli", contenu=text)

    records = index_document(doc, chunker, extractor)
    payload = records_to_contract_json(records)

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
