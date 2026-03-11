#!/usr/bin/env python3
"""Generation DSPy baseline sans GEPA a partir de graines manuelles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import dspy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "source" / "manual_seed_examples.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "generated" / "baseline_generations.jsonl"
SCHEMA_PATH = ROOT / "schemas" / "synthetic_record.schema.json"
PROMPT_PATH = ROOT / "prompts" / "baseline_generation.md"


class GenerateSyntheticRecord(dspy.Signature):
    """Generate exactly one JSON object matching the provided schema and prompt contract."""

    source_seed = dspy.InputField(desc="Manual seed example as compact JSON.")
    schema_json = dspy.InputField(desc="Target JSON schema.")
    prompt_contract = dspy.InputField(desc="Short generation contract.")
    output_json = dspy.OutputField(desc="Strict JSON object, no markdown.")


class BaselineGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(GenerateSyntheticRecord)

    def forward(self, source_seed: str, schema_json: str, prompt_contract: str) -> dict[str, Any]:
        prediction = self.predict(
            source_seed=source_seed,
            schema_json=schema_json,
            prompt_contract=prompt_contract,
        )
        return json.loads(prediction.output_json)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must contain JSON objects")
            rows.append(payload)
    return rows


def configure_lm(model_name: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if base_url:
        lm = dspy.LM(model=model_name, api_key=api_key, api_base=base_url)
    else:
        lm = dspy.LM(model=model_name, api_key=api_key)

    dspy.configure(lm=lm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal DSPy baseline generation pipeline.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input manual seed JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output generated JSONL")
    parser.add_argument(
        "--model",
        default=os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini"),
        help="DSPy model identifier or provider/model name",
    )
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of seeds to process")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_lm(args.model)

    seeds = load_jsonl(args.source)[: args.limit]
    schema_json = SCHEMA_PATH.read_text(encoding="utf-8")
    prompt_contract = PROMPT_PATH.read_text(encoding="utf-8")

    generator = BaselineGenerator()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            generated = generator(
                source_seed=json.dumps(seed, ensure_ascii=False),
                schema_json=schema_json,
                prompt_contract=prompt_contract,
            )
            handle.write(json.dumps(generated, ensure_ascii=False) + "\n")

    print(f"Wrote {len(seeds)} generated record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
