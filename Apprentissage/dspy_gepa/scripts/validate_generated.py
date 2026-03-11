#!/usr/bin/env python3
"""Validation minimale des sorties JSONL generees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _expect_type(value: Any, expected_type: type, path: str, errors: list[str]) -> None:
    if not isinstance(value, expected_type):
        errors.append(f"{path}: expected {expected_type.__name__}, got {type(value).__name__}")


def validate_record(record: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []
    prefix = f"line {line_no}"
    required_top_level = ["source_id", "instruction", "context", "response", "metadata"]

    for key in required_top_level:
        if key not in record:
            errors.append(f"{prefix}: missing key '{key}'")

    if errors:
        return errors

    _expect_type(record["source_id"], str, f"{prefix}.source_id", errors)
    _expect_type(record["instruction"], str, f"{prefix}.instruction", errors)
    _expect_type(record["context"], str, f"{prefix}.context", errors)
    _expect_type(record["response"], str, f"{prefix}.response", errors)
    _expect_type(record["metadata"], dict, f"{prefix}.metadata", errors)

    if errors:
        return errors

    for key in ["source_id", "instruction", "response"]:
        if not record[key].strip():
            errors.append(f"{prefix}.{key}: must not be empty")

    metadata = record["metadata"]
    required_meta = ["domain", "language", "difficulty", "synthetic", "tags"]
    for key in required_meta:
        if key not in metadata:
            errors.append(f"{prefix}.metadata: missing key '{key}'")

    allowed_top = set(required_top_level)
    extra_top = sorted(set(record) - allowed_top)
    if extra_top:
        errors.append(f"{prefix}: unexpected keys {extra_top}")

    allowed_meta = set(required_meta)
    extra_meta = sorted(set(metadata) - allowed_meta)
    if extra_meta:
        errors.append(f"{prefix}.metadata: unexpected keys {extra_meta}")

    if errors:
        return errors

    _expect_type(metadata["domain"], str, f"{prefix}.metadata.domain", errors)
    _expect_type(metadata["language"], str, f"{prefix}.metadata.language", errors)
    _expect_type(metadata["difficulty"], str, f"{prefix}.metadata.difficulty", errors)
    _expect_type(metadata["synthetic"], bool, f"{prefix}.metadata.synthetic", errors)
    _expect_type(metadata["tags"], list, f"{prefix}.metadata.tags", errors)

    if errors:
        return errors

    if metadata["language"] != "fr":
        errors.append(f"{prefix}.metadata.language: expected 'fr'")

    if metadata["difficulty"] not in {"basic", "intermediate", "advanced"}:
        errors.append(f"{prefix}.metadata.difficulty: invalid value '{metadata['difficulty']}'")

    if metadata["synthetic"] is not True:
        errors.append(f"{prefix}.metadata.synthetic: expected true")

    for index, tag in enumerate(metadata["tags"]):
        if not isinstance(tag, str) or not tag.strip():
            errors.append(f"{prefix}.metadata.tags[{index}]: expected non-empty string")

    return errors


def validate_jsonl(path: Path) -> list[str]:
    all_errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(payload, dict):
                all_errors.append(f"line {line_no}: expected a JSON object")
                continue
            all_errors.extend(validate_record(payload, line_no))
    return all_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated synthetic JSONL records.")
    parser.add_argument("input_path", type=Path, help="Path to the generated JSONL file")
    args = parser.parse_args()

    errors = validate_jsonl(args.input_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Validation failed with {len(errors)} error(s).")
        return 1

    print("Validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
