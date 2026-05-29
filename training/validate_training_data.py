#!/usr/bin/env python3
"""Validate CSV training datasets for ingredient extraction.

Checks:
  - Valid CSV with expected header
  - Quantity is a valid number or empty
  - Food is non-empty
  - No trailing whitespace on lines
  - No duplicate ingredient_text values
  - File ends with a newline
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_EXPECTED_HEADER = ["ingredient_text", "quantity", "unit", "food", "note"]


def validate_file(filepath: str) -> list[str]:
    path = Path(filepath)
    errors: list[str] = []

    if not path.exists():
        return [f"  file not found: {filepath}"]

    text = path.read_text()
    if not text:
        return ["  file is empty"]

    if not text.endswith("\n"):
        errors.append("  file does not end with a newline")

    raw_lines = text.splitlines()
    for i, line in enumerate(raw_lines, 1):
        if line != line.rstrip():
            errors.append(f"  line {i}: trailing whitespace")

    reader = csv.reader(raw_lines)
    header = next(reader, None)
    if header != _EXPECTED_HEADER:
        errors.append(f"  header must be {_EXPECTED_HEADER}, got {header}")
        return errors

    seen: dict[str, int] = {}
    for row_num, row in enumerate(reader, 2):
        if len(row) != 5:
            errors.append(f"  line {row_num}: expected 5 fields, got {len(row)}")
            continue

        ingredient_text, qty_raw, unit, food, note = row

        if not ingredient_text.strip():
            errors.append(f"  line {row_num}: ingredient_text is empty")

        if qty_raw:
            try:
                float(qty_raw)
            except ValueError:
                errors.append(f"  line {row_num}: quantity '{qty_raw}' is not a valid number")

        if not food:
            errors.append(f"  line {row_num}: food must not be empty")

        key = ingredient_text.strip().lower()
        if key in seen:
            errors.append(
                f"  line {row_num}: duplicate ingredient '{ingredient_text}' (first seen on line {seen[key]})"
            )
        else:
            seen[key] = row_num

    return errors


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.csv> [<file2.csv> ...]", file=sys.stderr)
        sys.exit(2)

    failed = False
    for filepath in sys.argv[1:]:
        errors = validate_file(filepath)
        if errors:
            print(f"FAIL {filepath}")
            for err in errors:
                print(err)
            failed = True
        else:
            with open(filepath) as f:
                count = sum(1 for _ in f) - 1
            print(f"OK   {filepath} ({count} entries)")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
