"""Shared CSV loader for the ingredient training dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

_NUEXTRACT_15_TEMPLATE = """\
<|input|>
### Template:
{
    "quantity": "",
    "unit": "",
    "food": "",
    "note": ""
}
### Text:
%s

<|output|>
"""


def load_training_data(
    path: Path,
) -> list[tuple[str, int | float | None, str, str, str]]:
    rows: list[tuple[str, int | float | None, str, str, str]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qty_raw = row["quantity"]
            if not qty_raw:
                qty = None
            elif "." in qty_raw:
                qty = float(qty_raw)
            else:
                qty = int(qty_raw)
            rows.append(
                (
                    row["ingredient_text"],
                    qty,
                    row["unit"],
                    row["food"],
                    row["note"],
                )
            )
    return rows


def build_messages_for_training(ingredient_text: str) -> list[dict[str, str]]:
    """Build OpenAI messages for training. Mirrors build_messages() in handlers/ingredient_parsing.py."""
    return [{"role": "user", "content": _NUEXTRACT_15_TEMPLATE % ingredient_text}]


def rows_to_dataset(rows: list[tuple[str, int | float | None, str, str, str]]) -> list[dict]:
    """Convert CSV rows to OpenAI messages format for SFTTrainer."""
    records = []
    for ingredient_text, qty, unit, food, note in rows:
        messages = build_messages_for_training(ingredient_text)
        output = {"quantity": qty, "unit": unit, "food": food, "note": note}
        messages.append({"role": "assistant", "content": json.dumps(output)})
        records.append({"messages": messages})
    return records
