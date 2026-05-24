"""Shared CSV loader for the ingredient training dataset."""

from __future__ import annotations

import csv
from pathlib import Path


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
