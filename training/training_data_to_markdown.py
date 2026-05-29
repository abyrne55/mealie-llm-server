#!/usr/bin/env python3
"""Render ingredients.csv as a readable Markdown file.

Writes to <input>.md (sibling of the input .csv). Exits 0 if already
up to date, 1 if the file was updated (for use as a pre-commit hook).

Usage:
    uv run python training/training_data_to_markdown.py training/ingredients.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.training_data import load_training_data  # noqa: E402

CSV_PATH = Path(__file__).parent / "ingredients.csv"

PROMPT_TEMPLATE = """\
```
<|input|>
### Template:
{
    "quantity": "",
    "unit": "",
    "food": "",
    "note": ""
}
### Text:
<ingredient string>

<|output|>
```"""


def escape_pipe(s: str) -> str:
    return s.replace("|", "\\|")


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return escape_pipe(str(value))


def generate(csv_path: Path) -> str:
    rows = load_training_data(csv_path)
    lines = [
        f"# Training Dataset ({len(rows)} entries)\n",
        "## Prompt\n",
        PROMPT_TEMPLATE,
        "",
        "## Examples\n",
        "| # | Original | Qty | Unit | Food | Note |",
        "|---|----------|-----|------|------|------|",
    ]
    for i, (ingredient, qty, unit, food, note) in enumerate(rows, 1):
        lines.append(f"| {i} | {escape_pipe(ingredient)} | {fmt(qty)} | {fmt(unit)} | {fmt(food)} | {fmt(note)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    md_path = csv_path.with_suffix(".md")
    content = generate(csv_path)

    if md_path.exists() and md_path.read_text() == content:
        sys.exit(0)

    md_path.write_text(content)
    print(f"Updated {md_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
