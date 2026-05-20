#!/usr/bin/env python3
"""Render ingredients.jsonl as a readable Markdown file.

Writes to <input>.md (sibling of the input .jsonl). Exits 0 if already
up to date, 1 if the file was updated (for use as a pre-commit hook).

Usage:
    uv run python scripts/jsonl_to_markdown.py tests/integration/ingredients.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

JSONL_PATH = Path(__file__).resolve().parent.parent / "tests" / "integration" / "ingredients.jsonl"
MD_PATH = Path(__file__).resolve().parent.parent / "tests" / "integration" / "ingredients.md"

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


def generate(jsonl_path: Path) -> str:
    rows: list[tuple[str, str, str, str, str]] = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            user_content = entry["messages"][0]["content"]
            m = re.search(r"### Text:\n(.+?)\n\n<\|output\|>", user_content, re.DOTALL)
            ingredient = m.group(1).strip() if m else "???"

            output = json.loads(entry["messages"][1]["content"])
            rows.append(
                (
                    escape_pipe(ingredient),
                    fmt(output["quantity"]),
                    fmt(output["unit"]),
                    fmt(output["food"]),
                    fmt(output["note"]),
                )
            )

    lines = [
        f"# Training Dataset ({len(rows)} entries)\n",
        "## Prompt\n",
        PROMPT_TEMPLATE,
        "",
        "## Examples\n",
        "| # | Original | Qty | Unit | Food | Note |",
        "|---|----------|-----|------|------|------|",
    ]
    for i, (orig, qty, unit, food, note) in enumerate(rows, 1):
        lines.append(f"| {i} | {orig} | {qty} | {unit} | {food} | {note} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else JSONL_PATH
    md_path = jsonl_path.with_suffix(".md")
    content = generate(jsonl_path)

    if md_path.exists() and md_path.read_text() == content:
        sys.exit(0)

    md_path.write_text(content)
    print(f"Updated {md_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
