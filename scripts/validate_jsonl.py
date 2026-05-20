#!/usr/bin/env python3
"""Validate JSONL training datasets for NuExtract ingredient extraction.

Checks:
  - Valid JSONL (one JSON object per line)
  - OpenAI messages format (user + assistant messages)
  - User message matches NuExtract 1.5 template (build_messages)
  - Assistant JSON: keys in order [quantity, unit, food, note], lowercase
  - Assistant JSON: formatting matches json.dumps() defaults
  - Assistant JSON: unit/note use "" not null; food is non-empty
  - No trailing whitespace, no blank lines, no duplicates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Must stay in sync with _NUEXTRACT_15_TEMPLATE in handlers/ingredient_parsing.py
_TEMPLATE = """\
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

_TEMPLATE_PREFIX = _TEMPLATE.split("%s")[0]
_TEMPLATE_SUFFIX = _TEMPLATE.split("%s")[1]
_REQUIRED_KEYS = ["quantity", "unit", "food", "note"]


def _validate_messages_structure(line_num: int, obj: dict, errors: list[str]) -> list[dict] | None:
    if "messages" not in obj:
        errors.append(f"  line {line_num}: missing 'messages' key")
        return None

    extra = set(obj.keys()) - {"messages"}
    if extra:
        errors.append(f"  line {line_num}: unexpected top-level keys: {extra}")

    messages = obj["messages"]
    if not isinstance(messages, list):
        errors.append(f"  line {line_num}: 'messages' must be a list")
        return None

    if len(messages) != 2:
        errors.append(f"  line {line_num}: expected 2 messages (user, assistant), got {len(messages)}")
        return None

    for i, expected_role in enumerate(["user", "assistant"]):
        msg = messages[i]
        role = msg.get("role")
        if role != expected_role:
            errors.append(f"  line {line_num}: message {i} role must be '{expected_role}', got '{role}'")
        if "content" not in msg:
            errors.append(f"  line {line_num}: {expected_role} message missing 'content'")
            return None
        extra_keys = set(msg.keys()) - {"role", "content"}
        if extra_keys:
            errors.append(f"  line {line_num}: {expected_role} message has unexpected keys: {extra_keys}")

    return messages


def _validate_user_content(line_num: int, content: str, errors: list[str]) -> str | None:
    if not content.startswith(_TEMPLATE_PREFIX):
        errors.append(f"  line {line_num}: user content doesn't match NuExtract template prefix")
        return None
    if not content.endswith(_TEMPLATE_SUFFIX):
        errors.append(f"  line {line_num}: user content doesn't match NuExtract template suffix")
        return None

    ingredient = content[len(_TEMPLATE_PREFIX) : -len(_TEMPLATE_SUFFIX)]
    if not ingredient.strip():
        errors.append(f"  line {line_num}: ingredient text is empty")
        return None

    return ingredient


def _validate_assistant_content(line_num: int, content: str, errors: list[str]) -> None:
    if content != content.rstrip():
        errors.append(f"  line {line_num}: assistant content has trailing whitespace")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"  line {line_num}: assistant content is not valid JSON: {e}")
        return

    if not isinstance(parsed, dict):
        errors.append(f"  line {line_num}: assistant content must be a JSON object")
        return

    keys = list(parsed.keys())
    if keys != _REQUIRED_KEYS:
        errors.append(f"  line {line_num}: keys must be {_REQUIRED_KEYS} in order, got {keys}")

    for key in parsed:
        if key != key.lower():
            errors.append(f"  line {line_num}: key '{key}' must be lowercase")

    quantity = parsed.get("quantity")
    if quantity is not None and not isinstance(quantity, (int, float)):
        errors.append(f"  line {line_num}: 'quantity' must be a number or null, got {type(quantity).__name__}")

    unit = parsed.get("unit")
    if unit is None:
        errors.append(f"  line {line_num}: 'unit' must be \"\" not null")
    elif not isinstance(unit, str):
        errors.append(f"  line {line_num}: 'unit' must be a string, got {type(unit).__name__}")

    food = parsed.get("food")
    if not isinstance(food, str):
        errors.append(f"  line {line_num}: 'food' must be a string, got {type(food).__name__}")
    elif food == "":
        errors.append(f"  line {line_num}: 'food' must not be empty")

    note = parsed.get("note")
    if note is None:
        errors.append(f"  line {line_num}: 'note' must be \"\" not null")
    elif not isinstance(note, str):
        errors.append(f"  line {line_num}: 'note' must be a string, got {type(note).__name__}")

    expected = json.dumps(parsed)
    if content != expected:
        errors.append(f"  line {line_num}: formatting doesn't match json.dumps() defaults")
        errors.append(f"    expected: {expected}")
        errors.append(f"    got:      {content}")


def _validate_line(line_num: int, raw: str, errors: list[str]) -> str | None:
    if not raw:
        errors.append(f"  line {line_num}: blank line")
        return None

    if raw != raw.rstrip():
        errors.append(f"  line {line_num}: trailing whitespace on line")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        errors.append(f"  line {line_num}: invalid JSON: {e}")
        return None

    messages = _validate_messages_structure(line_num, obj, errors)
    if messages is None:
        return None

    ingredient = _validate_user_content(line_num, messages[0]["content"], errors)
    _validate_assistant_content(line_num, messages[1]["content"], errors)

    return ingredient


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

    lines = text.splitlines()
    seen_ingredients: dict[str, int] = {}

    for i, line in enumerate(lines, 1):
        ingredient = _validate_line(i, line, errors)
        if ingredient is not None:
            if ingredient in seen_ingredients:
                errors.append(
                    f"  line {i}: duplicate ingredient '{ingredient}' "
                    f"(first seen on line {seen_ingredients[ingredient]})"
                )
            else:
                seen_ingredients[ingredient] = i

    return errors


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.jsonl> [<file2.jsonl> ...]", file=sys.stderr)
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
            count = sum(1 for _ in Path(filepath).open())
            print(f"OK   {filepath} ({count} entries)")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
