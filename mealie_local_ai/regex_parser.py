from __future__ import annotations

import re
import unicodedata
from fractions import Fraction

_UFRAC = "[¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]"
_QTY_RE = re.compile(rf"\d+\s+\d+/\d+|\d+\s*{_UFRAC}|\d+/\d+|\d+\.?\d*|{_UFRAC}")
_SIZE_RE = re.compile(r"(?:small|medium|large)\s+", re.IGNORECASE)


def _parse_qty(s: str) -> float:
    s = s.strip()
    if s and unicodedata.category(s[-1]) == "No":
        frac = unicodedata.numeric(s[-1])
        prefix = s[:-1].strip()
        return float(prefix) + frac if prefix else frac
    parts = s.split()
    if len(parts) == 2 and "/" in parts[1]:
        return float(parts[0]) + float(Fraction(parts[1]))
    if "/" in s:
        return float(Fraction(s))
    return float(s)


class RegexParser:
    def __init__(self) -> None:
        self._unit_re: re.Pattern | None = None
        self._food_lookup: dict[str, str] = {}
        self._unit_lookup: dict[str, str] = {}

    def build(self, foods: list[str], units: list[str]) -> None:
        self._food_lookup = {name.lower(): name for name in foods}
        self._unit_lookup = {u.lower(): u for u in units}
        escaped = [re.escape(u) for u in sorted(units, key=len, reverse=True)]
        pattern = rf"(?:{'|'.join(escaped)})\s+"
        self._unit_re = re.compile(pattern, re.IGNORECASE)

    def try_parse(self, ingredient_text: str) -> dict | None:
        if self._unit_re is None:
            return None

        text = ingredient_text.strip()
        if not text:
            return None

        qty_match = _QTY_RE.match(text)
        if not qty_match:
            return None

        qty = _parse_qty(qty_match.group(0))
        remaining = text[qty_match.end() :].strip()

        if not remaining:
            return None

        note = None
        if ", " in remaining:
            main, note = remaining.split(", ", 1)
        else:
            main = remaining

        unit_match = self._unit_re.match(main)
        if unit_match:
            food_part = main[unit_match.end() :]
            food = self._food_lookup.get(food_part.lower())
            if food is not None:
                unit_str = main[: unit_match.end()].rstrip()
                unit = self._unit_lookup.get(unit_str.lower())
                return {"quantity": qty, "unit": unit, "food": food, "note": note}

        last_space = main.rfind(" ")
        if last_space > 0:
            food_part = main[:last_space]
            unit_part = main[last_space + 1 :]
            food = self._food_lookup.get(food_part.lower())
            unit = self._unit_lookup.get(unit_part.lower())
            if food is not None and unit is not None:
                return {"quantity": qty, "unit": unit, "food": food, "note": note}

        size_match = _SIZE_RE.match(main)
        if size_match:
            food_part = main[size_match.end() :]
            food = self._food_lookup.get(food_part.lower())
            if food is not None:
                return {"quantity": qty, "unit": None, "food": food, "note": note}

        food = self._food_lookup.get(main.lower())
        if food is not None:
            return {"quantity": qty, "unit": None, "food": food, "note": note}

        return None
