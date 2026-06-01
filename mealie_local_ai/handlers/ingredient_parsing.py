from __future__ import annotations

import json
import logging
import re
import unicodedata
from fractions import Fraction
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from llama_cpp import LlamaGrammar

from mealie_local_ai.handlers.base import Handler
from mealie_local_ai.models import ChatCompletionRequest, ChatCompletionResponse, build_chat_completion_response
from mealie_local_ai.regex_parser import RegexParser, decimal_to_fraction, normalize_numeric_text

if TYPE_CHECKING:
    from llama_cpp import Llama
    from mealie_local_ai.food_resolver import FoodResolver
    from mealie_local_ai.mealie_client import MealieClient

logger = logging.getLogger(__name__)

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

_STRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "quantity": {"type": ["number", "null"]},
        "unit": {"type": ["string", "null"]},
        "food": {"type": "string"},
        "note": {"type": ["string", "null"]},
    },
    "required": ["quantity", "unit", "food", "note"],
    "additionalProperties": False,
}


def extract_ingredients(content: str | list) -> list[str]:
    if isinstance(content, list):
        text = "\n".join(part["text"] for part in content if part.get("type") == "text")
        return json.loads(text)
    return json.loads(content)


def build_messages(ingredient_text: str) -> list[dict[str, str]]:
    prompt = _NUEXTRACT_15_TEMPLATE % ingredient_text
    return [{"role": "user", "content": prompt}]


def _is_known_unit(value: str, unit_aliases: dict[str, list[str]]) -> bool:
    for aliases in unit_aliases.values():
        if value in aliases:
            return True
    return False


def null_unit_heuristic(
    original_text: str,
    unit: str | None,
    food: str | None,
    unit_aliases: dict[str, list[str]],
) -> dict[str, str | None]:
    if unit is None:
        return {"unit": None, "food": food}

    if unit == food:
        return {"unit": None, "food": food}

    if food is None and not _is_known_unit(unit, unit_aliases):
        return {"unit": None, "food": unit}

    text_lower = original_text.lower()
    text_words = set(text_lower.split())
    aliases = unit_aliases.get(unit, [unit])
    for alias in aliases:
        a = alias.lower()
        if a in text_words or a in text_lower:
            return {"unit": unit, "food": food}

    return {"unit": None, "food": food}


def resolve_unit(extracted: str | None, unit_aliases: dict[str, list[str]]) -> str | None:
    if not extracted:
        return None
    lower = extracted.lower().strip()
    for canonical, aliases in unit_aliases.items():
        for alias in aliases:
            if alias.lower() == lower:
                return canonical
    return None


def normalize_quantity(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s and unicodedata.category(s[-1]) == "No":
            frac = unicodedata.numeric(s[-1])
            prefix = s[:-1].strip()
            return float(prefix) + frac if prefix else frac
        parts = s.split()
        if len(parts) == 2 and "/" in parts[1]:
            return float(parts[0]) + float(Fraction(parts[1]))
        if len(parts) == 1 and "/" in parts[0]:
            return float(Fraction(parts[0]))
        return float(value)
    return float(value)


def extract_raw(ingredient_text: str, model: Llama, grammar: LlamaGrammar) -> dict:
    normalized = normalize_numeric_text(ingredient_text)
    messages = build_messages(normalized)
    response = model.create_chat_completion(
        messages=messages,
        grammar=grammar,
        temperature=0,
        max_tokens=-1,
    )
    content = response["choices"][0]["message"]["content"]
    content = re.sub(r"[\x00-\x1f\x7f]", "", content)
    return json.loads(content)


def parse_single_ingredient(
    ingredient_text: str,
    model: Llama,
    grammar: LlamaGrammar,
    unit_aliases: dict[str, list[str]],
    food_resolver: FoodResolver | None = None,
    foods: list[str] | None = None,
    regex_parser: RegexParser | None = None,
) -> tuple[dict, dict]:
    """Returns (result, trace) where trace captures each pipeline stage."""
    normalized = normalize_numeric_text(ingredient_text)
    trace: dict = {"normalized_input": normalized}

    if regex_parser is not None:
        regex_result = regex_parser.try_parse(normalized)
        if regex_result is not None:
            trace["source"] = "regex"
            trace["raw_extraction"] = dict(regex_result)
            regex_result["unit"] = resolve_unit(regex_result.get("unit"), unit_aliases)
            trace["resolved_unit"] = regex_result["unit"]
            if regex_result["note"]:
                regex_result["note"] = re.sub(
                    r"\d+\.\d+", lambda m: decimal_to_fraction(float(m.group())), regex_result["note"]
                )
            logger.info("Regex match: %r -> %s", ingredient_text, regex_result)
            return regex_result, trace

    trace["source"] = "llm"

    try:
        raw = extract_raw(ingredient_text, model, grammar)
        trace["raw_extraction"] = {
            "quantity": raw.get("quantity"),
            "unit": raw.get("unit"),
            "food": raw.get("food"),
            "note": raw.get("note"),
        }

        heuristic = null_unit_heuristic(normalized, raw.get("unit"), raw.get("food"), unit_aliases)
        trace["post_heuristic"] = dict(heuristic)

        raw["unit"] = resolve_unit(heuristic["unit"], unit_aliases)
        trace["resolved_unit"] = raw["unit"]
        raw["food"] = heuristic["food"]
        raw["quantity"] = normalize_quantity(raw.get("quantity"))

        extracted = dict(raw)

        if food_resolver and foods and raw["food"]:
            query = f"{raw['food']} {raw['note']}" if raw.get("note") else raw["food"]
            resolved_food, score, exact = food_resolver.match(query, foods)
            trace["resolved_food"] = {"food": resolved_food, "score": score, "exact": exact}
            if resolved_food is not None:
                raw["food"] = resolved_food
            suffix = " (exact)" if exact else " (fallback)" if resolved_food is None else ""
            logger.debug(
                "Ingredient: %r | extracted: %s | resolved: %s | food_score: %.3f%s",
                ingredient_text,
                json.dumps(extracted),
                json.dumps(raw),
                score,
                suffix,
            )
        else:
            logger.debug(
                "Ingredient: %r | extracted: %s | resolved: (skipped)",
                ingredient_text,
                json.dumps(extracted),
            )
    except Exception:
        logger.warning("Failed to parse ingredient %r; returning blank", ingredient_text, exc_info=True)
        raw = {"quantity": None, "unit": None, "food": "", "note": ingredient_text}

    if raw.get("note"):
        raw["note"] = re.sub(r"\d+\.\d+", lambda m: decimal_to_fraction(float(m.group())), raw["note"])

    return raw, trace


class IngredientParsingHandler(Handler):
    model_key = "ingredient_extractor"

    def __init__(self, food_resolver: FoodResolver | None = None, model_id: str = ""):
        self.reference_prompt = files("mealie_local_ai.prompts").joinpath("parse-recipe-ingredients.txt").read_text()
        self._food_resolver = food_resolver
        self._model_id = model_id
        self._regex_parser = RegexParser()

    async def handle(
        self,
        request: ChatCompletionRequest,
        model: Llama,
        mealie_client: MealieClient,
    ) -> ChatCompletionResponse:
        user_msg = next(m for m in request.messages if m.role == "user")
        ingredients = extract_ingredients(user_msg.content)

        foods = await mealie_client.get_foods()
        unit_aliases = await mealie_client.get_unit_aliases()

        all_units = [a for aliases in unit_aliases.values() for a in aliases]
        self._regex_parser.build(foods, all_units)

        grammar = LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))

        results = []
        for ingredient_text in ingredients:
            result, _ = parse_single_ingredient(
                ingredient_text,
                model,
                grammar,
                unit_aliases,
                food_resolver=self._food_resolver,
                foods=foods,
                regex_parser=self._regex_parser,
            )
            results.append(result)

        output = json.dumps({"ingredients": results})
        return build_chat_completion_response(content=output, model=self._model_id or "ingredient-parser")
