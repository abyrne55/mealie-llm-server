from __future__ import annotations

import json
import logging
from fractions import Fraction
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from llama_cpp import LlamaGrammar

from mealie_llm_server.handlers.base import Handler
from mealie_llm_server.models import ChatCompletionRequest, ChatCompletionResponse, build_chat_completion_response

if TYPE_CHECKING:
    from llama_cpp import Llama
    from mealie_llm_server.food_resolver import FoodResolver
    from mealie_llm_server.mealie_client import MealieClient

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


def extract_ingredients(content: str) -> list[str]:
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
        parts = value.strip().split()
        if len(parts) == 2 and "/" in parts[1]:
            return float(parts[0]) + float(Fraction(parts[1]))
        if len(parts) == 1 and "/" in parts[0]:
            return float(Fraction(parts[0]))
        return float(value)
    return float(value)


class IngredientParsingHandler(Handler):
    model_key = "ingredient_extractor"

    def __init__(self, food_resolver: FoodResolver | None = None, model_id: str = ""):
        self.reference_prompt = files("mealie_llm_server.prompts").joinpath("parse-recipe-ingredients.txt").read_text()
        self._food_resolver = food_resolver
        self._model_id = model_id

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

        grammar = LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))

        results = []
        for ingredient_text in ingredients:
            messages = build_messages(ingredient_text)
            response = model.create_chat_completion(
                messages=messages,
                grammar=grammar,
                temperature=0,
                max_tokens=-1,
            )
            content = response["choices"][0]["message"]["content"]
            raw = json.loads(content)

            heuristic = null_unit_heuristic(ingredient_text, raw.get("unit"), raw.get("food"), unit_aliases)
            raw["unit"] = resolve_unit(heuristic["unit"], unit_aliases)
            raw["food"] = heuristic["food"]
            raw["quantity"] = normalize_quantity(raw.get("quantity"))

            if self._food_resolver and foods and raw["food"]:
                raw["food"] = self._food_resolver.match(raw["food"], foods)

            results.append(raw)

        output = json.dumps({"ingredients": results})
        return build_chat_completion_response(content=output, model=self._model_id or "ingredient-parser")
