from __future__ import annotations

import json
import logging
from fractions import Fraction
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np
from llama_cpp import LlamaGrammar, LogitsProcessorList

from mealie_llm_server.handlers.base import Handler
from mealie_llm_server.models import ChatCompletionRequest, ChatCompletionResponse, build_chat_completion_response

if TYPE_CHECKING:
    from llama_cpp import Llama
    from mealie_llm_server.mealie_client import MealieClient

logger = logging.getLogger(__name__)

_NUEXTRACT_TEMPLATE = """\
{
    "quantity": "number",
    "unit": "string",
    "food": "string",
    "note": "verbatim-string"
}"""


def extract_ingredients(content: str) -> list[str]:
    return json.loads(content)


def build_nuextract_messages(ingredient_text: str) -> list[dict[str, str]]:
    prompt = f"# Template:\n{_NUEXTRACT_TEMPLATE}\n\n# Context:\n{ingredient_text}"
    return [{"role": "user", "content": prompt}]


def build_ingredient_schema(
    foods: list[str], units: list[str], *, allow_null_food: bool = True,
) -> dict[str, Any]:
    if foods:
        food_prop: dict[str, Any] = {"enum": foods + ([None] if allow_null_food else [])}
    else:
        food_prop = {"type": ["string", "null"] if allow_null_food else "string"}

    if units:
        unit_prop: dict[str, Any] = {"enum": units + [None]}
    else:
        unit_prop = {"type": ["string", "null"]}

    return {
        "type": "object",
        "properties": {
            "quantity": {"type": ["number", "null"]},
            "unit": unit_prop,
            "food": food_prop,
            "note": {"type": ["string", "null"]},
        },
        "required": ["quantity", "unit", "food", "note"],
        "additionalProperties": False,
    }


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
    aliases = unit_aliases.get(unit, [unit])
    for alias in aliases:
        if alias.lower() in text_lower:
            return {"unit": unit, "food": food}

    return {"unit": None, "food": food}



_FOOD_LOGPROB_THRESHOLD = -5.0


class _LogprobCapture:
    """Captures per-token logprobs during generation via logits_processor."""

    def __init__(self):
        self.step_logprobs: list[np.ndarray] = []

    def __call__(self, input_ids, logits):
        self.step_logprobs.append((logits - np.logaddexp.reduce(logits)).copy())
        return logits

    def reset(self):
        self.step_logprobs = []

    def food_confidence(self, model: Llama, output_text: str, food_value: str) -> float:
        token_ids = model.tokenize(output_text.encode("utf-8"), add_bos=False)
        for marker in (f'"food": "{food_value}"', f'"food":"{food_value}"'):
            idx = output_text.find(marker)
            if idx != -1:
                val_start = idx + marker.index(food_value)
                break
        else:
            return 0.0
        val_end = val_start + len(food_value)
        token_strs = [model.detokenize([tid]).decode("utf-8", errors="replace") for tid in token_ids]
        pos = 0
        food_logprobs = []
        for step, (tid, ts) in enumerate(zip(token_ids, token_strs)):
            token_end = pos + len(ts)
            if pos < val_end and token_end > val_start and step < len(self.step_logprobs):
                food_logprobs.append(float(self.step_logprobs[step][tid]))
            pos = token_end
        if not food_logprobs:
            return 0.0
        return min(food_logprobs)


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
    model_key = "ingredient_parsing"

    def __init__(self):
        self.reference_prompt = (
            files("mealie_llm_server.prompts")
            .joinpath("parse-recipe-ingredients.txt")
            .read_text()
        )

    async def handle(
        self,
        request: ChatCompletionRequest,
        model: Llama,
        mealie_client: MealieClient,
    ) -> ChatCompletionResponse:
        user_msg = next(m for m in request.messages if m.role == "user")
        ingredients = extract_ingredients(user_msg.content)

        foods = await mealie_client.get_foods()
        units = await mealie_client.get_units()
        unit_aliases = await mealie_client.get_unit_aliases()

        schema = build_ingredient_schema(foods, units, allow_null_food=not foods)

        capture = _LogprobCapture() if foods else None
        logits_processor = LogitsProcessorList([capture]) if capture else None

        results = []
        for ingredient_text in ingredients:
            if capture:
                capture.reset()
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
            messages = build_nuextract_messages(ingredient_text)
            response = model.create_chat_completion(
                messages=messages,
                grammar=grammar,
                temperature=0,
                max_tokens=-1,
                logits_processor=logits_processor,
            )
            content = response["choices"][0]["message"]["content"]
            raw = json.loads(content)

            heuristic = null_unit_heuristic(ingredient_text, raw.get("unit"), raw.get("food"), unit_aliases)
            raw["unit"] = heuristic["unit"]
            raw["food"] = heuristic["food"]
            raw["quantity"] = normalize_quantity(raw.get("quantity"))

            if capture and raw["food"] is not None:
                confidence = capture.food_confidence(model, content, raw["food"])
                logger.debug(
                    "Food confidence for %r: food=%r confidence=%.2f",
                    ingredient_text, raw["food"], confidence,
                )
                if confidence < _FOOD_LOGPROB_THRESHOLD:
                    logger.debug(
                        "Low-confidence food %r (%.2f) for %r, nulling out",
                        raw["food"], confidence, ingredient_text,
                    )
                    raw["food"] = None

            results.append(raw)

        output = json.dumps({"ingredients": results})
        return build_chat_completion_response(content=output, model="nuextract-2.0-2b")
