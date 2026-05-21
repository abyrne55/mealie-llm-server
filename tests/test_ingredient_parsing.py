import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mealie_local_ai.handlers.ingredient_parsing import (
    extract_ingredients,
    build_messages,
    null_unit_heuristic,
    resolve_unit,
    normalize_quantity,
)
from mealie_local_ai.models import ChatCompletionRequest


class TestExtractIngredients:
    def test_parses_json_list(self):
        content = '["1 cup flour", "2 eggs"]'
        result = extract_ingredients(content)
        assert result == ["1 cup flour", "2 eggs"]

    def test_single_ingredient(self):
        content = '["1 tbsp olive oil"]'
        result = extract_ingredients(content)
        assert result == ["1 tbsp olive oil"]


class TestBuildMessages:
    def test_nuextract_15_format(self):
        messages = build_messages("1 cup flour")
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "<|input|>" in content
        assert "### Template:" in content
        assert "### Text:" in content
        assert "<|output|>" in content
        assert "1 cup flour" in content


class TestNullUnitHeuristic:
    def test_strips_unit_not_in_input(self):
        result = null_unit_heuristic(
            original_text="3 eggs",
            unit="piece",
            food="egg",
            unit_aliases={},
        )
        assert result["unit"] is None
        assert result["food"] == "egg"

    def test_keeps_unit_when_abbreviation_in_input(self):
        result = null_unit_heuristic(
            original_text="1 tbsp olive oil",
            unit="tablespoon",
            food="olive oil",
            unit_aliases={"tablespoon": ["tablespoon", "tbsp", "tablespoons"]},
        )
        assert result["unit"] == "tablespoon"

    def test_keeps_unit_when_exact_match_in_input(self):
        result = null_unit_heuristic(
            original_text="1 cup flour",
            unit="cup",
            food="flour",
            unit_aliases={},
        )
        assert result["unit"] == "cup"

    def test_swaps_unit_and_food_when_unit_is_food(self):
        result = null_unit_heuristic(
            original_text="2 jalapeños",
            unit="jalapeño",
            food=None,
            unit_aliases={},
        )
        assert result["unit"] is None
        assert result["food"] == "jalapeño"

    def test_strips_unit_matching_food(self):
        result = null_unit_heuristic(
            original_text="3 eggs",
            unit="egg",
            food="egg",
            unit_aliases={},
        )
        assert result["unit"] is None
        assert result["food"] == "egg"

    def test_null_unit_passthrough(self):
        result = null_unit_heuristic(
            original_text="salt and pepper to taste",
            unit=None,
            food=None,
            unit_aliases={},
        )
        assert result["unit"] is None
        assert result["food"] is None

    def test_keeps_unit_joined_to_number(self):
        result = null_unit_heuristic(
            original_text="1lb ground chicken",
            unit="lb",
            food="chicken",
            unit_aliases={"pound": ["pound", "pounds", "lb", "lbs"]},
        )
        assert result["unit"] == "lb"
        assert result["food"] == "chicken"


class TestResolveUnit:
    def test_resolves_canonical(self):
        aliases = {"cup": ["cup", "cups", "c"], "tablespoon": ["tablespoon", "tbsp"]}
        assert resolve_unit("cup", aliases) == "cup"

    def test_resolves_alias(self):
        aliases = {"cup": ["cup", "cups", "c"], "tablespoon": ["tablespoon", "tbsp"]}
        assert resolve_unit("tbsp", aliases) == "tablespoon"

    def test_resolves_case_insensitive(self):
        aliases = {"cup": ["cup", "cups", "c"]}
        assert resolve_unit("Cups", aliases) == "cup"

    def test_returns_none_for_unknown(self):
        aliases = {"cup": ["cup", "cups"]}
        assert resolve_unit("pieces", aliases) is None

    def test_returns_none_for_empty(self):
        assert resolve_unit(None, {}) is None
        assert resolve_unit("", {}) is None


class TestNormalizeQuantity:
    def test_integer(self):
        assert normalize_quantity(2) == 2.0

    def test_float(self):
        assert normalize_quantity(0.5) == 0.5

    def test_none(self):
        assert normalize_quantity(None) is None

    def test_mixed_number_string(self):
        assert normalize_quantity("1 1/2") == 1.5

    def test_fraction_string(self):
        assert normalize_quantity("1/2") == 0.5

    def test_whole_number_string(self):
        assert normalize_quantity("3") == 3.0


class TestIngredientParsingHandler:
    def _make_mock_mealie(self):
        mealie = AsyncMock()
        mealie.get_foods.return_value = ["flour", "egg", "olive oil"]
        mealie.get_units.return_value = ["cup", "tablespoon", "teaspoon"]
        mealie.get_unit_aliases.return_value = {
            "cup": ["cup", "cups"],
            "tablespoon": ["tablespoon", "tbsp"],
            "teaspoon": ["teaspoon", "tsp"],
        }
        return mealie

    def _make_mock_model(self, responses):
        model = MagicMock()
        model.create_chat_completion = MagicMock(
            side_effect=[{"choices": [{"message": {"content": json.dumps(r)}}]} for r in responses]
        )
        return model

    @pytest.mark.asyncio
    async def test_handle_single_ingredient(self):
        from mealie_local_ai.handlers.ingredient_parsing import IngredientParsingHandler

        handler = IngredientParsingHandler()
        mealie = self._make_mock_mealie()
        model = self._make_mock_model([{"quantity": 1, "unit": "cup", "food": "flour", "note": None}])
        request = ChatCompletionRequest.model_validate(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Parse ingredient strings into components."},
                    {"role": "user", "content": '["1 cup flour"]'},
                ],
            }
        )
        with patch("mealie_local_ai.handlers.ingredient_parsing.LlamaGrammar"):
            response = await handler.handle(request, model, mealie)
        result = json.loads(response.choices[0].message.content)
        assert len(result["ingredients"]) == 1
        assert result["ingredients"][0]["food"] == "flour"
        assert result["ingredients"][0]["unit"] == "cup"

    @pytest.mark.asyncio
    async def test_handle_batch_preserves_order(self):
        from mealie_local_ai.handlers.ingredient_parsing import IngredientParsingHandler

        handler = IngredientParsingHandler()
        mealie = self._make_mock_mealie()
        model = self._make_mock_model(
            [
                {"quantity": 1, "unit": "cup", "food": "flour", "note": None},
                {"quantity": 2, "unit": None, "food": "egg", "note": None},
            ]
        )
        request = ChatCompletionRequest.model_validate(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Parse ingredient strings into components."},
                    {"role": "user", "content": '["1 cup flour", "2 eggs"]'},
                ],
            }
        )
        with patch("mealie_local_ai.handlers.ingredient_parsing.LlamaGrammar"):
            response = await handler.handle(request, model, mealie)
        result = json.loads(response.choices[0].message.content)
        assert len(result["ingredients"]) == 2
        assert result["ingredients"][0]["food"] == "flour"
        assert result["ingredients"][1]["food"] == "egg"
        assert model.create_chat_completion.call_count == 2
