from __future__ import annotations

import json

import pytest
from llama_cpp import LlamaGrammar

from mealie_llm_server.handlers.ingredient_parsing import (
    _STRUCTURE_SCHEMA,
    build_messages,
    normalize_quantity,
    null_unit_heuristic,
    resolve_unit,
)


def parse_ingredient(
    ingredient_text: str,
    llm_model,
    model_id: str,
    food_matcher,
    foods: list[str],
    unit_aliases: dict[str, list[str]],
) -> dict:
    messages = build_messages(ingredient_text, model_id)
    grammar = LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))
    response = llm_model.create_chat_completion(
        messages=messages,
        grammar=grammar,
        temperature=0,
        max_tokens=-1,
    )
    raw = json.loads(response["choices"][0]["message"]["content"])

    heuristic = null_unit_heuristic(ingredient_text, raw.get("unit"), raw.get("food"), unit_aliases)
    raw["unit"] = resolve_unit(heuristic["unit"], unit_aliases)
    raw["food"] = heuristic["food"]
    raw["quantity"] = normalize_quantity(raw.get("quantity"))

    if food_matcher and foods and raw["food"]:
        raw["food"] = food_matcher.match(raw["food"], foods)

    return raw


BASIC_CASES = [
    ("1 cup flour", 1.0, "cup", {"flour", "unbleached all-purpose flour"}, None),
    ("2 tablespoons olive oil", 2.0, "tablespoon", {"olive oil", "extra virgin olive oil"}, None),
    ("3 eggs", 3.0, None, {"egg"}, None),
    ("1 1/2 cups chicken broth", 1.5, "cup", {"chicken broth", "chicken stock"}, None),
]

ABBREVIATION_CASES = [
    ("4 tbsp butter", 4.0, "tablespoon", {"butter"}, None),
    ("2 tsp vanilla extract", 2.0, "teaspoon", {"vanilla extract"}, None),
    ("1 lb chicken breast", 1.0, "pound", {"chicken breast"}, None),
    ("8 oz cream cheese", 8.0, "ounce", {"cream cheese"}, None),
]

NOTES_CASES = [
    ("8 oz cream cheese, softened", 8.0, "ounce", {"cream cheese"}, "softened"),
    ("2 cloves garlic, minced", 2.0, "clove", {"garlic"}, "minced"),
    ("1 can diced tomatoes", 1.0, "can", {"tomato", "tomatoes", "diced tomato", "diced tomatoes"}, "diced"),
    ("1 small onion, diced", 1.0, None, {"onion"}, "diced"),
]

NOTES_SUBSTRING_CASES = [
    ("3 medium potatoes, peeled and cubed", 3.0, None, {"potato", "potatoes"}, "peeled"),
]

CHOWDOWN_CASES = [
    ("1/2 cup salted butter, softened", 0.5, "cup", {"butter"}, "softened"),
    ("1 cup canned whole berry cranberry sauce", 1.0, "cup", {"cranberry sauce"}, None),
    ("1 cup arborio rice", 1.0, "cup", {"rice", "arborio rice"}, None),
    ("2 cups chicken stock", 2.0, "cup", {"chicken stock"}, None),
    ("1lb ground chicken", 1.0, "pound", {"ground chicken"}, None),
]

CHOWDOWN_LEMON_ZEST = [
    ("1 tablespoon fresh lemon zest (about 1 lemon)", 1.0, "tablespoon", {"lemon", "lemon zest"}, "zest"),
]

NO_MATCH_CASES = [
    ("1 tbsp gochujang", 1.0, "tablespoon", None),
]

STRETCH_CASES = [
    ("1 cup chickpea cooking liquid", 1.0, "cup", {"aquafaba"}, None),
    ("1 bunch green onions, sliced", 1.0, "bunch", {"scallion", "scallions"}, "sliced"),
    ("1 can corn", 1.0, "can", {"sweet corn", "canned corn", "corn"}, None),
]


@pytest.mark.parametrize(
    "ingredient, exp_qty, exp_unit, exp_foods, exp_note",
    BASIC_CASES + ABBREVIATION_CASES + NOTES_CASES + CHOWDOWN_CASES,
    ids=[c[0] for c in BASIC_CASES + ABBREVIATION_CASES + NOTES_CASES + CHOWDOWN_CASES],
)
def test_exact_note(ingredient, exp_qty, exp_unit, exp_foods, exp_note, llm_model, model_id, food_matcher, foods, unit_aliases):
    result = parse_ingredient(ingredient, llm_model, model_id, food_matcher, foods, unit_aliases)
    assert result["quantity"] == exp_qty, f"quantity: {result['quantity']} != {exp_qty}"
    assert result["unit"] == exp_unit, f"unit: {result['unit']} != {exp_unit}"
    if exp_foods is not None:
        assert result["food"] in exp_foods, f"food: {result['food']} not in {exp_foods}"
    else:
        assert result["food"] is None, f"food: expected None, got {result['food']}"
    if exp_note is not None:
        assert result["note"] is not None, f"note: expected '{exp_note}', got None"
        assert result["note"].lower() == exp_note.lower(), f"note: {result['note']} != {exp_note}"
    else:
        if result["note"] is not None:
            assert result["note"].strip() == "", f"note: expected None/empty, got {result['note']}"


@pytest.mark.parametrize(
    "ingredient, exp_qty, exp_unit, exp_foods, note_substr",
    NOTES_SUBSTRING_CASES + CHOWDOWN_LEMON_ZEST,
    ids=[c[0] for c in NOTES_SUBSTRING_CASES + CHOWDOWN_LEMON_ZEST],
)
def test_note_contains(ingredient, exp_qty, exp_unit, exp_foods, note_substr, llm_model, model_id, food_matcher, foods, unit_aliases):
    result = parse_ingredient(ingredient, llm_model, model_id, food_matcher, foods, unit_aliases)
    assert result["quantity"] == exp_qty, f"quantity: {result['quantity']} != {exp_qty}"
    assert result["unit"] == exp_unit, f"unit: {result['unit']} != {exp_unit}"
    assert result["food"] in exp_foods, f"food: {result['food']} not in {exp_foods}"
    food_str = result["food"] or ""
    note_str = result["note"] or ""
    assert note_substr.lower() in note_str.lower() or note_substr.lower() in food_str.lower(), \
        f"'{note_substr}' not found in note={result['note']!r} or food={result['food']!r}"


@pytest.mark.parametrize(
    "ingredient, exp_qty, exp_unit, exp_food_none",
    NO_MATCH_CASES,
    ids=[c[0] for c in NO_MATCH_CASES],
)
def test_no_food_match(ingredient, exp_qty, exp_unit, exp_food_none, llm_model, model_id, food_matcher, foods, unit_aliases):
    result = parse_ingredient(ingredient, llm_model, model_id, food_matcher, foods, unit_aliases)
    assert result["quantity"] == exp_qty, f"quantity: {result['quantity']} != {exp_qty}"
    assert result["unit"] == exp_unit, f"unit: {result['unit']} != {exp_unit}"
    assert result["food"] is None, f"food: expected None (no match), got {result['food']}"


@pytest.mark.parametrize(
    "ingredient, exp_qty, exp_unit, exp_foods, exp_note",
    STRETCH_CASES,
    ids=[c[0] for c in STRETCH_CASES],
)
def test_stretch_synonym_resolution(ingredient, exp_qty, exp_unit, exp_foods, exp_note, llm_model, model_id, food_matcher, foods, unit_aliases):
    result = parse_ingredient(ingredient, llm_model, model_id, food_matcher, foods, unit_aliases)
    assert result["quantity"] == exp_qty, f"quantity: {result['quantity']} != {exp_qty}"
    assert result["unit"] == exp_unit, f"unit: {result['unit']} != {exp_unit}"
    assert result["food"] in exp_foods, f"food: {result['food']} not in {exp_foods}"
    if exp_note is not None:
        assert result["note"] is not None, f"note: expected '{exp_note}', got None"
        assert exp_note.lower() in result["note"].lower(), f"note: '{exp_note}' not in {result['note']}"
