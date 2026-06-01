"""Component tests for LLM extraction with curated test cases.

Validates extraction output structure and correctness using a real LLM model.
No Mealie dependency — tests raw extraction only.
"""

from __future__ import annotations

import pytest

from mealie_local_ai.handlers.ingredient_parsing import (
    extract_raw,
    normalize_quantity,
)


SIMPLE_CASES = [
    ("1 cup flour", 1.0, "cup", "flour"),
    ("2 tablespoons olive oil", 2.0, "tablespoon", "olive oil"),
    ("3 eggs", 3.0, None, "egg"),
    ("1 teaspoon salt", 1.0, "teaspoon", "salt"),
    ("4 tbsp butter", 4.0, "tablespoon", "butter"),
    ("1 lb ground beef", 1.0, "pound", "ground beef"),
]

FRACTION_CASES = [
    ("1/2 cup sugar", 0.5, "cup", "sugar"),
    ("1/4 teaspoon baking soda", 0.25, "teaspoon", "baking soda"),
    ("3/4 cup milk", 0.75, "cup", "milk"),
]

MIXED_NUMBER_CASES = [
    ("1 1/2 cups chicken broth", 1.5, "cup", "chicken broth"),
    ("2 1/2 tablespoons soy sauce", 2.5, "tablespoon", "soy sauce"),
]

NO_UNIT_CASES = [
    ("3 eggs", 3.0, "egg"),
    ("2 bananas", 2.0, "banana"),
    ("1 onion", 1.0, "onion"),
]


@pytest.mark.parametrize(
    "text, exp_qty, exp_unit, exp_food",
    SIMPLE_CASES,
    ids=[c[0] for c in SIMPLE_CASES],
)
def test_simple_extraction(text, exp_qty, exp_unit, exp_food, llm_model, grammar):
    raw = extract_raw(text, llm_model, grammar)
    qty = normalize_quantity(raw.get("quantity"))
    assert qty == pytest.approx(exp_qty, abs=0.01)
    assert raw.get("food"), f"food should not be empty for '{text}'"


@pytest.mark.parametrize(
    "text, exp_qty, exp_unit, exp_food",
    FRACTION_CASES,
    ids=[c[0] for c in FRACTION_CASES],
)
def test_fraction_extraction(text, exp_qty, exp_unit, exp_food, llm_model, grammar):
    raw = extract_raw(text, llm_model, grammar)
    qty = normalize_quantity(raw.get("quantity"))
    assert qty == pytest.approx(exp_qty, abs=0.01)
    assert raw.get("food"), f"food should not be empty for '{text}'"


@pytest.mark.parametrize(
    "text, exp_qty, exp_unit, exp_food",
    MIXED_NUMBER_CASES,
    ids=[c[0] for c in MIXED_NUMBER_CASES],
)
def test_mixed_number_extraction(text, exp_qty, exp_unit, exp_food, llm_model, grammar):
    raw = extract_raw(text, llm_model, grammar)
    qty = normalize_quantity(raw.get("quantity"))
    assert qty == pytest.approx(exp_qty, abs=0.01)
    assert raw.get("food"), f"food should not be empty for '{text}'"


WITH_NOTE_CASES = [
    ("2 cloves garlic, minced", "garlic", "minced"),
    ("1 onion, diced", "onion", "diced"),
    ("1 cup fresh basil, chopped", "basil", "chopped"),
]


@pytest.mark.parametrize(
    "text, exp_food, exp_note",
    WITH_NOTE_CASES,
    ids=[c[0] for c in WITH_NOTE_CASES],
)
def test_extraction_with_note(text, exp_food, exp_note, llm_model, grammar):
    raw = extract_raw(text, llm_model, grammar)
    assert raw.get("food"), f"food should not be empty for '{text}'"
    note = raw.get("note") or ""
    assert note, f"note should not be empty for '{text}'"


def test_no_quantity_ingredient(llm_model, grammar):
    raw = extract_raw("salt and pepper to taste", llm_model, grammar)
    assert raw.get("food"), "food should not be empty"


def test_multiword_food(llm_model, grammar):
    raw = extract_raw("1 tablespoon tomato paste", llm_model, grammar)
    food = raw.get("food") or ""
    assert "tomato" in food.lower()


def test_parenthetical(llm_model, grammar):
    raw = extract_raw("1 (14-ounce) can diced tomatoes", llm_model, grammar)
    assert raw.get("food"), "food should not be empty"


def test_output_structure(llm_model, grammar):
    raw = extract_raw("1 cup flour", llm_model, grammar)
    assert isinstance(raw, dict)
    assert "quantity" in raw
    assert "unit" in raw
    assert "food" in raw
    assert "note" in raw
