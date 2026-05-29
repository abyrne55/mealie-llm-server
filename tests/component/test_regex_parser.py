"""Component tests for RegexParser with a realistic vocabulary.

Tests the parser as a complete component with a curated food/unit vocabulary,
validating end-to-end parsing correctness (qty + unit + food together).
"""

import pytest

from mealie_local_ai.regex_parser import RegexParser

FOODS = [
    "all-purpose flour",
    "baking powder",
    "baking soda",
    "brown sugar",
    "butter",
    "chicken breast",
    "chicken broth",
    "cream cheese",
    "eggs",
    "flour",
    "garlic",
    "green onions",
    "ground beef",
    "ground chicken",
    "heavy cream",
    "lemon",
    "olive oil",
    "onion",
    "parmesan cheese",
    "peanut butter",
    "potatoes",
    "rice",
    "rice vinegar",
    "sour cream",
    "soy sauce",
    "sugar",
    "tomato paste",
    "vanilla extract",
    "vegetable oil",
]

UNITS = [
    "bunch",
    "can",
    "clove",
    "cloves",
    "cup",
    "cups",
    "grams",
    "lb",
    "lbs",
    "ounces",
    "oz",
    "pack",
    "packs",
    "pinch",
    "pound",
    "pounds",
    "slices",
    "tablespoon",
    "tablespoons",
    "tbsp",
    "teaspoon",
    "teaspoons",
    "tsp",
]


@pytest.fixture(scope="module")
def parser():
    p = RegexParser()
    p.build(FOODS, UNITS)
    return p


class TestSimpleIngredients:
    def test_qty_unit_food(self, parser):
        result = parser.try_parse("1 cup flour")
        assert result == {"quantity": 1.0, "unit": "cup", "food": "flour", "note": None}

    def test_qty_unit_food_multiword(self, parser):
        result = parser.try_parse("2 tablespoons olive oil")
        assert result == {"quantity": 2.0, "unit": "tablespoons", "food": "olive oil", "note": None}

    def test_qty_food_no_unit(self, parser):
        result = parser.try_parse("3 eggs")
        assert result == {"quantity": 3.0, "unit": None, "food": "eggs", "note": None}

    def test_qty_unit_food_with_note(self, parser):
        result = parser.try_parse("2 cloves garlic, minced")
        assert result == {"quantity": 2.0, "unit": "cloves", "food": "garlic", "note": "minced"}

    def test_unit_abbreviation(self, parser):
        result = parser.try_parse("4 tbsp butter")
        assert result == {"quantity": 4.0, "unit": "tbsp", "food": "butter", "note": None}


class TestComplexQuantities:
    def test_fraction(self, parser):
        result = parser.try_parse("1/4 teaspoon baking soda")
        assert result["quantity"] == pytest.approx(0.25, abs=0.01)
        assert result["food"] == "baking soda"

    def test_mixed_number(self, parser):
        result = parser.try_parse("1 1/2 cups chicken broth")
        assert result["quantity"] == pytest.approx(1.5, abs=0.01)
        assert result["food"] == "chicken broth"

    def test_decimal(self, parser):
        result = parser.try_parse("1.5 cups flour")
        assert result["quantity"] == pytest.approx(1.5, abs=0.01)

    def test_unicode_fraction(self, parser):
        result = parser.try_parse("¼ cup soy sauce")
        assert result["quantity"] == pytest.approx(0.25, abs=0.01)
        assert result["food"] == "soy sauce"

    def test_unicode_mixed(self, parser):
        result = parser.try_parse("2½ cups chicken broth")
        assert result["quantity"] == pytest.approx(2.5, abs=0.01)
        assert result["food"] == "chicken broth"


class TestEdgeCases:
    def test_food_unit_reversed(self, parser):
        result = parser.try_parse("1 garlic clove")
        assert result == {"quantity": 1.0, "unit": "clove", "food": "garlic", "note": None}

    def test_size_adjective(self, parser):
        result = parser.try_parse("1 small onion, diced")
        assert result == {"quantity": 1.0, "unit": None, "food": "onion", "note": "diced"}

    def test_no_space_qty_unit(self, parser):
        result = parser.try_parse("1lb ground chicken")
        assert result == {"quantity": 1.0, "unit": "lb", "food": "ground chicken", "note": None}

    def test_case_insensitive(self, parser):
        result = parser.try_parse("2 TBSP BUTTER")
        assert result is not None
        assert result["food"] == "butter"

    def test_empty_string(self, parser):
        assert parser.try_parse("") is None

    def test_no_quantity(self, parser):
        assert parser.try_parse("egg") is None

    def test_unrecognized_food(self, parser):
        assert parser.try_parse("1 cup unicorn tears") is None

    def test_not_built(self):
        p = RegexParser()
        assert p.try_parse("1 cup flour") is None
