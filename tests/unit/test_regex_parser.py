import pytest

from mealie_local_ai.regex_parser import RegexParser

FOODS = [
    "flour",
    "olive oil",
    "eggs",
    "chicken broth",
    "butter",
    "vanilla extract",
    "chicken breast",
    "cream cheese",
    "garlic",
    "onion",
    "potatoes",
    "soy sauce",
    "rice",
    "corn",
    "green onions",
    "lemon",
    "sugar",
    "parmesan cheese",
    "ground turkey",
    "ground chicken",
    "ground beef",
    "heavy cream",
    "chicken stock",
    "baking soda",
    "baking powder",
    "vegetable oil",
    "rice vinegar",
    "brown sugar",
    "peanut butter",
    "sour cream",
    "tomato paste",
]

UNITS = [
    "cup",
    "cups",
    "tablespoon",
    "tablespoons",
    "tbsp",
    "teaspoon",
    "teaspoons",
    "tsp",
    "lb",
    "lbs",
    "oz",
    "ounces",
    "pound",
    "pounds",
    "can",
    "clove",
    "cloves",
    "bunch",
    "pack",
    "packs",
    "slices",
    "grams",
    "pinch",
]


@pytest.fixture
def parser():
    p = RegexParser()
    p.build(FOODS, UNITS)
    return p


class TestPositiveMatches:
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

    def test_food_unit_reversed(self, parser):
        result = parser.try_parse("1 garlic clove")
        assert result == {"quantity": 1.0, "unit": "clove", "food": "garlic", "note": None}

    def test_size_adjective(self, parser):
        result = parser.try_parse("1 small onion, diced")
        assert result == {"quantity": 1.0, "unit": None, "food": "onion", "note": "diced"}

    def test_size_adjective_medium(self, parser):
        result = parser.try_parse("3 medium potatoes, peeled and cubed")
        assert result == {"quantity": 3.0, "unit": None, "food": "potatoes", "note": "peeled and cubed"}

    def test_fraction_quantity(self, parser):
        result = parser.try_parse("1/4 teaspoon baking soda")
        assert result["quantity"] == pytest.approx(0.25, abs=0.01)
        assert result["unit"] == "teaspoon"
        assert result["food"] == "baking soda"

    def test_mixed_number_quantity(self, parser):
        result = parser.try_parse("1 1/2 cups chicken broth")
        assert result["quantity"] == pytest.approx(1.5, abs=0.01)
        assert result["unit"] == "cups"
        assert result["food"] == "chicken broth"

    def test_decimal_quantity(self, parser):
        result = parser.try_parse("1.5 cups flour")
        assert result["quantity"] == pytest.approx(1.5, abs=0.01)

    def test_unicode_fraction_standalone(self, parser):
        result = parser.try_parse("¼ cup soy sauce")
        assert result["quantity"] == pytest.approx(0.25, abs=0.01)
        assert result["unit"] == "cup"
        assert result["food"] == "soy sauce"

    def test_unicode_fraction_mixed(self, parser):
        result = parser.try_parse("2½ cups chicken broth")
        assert result["quantity"] == pytest.approx(2.5, abs=0.01)
        assert result["unit"] == "cups"
        assert result["food"] == "chicken broth"

    def test_unicode_fraction_mixed_with_space(self, parser):
        result = parser.try_parse("1 ½ cups rice")
        assert result["quantity"] == pytest.approx(1.5, abs=0.01)
        assert result["unit"] == "cups"
        assert result["food"] == "rice"

    def test_no_space_between_qty_and_unit(self, parser):
        result = parser.try_parse("1lb ground chicken")
        assert result == {"quantity": 1.0, "unit": "lb", "food": "ground chicken", "note": None}

    def test_unit_abbreviation(self, parser):
        result = parser.try_parse("4 tbsp butter")
        assert result == {"quantity": 4.0, "unit": "tbsp", "food": "butter", "note": None}


class TestNegativeMatches:
    def test_adjective_before_food(self, parser):
        assert parser.try_parse("1 cup salted butter") is None

    def test_parenthetical(self, parser):
        assert parser.try_parse("1 (14-ounce) can tomatoes") is None

    def test_or_alternative(self, parser):
        assert parser.try_parse("2 tablespoons soy sauce or tamari") is None

    def test_dual_measure(self, parser):
        assert parser.try_parse("2 cups/8 ounces frozen edamame") is None

    def test_no_quantity(self, parser):
        assert parser.try_parse("egg") is None

    def test_unrecognized_food(self, parser):
        assert parser.try_parse("1 cup unicorn tears") is None

    def test_empty_string(self, parser):
        assert parser.try_parse("") is None

    def test_complex_description(self, parser):
        assert parser.try_parse("1 cup canned whole berry cranberry sauce") is None

    def test_unknown_unit_known_food(self, parser):
        assert parser.try_parse("1 sprig flour") is None

    def test_not_built(self):
        p = RegexParser()
        assert p.try_parse("1 cup flour") is None


class TestCaseInsensitivity:
    def test_uppercase_food(self, parser):
        result = parser.try_parse("1 cup Flour")
        assert result is not None
        assert result["food"] == "flour"

    def test_uppercase_unit(self, parser):
        result = parser.try_parse("1 Cup flour")
        assert result is not None
        assert result["unit"] == "cup"

    def test_all_caps(self, parser):
        result = parser.try_parse("2 TBSP BUTTER")
        assert result is not None
        assert result["food"] == "butter"
        assert result["unit"] == "tbsp"
