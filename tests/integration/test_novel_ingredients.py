"""Test the fine-tuned model against 100 randomly generated ingredient strings
that were NOT in the training data. Validates structural correctness (valid JSON,
correct keys, reasonable types) rather than exact food resolution."""

from __future__ import annotations

import json
import random

import pytest
from llama_cpp import LlamaGrammar

from mealie_llm_server.handlers.ingredient_parsing import (
    _STRUCTURE_SCHEMA,
    build_messages,
    normalize_quantity,
    null_unit_heuristic,
    resolve_unit,
)

SEED = 12345

NOVEL_FOODS = [
    "abalone",
    "acorn squash",
    "allspice",
    "apple",
    "apricot",
    "artichoke",
    "arugula",
    "apricot kernel",
    "avocado",
    "baby bok choy",
    "bacon",
    "banana",
    "barramundi",
    "bay scallop",
    "bean sprout",
    "beef liver",
    "beef steak",
    "barberry",
    "bell pepper",
    "blackberry",
    "blue cheese",
    "blueberry",
    "bok choy",
    "brazil nut",
    "bread",
    "bread crumb",
    "brisket",
    "broccoli",
    "brussels sprout",
    "butternut squash",
    "cantaloupe",
    "caraway",
    "carrot",
    "cashew",
    "catfish",
    "cauliflower",
    "celeriac",
    "celery",
    "chard",
    "cheddar cheese",
    "cherry",
    "chestnut",
    "coconut milk",
    "collard greens",
    "coriander powder",
    "cottage cheese",
    "crab",
    "cranberry",
    "cream",
    "cucumber",
    "date",
    "dijon mustard",
    "dragon fruit",
    "duck breast",
    "eggplant",
    "elderberry",
    "fennel",
    "feta cheese",
    "fig",
    "goat cheese",
    "gochujang",
    "gooseberry",
    "grape",
    "grapefruit",
    "greek yogurt",
    "habanero pepper",
    "hazelnut",
    "honeydew melon",
    "horseradish",
    "ice cream",
    "jackfruit",
    "jasmine rice",
    "jicama",
    "kale",
    "kiwi",
    "kohlrabi",
    "kumquat",
    "lamb",
    "lentil sprout",
    "lettuce",
    "lobster bisque",
    "macadamia",
    "mango",
    "melon",
    "monkfish",
    "mozzarella cheese",
    "mushroom seasoning",
    "mussel",
    "napa cabbage",
    "nectarine",
    "octopus",
    "okra",
    "orange",
    "papaya",
    "parsnip",
    "passion fruit",
    "peach",
    "pear",
    "pecan",
    "pineapple",
    "plantain",
    "plum",
    "poblano pepper",
    "pomegranate",
    "pork belly",
    "pork chop",
    "pork shoulder",
    "potato",
    "prosciutto",
    "pumpkin",
    "pumpkin seed",
    "quail egg",
    "quinoa",
    "radicchio",
    "radish",
    "raisin",
    "raspberry",
    "red cabbage",
    "red onion",
    "rhubarb",
    "ricotta cheese",
    "romaine lettuce",
    "salmon",
    "sardine",
    "scallop",
    "shallot",
    "shiitake mushroom",
    "silken tofu",
    "smoked salmon",
    "spinach",
    "squid",
    "strawberry",
    "sun dried tomato",
    "sunflower seed",
    "sweet potato",
    "swordfish",
    "tamarind",
    "tangerine",
    "thickened cream",
    "truffle cheese",
    "tuna",
    "turkey breast",
    "turnip",
    "venison",
    "walnut",
    "wasabi",
    "water chestnut",
    "watercress",
    "watermelon",
    "yam",
    "yuzu",
    "zander",
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
    "pound",
    "pounds",
    "lb",
    "lbs",
    "ounce",
    "ounces",
    "oz",
    "gram",
    "grams",
    "g",
    "kg",
    "kilogram",
    "milliliter",
    "ml",
    "liter",
    "bunch",
    "clove",
    "cloves",
    "can",
    "slice",
    "slices",
    "piece",
    "pieces",
    "pinch",
    "dash",
    "handful",
    "sprig",
    "sprigs",
    "stalk",
    "stalks",
    "head",
    "quart",
    "pint",
    "gallon",
]

QUANTITIES = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "8",
    "10",
    "12",
    "1/2",
    "1/3",
    "1/4",
    "3/4",
    "2/3",
    "1 1/2",
    "2 1/2",
    "3/8",
    "0.5",
    "0.25",
    "1.5",
    "2.5",
]

NOTES = [
    "chopped",
    "diced",
    "minced",
    "sliced",
    "thinly sliced",
    "finely chopped",
    "roughly chopped",
    "grated",
    "shredded",
    "julienned",
    "crushed",
    "ground",
    "toasted",
    "roasted",
    "melted",
    "softened",
    "room temperature",
    "peeled",
    "seeded",
    "deveined",
    "deboned",
    "skin-on",
    "bone-in",
    "fresh",
    "frozen",
    "dried",
    "canned",
    "packed in water",
    "plus more for garnish",
    "or to taste",
    "divided",
    "optional",
    "at room temperature",
    "cut into 1-inch pieces",
    "halved",
]


def _generate_random_ingredients(n: int = 100, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    ingredients = []
    for _ in range(n):
        food = rng.choice(NOVEL_FOODS)
        style = rng.randint(0, 5)

        if style == 0:
            qty = rng.choice(QUANTITIES)
            unit = rng.choice(UNITS)
            note = rng.choice(NOTES)
            text = f"{qty} {unit} {food}, {note}"
        elif style == 1:
            qty = rng.choice(QUANTITIES)
            unit = rng.choice(UNITS)
            text = f"{qty} {unit} {food}"
        elif style == 2:
            qty = rng.choice(QUANTITIES)
            text = f"{qty} {food}"
        elif style == 3:
            qty = rng.choice(QUANTITIES)
            unit = rng.choice(UNITS)
            note1 = rng.choice(NOTES)
            note2 = rng.choice(NOTES)
            text = f"{qty} {unit} {food}, {note1} and {note2}"
        elif style == 4:
            text = f"{food}, for garnish"
        else:
            qty = rng.choice(QUANTITIES)
            unit = rng.choice(UNITS)
            note = rng.choice(NOTES)
            text = f"{qty} {unit} {note} {food}"

        ingredients.append(text)
    return ingredients


NOVEL_INGREDIENTS = _generate_random_ingredients()


def _parse(ingredient_text, llm_model, food_resolver, foods, unit_aliases):
    messages = build_messages(ingredient_text)
    grammar = LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))
    response = llm_model.create_chat_completion(
        messages=messages,
        grammar=grammar,
        temperature=0,
        max_tokens=-1,
    )
    raw = json.loads(response["choices"][0]["message"]["content"])

    heuristic = null_unit_heuristic(
        ingredient_text,
        raw.get("unit"),
        raw.get("food"),
        unit_aliases,
    )
    raw["unit"] = resolve_unit(heuristic["unit"], unit_aliases)
    raw["food"] = heuristic["food"]
    raw["quantity"] = normalize_quantity(raw.get("quantity"))

    if food_resolver and foods and raw["food"]:
        raw["food"] = food_resolver.match(raw["food"], foods)

    return raw


@pytest.mark.parametrize("ingredient", NOVEL_INGREDIENTS, ids=NOVEL_INGREDIENTS)
def test_novel_ingredient_structure(
    ingredient,
    llm_model,
    food_resolver,
    foods,
    unit_aliases,
):
    result = _parse(ingredient, llm_model, food_resolver, foods, unit_aliases)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert set(result.keys()) == {"quantity", "unit", "food", "note"}, f"Wrong keys: {set(result.keys())}"

    if result["quantity"] is not None:
        assert isinstance(result["quantity"], (int, float)), (
            f"quantity should be numeric, got {type(result['quantity'])}"
        )
        assert result["quantity"] > 0, f"quantity should be positive, got {result['quantity']}"

    assert result["unit"] is None or isinstance(result["unit"], str), (
        f"unit should be str or None, got {type(result['unit'])}"
    )
    assert result["food"] is None or isinstance(result["food"], str), (
        f"food should be str or None, got {type(result['food'])}"
    )
    assert result["food"], f"food should not be empty for '{ingredient}'"
    note = result.get("note")
    assert note is None or isinstance(note, str), "note should be str or None"
