"""End-to-end ingredient parsing tests against a live Mealie instance.

Curated test cases covering extraction -> unit resolution -> food resolution.
"""

from __future__ import annotations

import re

import pytest

from mealie_local_ai.handlers.ingredient_parsing import (
    parse_single_ingredient,
    resolve_unit,
)

_XFAIL_REGISTRY: dict[str, str] = {
    "1/4 teaspoon salt": "resolver maps 'salt' to 'curing salt' (no exact match in DB)",
    "1 cup chickpea cooking liquid": "LLM extracts food as 'chickpea' instead of 'chickpea cooking liquid'",
    "1 bunch green onions, sliced": "resolver doesn't map 'green onions' to 'scallion'",
    "1 can corn": "resolver maps 'corn' to 'corn oil' instead of 'sweet corn'",
    "1 cup long-grain white rice (basmati or jasmine)": "resolver maps to 'brown long grain rice' instead of 'basmati rice'",
    "¼ cup dry white wine": "LLM drops 'dry' qualifier from note",
    "½ to ⅔ cup sugar": "LLM captures range remainder but drops context ('up to ⅔ cup')",
    "1.5 to 2 pounds zucchini (about 3 to 4 medium)": "LLM uses first value (1.5) not midpoint (1.75) for range qty",
    "2 heaped cups cherry tomatoes, about 0.75 pound": "LLM note format 'about ¾ pound' doesn't match expected '(¾ lb)'",
    "1 large garlic clove": "LLM puts 'clove' in note instead of unit field",
    "1 pound thin spaghetti": "LLM drops 'thin' qualifier from note",
    "1 packed cup fresh basil leaves, plus more for garnish if desired": "LLM truncates note to 'fresh', drops garnish info",
    "Coarsely chopped cilantro leaves and tender stems, for serving": "LLM hallucates qty=1 and truncates note",
}

_FOOD_ALIASES: dict[str, set[str]] = {
    "1 tablespoon whole black peppercorns": {"peppercorn", "peppercorns"},
}

CURATED_CASES = [
    ("1 cup flour", 1.0, "cup", "flour", ""),
    ("2 tablespoons olive oil", 2.0, "tablespoon", "olive oil", ""),
    ("3 eggs", 3.0, None, "eggs", ""),
    ("1/4 teaspoon salt", 0.25, "teaspoon", "salt", ""),
    ("1 1/2 cups chicken broth", 1.5, "cup", "chicken broth", ""),
    ("2 cloves garlic, minced", 2.0, "clove", "garlic", "minced"),
    ("1 tablespoon tomato paste", 1.0, "tablespoon", "tomato paste", ""),
    ("1 lb ground beef", 1.0, "pound", "ground beef", ""),
    ("1/2 cup soy sauce", 0.5, "cup", "soy sauce", ""),
    ("1 tablespoon whole black peppercorns", 1.0, "tablespoon", "peppercorn", ""),
    ("1 cup chickpea cooking liquid", 1.0, "cup", "aquafaba", ""),
    ("1 bunch green onions, sliced", 1.0, "bunch", "scallion", "sliced"),
    ("1 can corn", 1.0, "can", "sweet corn", ""),
    ("2 tablespoons rice vinegar", 2.0, "tablespoon", "rice wine vinegar", ""),
    ("1 cup long-grain white rice (basmati or jasmine)", 1.0, "cup", "basmati rice", "(basmati or jasmine)"),
    # Novel-style structural checks
    ("3 pounds fresh broccoli", 3.0, "pound", None, ""),
    ("1 kilogram pineapple, fresh", 1.0, "kilogram", None, "fresh"),
    ("2 cups arugula, at room temperature", 2.0, "cup", None, ""),
    ("6 ounces walnut, frozen", 6.0, "ounce", None, "frozen"),
    ("1 pinch abalone", 1.0, "pinch", None, ""),
    # CSV training failures — fraction inputs
    ("¼ cup dry white wine", 0.25, "cup", "white wine", "dry"),
    ("½ lemon, squeezed as needed (optional)", 0.5, None, "lemon", "optional, squeezed as needed"),
    ("1½ lemons, squeezed as needed (optional)", 1.5, None, "lemons", "optional, squeezed as needed"),
    ("½ to ⅔ cup sugar", 0.5, "cup", "sugar", "up to ⅔ cup"),
    # CSV training failures — decimal inputs
    ("0.333 cup unsweetened smooth, natural peanut butter", 0.333, "cup", "peanut butter", "smooth, unsweetened"),
    ("1.5 to 2 pounds zucchini (about 3 to 4 medium)", 1.75, "pound", "zucchini", "(3-4 medium zucchini)"),
    ("2 heaped cups cherry tomatoes, about 0.75 pound", 2.0, "cup", "cherry tomatoes", "(¾ lb)"),
    (
        "1 (15-ounce) can chickpeas or 1.5 cups cooked chickpeas, rinsed and patted dry",
        1.0,
        "can",
        "chickpeas",
        "(15 oz / 1½ cups) drained, rinsed, patted dry",
    ),
    # CSV training failures — integer/other inputs
    ("2 large eggs", 2.0, None, "eggs", ""),
    ("1 large garlic clove", 1.0, "clove", "garlic", ""),
    ("2 garlic cloves, minced or puréed", 2.0, "clove", "garlic", "minced or puréed"),
    ("3 medium potatoes, peeled and cubed", 3.0, None, "potatoes", "peeled and cubed"),
    (
        "2 medium leeks, trimmed, halved lengthwise, thinly sliced crosswise then rinsed",
        2.0,
        None,
        "leeks",
        "trimmed, halved lengthwise, thinly sliced crosswise then rinsed",
    ),
    ("1 tablespoon chopped flat-leaf parsley", 1.0, "tablespoon", "parsley", "chopped, flat-leaf"),
    ("1 pound thin spaghetti", 1.0, "pound", "spaghetti", "thin preferred"),
    ("1 cup canned whole berry cranberry sauce", 1.0, "cup", "cranberry sauce", "canned, whole berry"),
    ("2 cups/8 ounces frozen shelled edamame", 2.0, "cup", "edamame", "(8 oz) shelled, frozen"),
    (
        "1 packed cup fresh basil leaves, plus more for garnish if desired",
        1.0,
        "cup",
        "basil",
        "fresh, plus more for garnish",
    ),
    (
        "Coarsely chopped cilantro leaves and tender stems, for serving",
        None,
        None,
        "cilantro",
        "leaves & stems coarsely chopped, for serving",
    ),
]


def _build_params():
    params = []
    for entry in CURATED_CASES:
        ingredient_text = entry[0]
        marks = ()
        if ingredient_text in _XFAIL_REGISTRY:
            marks = (pytest.mark.xfail(reason=_XFAIL_REGISTRY[ingredient_text]),)
        params.append(pytest.param(*entry, id=ingredient_text, marks=marks))
    return params


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        s = re.sub(r"[^\w\s]", "", s.lower())
        s = f" {s} "
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    a_ng, b_ng = ngrams(a), ngrams(b)
    if not a_ng or not b_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng | b_ng)


@pytest.mark.parametrize("ingredient, exp_qty, exp_raw_unit, exp_food, exp_note", _build_params())
def test_ingredient_parsing(
    ingredient,
    exp_qty,
    exp_raw_unit,
    exp_food,
    exp_note,
    llm_model,
    food_resolver,
    foods,
    unit_aliases,
    grammar,
    regex_parser,
    results_collector,
):
    result, trace = parse_single_ingredient(
        ingredient,
        llm_model,
        grammar,
        unit_aliases,
        food_resolver=food_resolver,
        foods=foods,
        regex_parser=regex_parser,
    )

    exp_unit = resolve_unit(exp_raw_unit, unit_aliases)
    actual_note = result.get("note") or ""

    mismatches = []

    if exp_qty is not None:
        if result["quantity"] is None or abs(result["quantity"] - exp_qty) > 0.01:
            mismatches.append({"field": "quantity", "expected": exp_qty, "actual": result["quantity"]})
    elif result["quantity"] is not None:
        mismatches.append({"field": "quantity", "expected": None, "actual": result["quantity"]})

    if result["unit"] != exp_unit:
        mismatches.append({"field": "unit", "expected": exp_unit, "actual": result["unit"]})

    if exp_food is not None:
        accepted_foods = _FOOD_ALIASES.get(ingredient, {exp_food})
        if result["food"] not in accepted_foods:
            mismatches.append({"field": "food", "expected": exp_food, "actual": result["food"]})

    if exp_note and not actual_note:
        mismatches.append({"field": "note", "expected": exp_note, "actual": actual_note})
    elif exp_note and actual_note and _ngram_jaccard(actual_note, exp_note) < 0.3:
        mismatches.append({"field": "note", "expected": exp_note, "actual": actual_note})

    results_collector.append(
        {
            "input": ingredient,
            "actual": {
                "quantity": result.get("quantity"),
                "unit": result.get("unit"),
                "food": result.get("food"),
                "note": actual_note,
            },
            "expected": {"quantity": exp_qty, "unit": exp_unit, "food": exp_food, "note": exp_note},
            "status": "pass" if not mismatches else "fail",
            "mismatches": mismatches,
            "trace": trace,
        }
    )

    if mismatches:
        details = "; ".join(f"{m['field']}: {m['actual']!r} != {m['expected']!r}" for m in mismatches)
        pytest.fail(f"Mismatches: {details}")


def test_structural_correctness(llm_model, food_resolver, foods, unit_aliases, grammar, regex_parser):
    """Verify result structure for a novel ingredient."""
    result, _trace = parse_single_ingredient(
        "2 cups diced mango",
        llm_model,
        grammar,
        unit_aliases,
        food_resolver=food_resolver,
        foods=foods,
        regex_parser=regex_parser,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"quantity", "unit", "food", "note"}
    if result["quantity"] is not None:
        assert isinstance(result["quantity"], (int, float))
        assert result["quantity"] > 0
    assert result["food"], "food should not be empty"
