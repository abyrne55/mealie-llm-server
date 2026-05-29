"""End-to-end ingredient parsing tests against a live Mealie instance.

Curated test cases covering extraction -> unit resolution -> food resolution.
"""

from __future__ import annotations

import json
import re

import pytest
from llama_cpp import LlamaGrammar

from mealie_local_ai.handlers.ingredient_parsing import (
    _STRUCTURE_SCHEMA,
    build_messages,
    normalize_quantity,
    null_unit_heuristic,
    resolve_unit,
)

_XFAIL_REGISTRY: dict[str, str] = {
    "1 cup chickpea cooking liquid": "resolver doesn't map 'chickpea cooking liquid' to 'aquafaba'",
    "1 bunch green onions, sliced": "resolver doesn't map 'green onions' to 'scallion'",
    "1 can corn": "resolver maps 'corn' to wrong entry instead of 'sweet corn'",
    "1 cup long-grain white rice (basmati or jasmine)": "resolver maps to 'brown long grain rice' instead of 'basmati rice'",
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
    ("2 tablespoons rice vinegar", 2.0, "tablespoon", "rice vinegar", ""),
    ("1 cup long-grain white rice (basmati or jasmine)", 1.0, "cup", "basmati rice", "(basmati or jasmine)"),
    # Novel-style structural checks
    ("3 pounds fresh broccoli", 3.0, "pound", None, ""),
    ("1 kilogram pineapple, fresh", 1.0, "kilogram", None, "fresh"),
    ("2 cups arugula, at room temperature", 2.0, "cup", None, ""),
    ("6 ounces walnut, frozen", 6.0, "ounce", None, "frozen"),
    ("1 pinch abalone", 1.0, "pinch", None, ""),
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


def parse_ingredient(ingredient_text, llm_model, food_resolver, foods, unit_aliases):
    messages = build_messages(ingredient_text)
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

    if food_resolver and foods and raw["food"]:
        resolved_food, _, _ = food_resolver.match(raw["food"], foods)
        if resolved_food is not None:
            raw["food"] = resolved_food

    return raw


@pytest.mark.parametrize("ingredient, exp_qty, exp_raw_unit, exp_food, exp_note", _build_params())
def test_ingredient_parsing(
    ingredient, exp_qty, exp_raw_unit, exp_food, exp_note, llm_model, food_resolver, foods, unit_aliases
):
    result = parse_ingredient(ingredient, llm_model, food_resolver, foods, unit_aliases)

    if exp_qty is not None:
        assert result["quantity"] == pytest.approx(exp_qty, abs=0.01), f"quantity: {result['quantity']} != {exp_qty}"
    else:
        assert result["quantity"] is None, f"quantity: expected None, got {result['quantity']}"

    exp_unit = resolve_unit(exp_raw_unit, unit_aliases)
    assert result["unit"] == exp_unit, f"unit: {result['unit']} != {exp_unit}"

    if exp_food is not None:
        accepted_foods = _FOOD_ALIASES.get(ingredient, {exp_food})
        assert result["food"] in accepted_foods, f"food: {result['food']} not in {accepted_foods}"

    actual_note = result.get("note") or ""
    if not exp_note:
        pass
    elif not actual_note:
        pytest.fail(f"note: expected '{exp_note}', got empty")
    else:
        sim = _ngram_jaccard(actual_note, exp_note)
        assert sim >= 0.3, f"note: '{actual_note}' too different from '{exp_note}' (jaccard={sim:.2f})"


def test_structural_correctness(llm_model, food_resolver, foods, unit_aliases):
    """Verify result structure for a novel ingredient."""
    result = parse_ingredient("2 cups diced mango", llm_model, food_resolver, foods, unit_aliases)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"quantity", "unit", "food", "note"}
    if result["quantity"] is not None:
        assert isinstance(result["quantity"], (int, float))
        assert result["quantity"] > 0
    assert result["food"], "food should not be empty"
