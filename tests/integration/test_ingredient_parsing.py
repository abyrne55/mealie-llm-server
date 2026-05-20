from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from llama_cpp import LlamaGrammar

from mealie_llm_server.handlers.ingredient_parsing import (
    _STRUCTURE_SCHEMA,
    build_messages,
    normalize_quantity,
    null_unit_heuristic,
    resolve_unit,
)

_JSONL_PATH = Path(__file__).parent / "ingredients.jsonl"

_TEXT_RE = re.compile(r"### Text:\n(.+?)\n\n<\|output\|>", re.DOTALL)

_XFAIL_REGISTRY: dict[str, str] = {
    "1 cup canned whole berry cranberry sauce": "model inconsistently extracts 'canned' as note",
    "1 cup chickpea cooking liquid": "embedding matcher doesn't map 'chickpea cooking liquid' to 'aquafaba'",
    "1 bunch green onions, sliced": "embedding matcher doesn't map 'onions' to 'scallion'",
    "1 can corn": "embedding matcher maps 'corn' to 'corn oil' instead of 'sweet corn'",
}


def _load_jsonl() -> list[tuple[str, float | None, str, str, str]]:
    entries = []
    for line in _JSONL_PATH.read_text().splitlines():
        row = json.loads(line)
        user_content = row["messages"][0]["content"]
        m = _TEXT_RE.search(user_content)
        if not m:
            continue
        ingredient_text = m.group(1).strip()
        expected = json.loads(row["messages"][1]["content"])
        exp_qty = normalize_quantity(expected.get("quantity"))
        exp_raw_unit = expected.get("unit", "")
        exp_food = expected.get("food", "")
        exp_note = expected.get("note", "")
        entries.append((ingredient_text, exp_qty, exp_raw_unit, exp_food, exp_note))
    return entries


def _build_params():
    params = []
    for entry in _load_jsonl():
        ingredient_text = entry[0]
        marks = ()
        if ingredient_text in _XFAIL_REGISTRY:
            marks = (pytest.mark.xfail(reason=_XFAIL_REGISTRY[ingredient_text]),)
        params.append(pytest.param(*entry, id=ingredient_text, marks=marks))
    return params


JSONL_CASES = _build_params()


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        s = re.sub(r"[^\w\s]", "", s.lower())
        s = f" {s} "
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    a_ng, b_ng = ngrams(a), ngrams(b)
    if not a_ng or not b_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng | b_ng)


def parse_ingredient(
    ingredient_text: str,
    llm_model,
    food_resolver,
    foods: list[str],
    unit_aliases: dict[str, list[str]],
) -> dict:
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
        raw["food"] = food_resolver.match(raw["food"], foods)

    return raw


@pytest.mark.parametrize("ingredient, exp_qty, exp_raw_unit, exp_food, exp_note", JSONL_CASES)
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

    assert result["food"] == exp_food, f"food: {result['food']} != {exp_food}"

    actual_note = result.get("note") or ""
    if not exp_note:
        pass
    elif not actual_note:
        pytest.fail(f"note: expected '{exp_note}', got empty")
    else:
        sim = _ngram_jaccard(actual_note, exp_note)
        assert sim >= 0.3, f"note: '{actual_note}' too different from '{exp_note}' (jaccard={sim:.2f})"
