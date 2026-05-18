#!/usr/bin/env python3
"""Benchmark GBNF-constrained vs free-form ingredient parsing with NuExtract-2.0-2B."""

import json
import os
import sys
import time

import httpx
import numpy as np
from llama_cpp import Llama, LlamaGrammar, LogitsProcessorList

MEALIE_URL = os.environ["MEALIE_URL"]
MEALIE_API_KEY = os.environ["MEALIE_API_KEY"]
MODEL_REPO = "numind/NuExtract-2.0-2B-GGUF"
MODEL_FILE = "*Q6_K.gguf"
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "..", ".tmp", "models")

NUEXTRACT_TEMPLATE = """\
{
    "quantity": "number",
    "unit": "string",
    "food": "string",
    "note": "verbatim-string"
}"""

TEST_CASES = [
    {"input": "1 cup flour", "food": "flour", "unit": "cup", "qty": 1.0},
    {"input": "2 tablespoons olive oil", "food": "olive oil", "unit": "tablespoon", "qty": 2.0},
    {"input": "3 eggs", "food": "egg", "unit": None, "qty": 3.0},
    {"input": "1/2 teaspoon salt", "food": "salt", "unit": "teaspoon", "qty": 0.5},
    {"input": "1 lb boneless skinless chicken breast", "food": "chicken breast", "unit": "pound", "qty": 1.0},
    {"input": "2 cloves garlic, minced", "food": "garlic", "unit": "clove", "qty": 2.0},
    {"input": "1 can (14 oz) diced tomatoes", "food": "diced tomatoes", "unit": "can", "qty": 1.0},
    {"input": "freshly ground black pepper to taste", "food": "black pepper", "unit": None, "qty": None},
    {"input": "1 1/2 cups chicken broth", "food": "chicken broth", "unit": "cup", "qty": 1.5},
    {"input": "3 medium potatoes, peeled and cubed", "food": "potato", "unit": None, "qty": 3.0},
]


def fetch_mealie_data():
    """Fetch foods and units from Mealie."""
    headers = {"Authorization": f"Bearer {MEALIE_API_KEY}"}
    client = httpx.Client(headers=headers, timeout=30)

    foods_resp = client.get(f"{MEALIE_URL}/api/foods", params={"perPage": "-1"})
    foods_resp.raise_for_status()
    units_resp = client.get(f"{MEALIE_URL}/api/units", params={"perPage": "-1"})
    units_resp.raise_for_status()

    foods = []
    seen = set()
    for item in foods_resp.json()["items"]:
        for key in ("name", "plural_name"):
            val = item.get(key)
            if val and val not in seen:
                foods.append(val)
                seen.add(val)
        for alias in item.get("aliases", []):
            val = alias.get("name")
            if val and val not in seen:
                foods.append(val)
                seen.add(val)

    units = []
    seen = set()
    for item in units_resp.json()["items"]:
        for key in ("name", "plural_name", "abbreviation", "plural_abbreviation"):
            val = item.get(key)
            if val and val not in seen:
                units.append(val)
                seen.add(val)
        for alias in item.get("aliases", []):
            val = alias.get("name")
            if val and val not in seen:
                units.append(val)
                seen.add(val)

    return foods, units


def build_schema(foods, units, constrained=True):
    if constrained and foods:
        food_prop = {"enum": foods + [None]}
    else:
        food_prop = {"type": ["string", "null"]}

    if constrained and units:
        unit_prop = {"enum": units + [None]}
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


def build_messages(ingredient_text):
    prompt = f"# Template:\n{NUEXTRACT_TEMPLATE}\n\n# Context:\n{ingredient_text}"
    return [{"role": "user", "content": prompt}]


def score_result(result, expected):
    """Score a result: 1 point each for food, unit, qty match."""
    scores = {}

    # Food: fuzzy match (case-insensitive, check if expected appears in result or vice versa)
    r_food = (result.get("food") or "").lower().strip()
    e_food = (expected["food"] or "").lower().strip()
    if e_food and r_food:
        scores["food"] = 1 if (e_food in r_food or r_food in e_food) else 0
    elif not e_food and not r_food:
        scores["food"] = 1
    else:
        scores["food"] = 0

    # Unit: similar fuzzy
    r_unit = (result.get("unit") or "").lower().strip()
    e_unit = (expected["unit"] or "").lower().strip()
    if e_unit and r_unit:
        scores["unit"] = 1 if (e_unit in r_unit or r_unit in e_unit) else 0
    elif not e_unit and not r_unit:
        scores["unit"] = 1
    else:
        scores["unit"] = 0

    # Quantity
    r_qty = result.get("quantity")
    e_qty = expected["qty"]
    if e_qty is None and r_qty is None:
        scores["qty"] = 1
    elif e_qty is not None and r_qty is not None:
        scores["qty"] = 1 if abs(float(r_qty) - e_qty) < 0.01 else 0
    else:
        scores["qty"] = 0

    return scores


def run_benchmark(model, foods, units):
    constrained_schema = build_schema(foods, units, constrained=True)
    free_schema = build_schema(foods, units, constrained=False)

    print(f"\nEnum sizes: {len(foods)} foods, {len(units)} units")
    print(f"Test cases: {len(TEST_CASES)}\n")
    print("=" * 100)

    for mode, schema, label in [
        ("constrained", constrained_schema, "GBNF Constrained (current)"),
        ("free", free_schema, "Free-form (no enum)"),
    ]:
        print(f"\n{'=' * 100}")
        print(f" {label}")
        print(f"{'=' * 100}\n")

        total_time = 0
        total_scores = {"food": 0, "unit": 0, "qty": 0}
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema))

        for tc in TEST_CASES:
            messages = build_messages(tc["input"])

            start = time.perf_counter()
            response = model.create_chat_completion(
                messages=messages,
                grammar=grammar,
                temperature=0,
                max_tokens=-1,
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
            scores = score_result(result, tc)

            for k, v in scores.items():
                total_scores[k] += v

            status = "OK" if all(scores.values()) else "MISS"
            print(f"  [{status}] {tc['input']:<50} ({elapsed:.1f}s)")
            print(f"         got: food={result.get('food')!r:30} unit={result.get('unit')!r:15} qty={result.get('quantity')}")
            if not all(scores.values()):
                print(f"         exp: food={tc['food']!r:30} unit={tc['unit']!r:15} qty={tc['qty']}")
            print()

        n = len(TEST_CASES)
        print(f"  Summary for {label}:")
        print(f"    Total time:   {total_time:.1f}s ({total_time/n:.1f}s/ingredient)")
        print(f"    Food accuracy: {total_scores['food']}/{n} ({100*total_scores['food']/n:.0f}%)")
        print(f"    Unit accuracy: {total_scores['unit']}/{n} ({100*total_scores['unit']/n:.0f}%)")
        print(f"    Qty accuracy:  {total_scores['qty']}/{n} ({100*total_scores['qty']/n:.0f}%)")
        print(f"    Overall:       {sum(total_scores.values())}/{3*n} ({100*sum(total_scores.values())/(3*n):.0f}%)")


def main():
    print("Fetching foods and units from Mealie...")
    foods, units = fetch_mealie_data()
    print(f"  Got {len(foods)} foods, {len(units)} units")

    print(f"\nLoading NuExtract-2.0-2B from {MODEL_REPO}...")
    print(f"  Cache dir: {MODEL_CACHE}")
    os.makedirs(MODEL_CACHE, exist_ok=True)

    model = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        n_ctx=4096,
        verbose=False,
        cache_dir=MODEL_CACHE,
    )
    print("  Model loaded.")

    run_benchmark(model, foods, units)
    model.close()


if __name__ == "__main__":
    main()
