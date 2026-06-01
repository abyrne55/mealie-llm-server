"""Evaluate raw LLM extraction accuracy against the training CSV and novel ingredients.

No Mealie dependency — tests raw model input/output pairs only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from llama_cpp import Llama, LlamaGrammar

from mealie_local_ai.config import Settings
from mealie_local_ai.handlers.ingredient_parsing import (
    _STRUCTURE_SCHEMA,
    build_messages,
    normalize_quantity,
)
from training.training_data import load_training_data

CSV_PATH = Path(__file__).parent / "ingredients.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _plural_eq(a: str, b: str) -> bool:
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower == b_lower:
        return True
    if a_lower + "s" == b_lower or b_lower + "s" == a_lower:
        return True
    if a_lower + "es" == b_lower or b_lower + "es" == a_lower:
        return True
    return False


def _jaccard(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llm_model():
    model_id = os.environ.get(
        "MODEL_INGREDIENT_EXTRACTOR",
        "DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K",
    )
    try:
        if Settings.is_local_gguf(model_id):
            model = Llama(
                model_path=model_id,
                n_ctx=512,
                n_gpu_layers=-1,
                flash_attn=True,
                verbose=False,
            )
        else:
            repo_id, filename = Settings.parse_model_id(model_id)
            cache_dir = os.environ.get(
                "MODEL_CACHE_DIR",
                str(Path(__file__).resolve().parent.parent / ".tmp" / "models"),
            )
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=512,
                n_gpu_layers=-1,
                flash_attn=True,
                verbose=False,
                cache_dir=cache_dir,
            )
    except Exception as e:
        pytest.skip(f"Failed to load model {model_id}: {e}")
    yield model
    model.close()


@pytest.fixture(scope="session")
def grammar():
    return LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))


@pytest.fixture(scope="session")
def results_collector():
    results: list[dict] = []
    yield results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "extractor-report.json"
    sorted_results = sorted(results, key=lambda r: (r.get("suite", ""), r["status"] != "fail", r["input"]))
    with open(report_path, "w") as f:
        json.dump(sorted_results, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract(ingredient_text: str, model: Llama, grammar: LlamaGrammar) -> dict:
    messages = build_messages(ingredient_text)
    response = model.create_chat_completion(
        messages=messages,
        grammar=grammar,
        temperature=0,
        max_tokens=-1,
    )
    content = response["choices"][0]["message"]["content"]
    content = re.sub(r"[\x00-\x1f\x7f]", "", content)
    return json.loads(content)


def _fmt_fields(qty, unit, food, note) -> str:
    parts = [f"qty={qty}"]
    parts.append(f"unit={unit or ''}")
    parts.append(f"food={food or ''}")
    if note:
        parts.append(f"note={note}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CSV accuracy tests
# ---------------------------------------------------------------------------

_csv_rows = load_training_data(CSV_PATH)
_csv_ids = [row[0] for row in _csv_rows]


@pytest.mark.parametrize("ingredient_text,exp_qty,exp_unit,exp_food,exp_note", _csv_rows, ids=_csv_ids)
def test_csv_extraction(
    ingredient_text,
    exp_qty,
    exp_unit,
    exp_food,
    exp_note,
    llm_model,
    grammar,
    results_collector,
):
    raw = _extract(ingredient_text, llm_model, grammar)

    actual_qty = normalize_quantity(raw.get("quantity"))
    expected_qty = normalize_quantity(exp_qty)
    actual_unit = raw.get("unit") or ""
    actual_food = raw.get("food") or ""
    actual_note = raw.get("note") or ""

    mismatches = []

    if expected_qty is None and actual_qty is not None:
        mismatches.append("quantity")
    elif expected_qty is not None and actual_qty is None:
        mismatches.append("quantity")
    elif expected_qty is not None and actual_qty is not None:
        if abs(expected_qty - actual_qty) > 0.01:
            mismatches.append("quantity")

    if not _plural_eq(actual_unit, exp_unit):
        mismatches.append("unit")

    if not _plural_eq(actual_food, exp_food):
        mismatches.append("food")

    note_jaccard = _jaccard(actual_note, exp_note)
    if exp_note and note_jaccard < 0.3:
        mismatches.append("note")
    elif not exp_note and actual_note:
        pass

    status = "fail" if mismatches else "pass"
    actual_fmt = _fmt_fields(actual_qty, actual_unit, actual_food, actual_note)
    result_line = f'{status.upper()} "{ingredient_text}" → {actual_fmt}'
    exp_map = {"quantity": expected_qty, "unit": exp_unit, "food": exp_food, "note": exp_note}
    if mismatches:
        expected_parts = [f"{f}={exp_map[f]}" for f in mismatches]
        result_line += f" [expected: {', '.join(expected_parts)}]"
    elif exp_note and note_jaccard < 1.0:
        result_line += f" [jaccard={note_jaccard:.2f}]"
    print(result_line)

    results_collector.append(
        {
            "suite": "csv",
            "input": ingredient_text,
            "actual": {
                "quantity": actual_qty,
                "unit": actual_unit,
                "food": actual_food,
                "note": actual_note,
            },
            "expected": {
                "quantity": float(expected_qty) if expected_qty is not None else None,
                "unit": exp_unit,
                "food": exp_food,
                "note": exp_note,
            },
            "status": status,
            "jaccard": note_jaccard if exp_note else None,
            "mismatches": mismatches,
        }
    )

    assert not mismatches, f"Mismatches on {mismatches}: got {actual_fmt}"


# ---------------------------------------------------------------------------
# Novel ingredient structural tests
# ---------------------------------------------------------------------------

NOVEL_INGREDIENTS = [
    "1 kilogram fresh pork shoulder",
    "0.25 cucumber",
    "1/4 tsp quail egg",
    "0.5 pumpkin seed",
    "coriander powder, for garnish",
    "1/4 teaspoon turkey breast",
    "2/3 stalks at room temperature swordfish",
    "8 tbsp coconut milk",
    "1/3 cream",
    "1/3 cups arugula, at room temperature and canned",
    "6 stalks abalone, ground",
    "2.5 can kale, roughly chopped",
    "3/4 oz watercress",
    "sweet potato, for garnish",
    "10 tsp monkfish, optional",
    "0.5 lbs room temperature elderberry",
    "3/8 pound yuzu, minced and crushed",
    "5 oz lamb, bone-in",
    "6 thickened cream",
    "1 pecan",
    "1 1/2 teaspoon barramundi, frozen",
    "1/4 ounces cottage cheese, at room temperature",
    "1.5 tablespoons caraway, chopped and optional",
    "melon, for garnish",
    "potato, for garnish",
    "4 dijon mustard",
    "3 pinch turnip",
    "10 coconut milk",
    "6 kg dragon fruit",
    "chard, for garnish",
    "brazil nut, for garnish",
    "1 pinch abalone",
    "cranberry, for garnish",
    "3/8 bread",
    "1 1/2 can pomegranate, ground",
    "1 bunch grape",
    "3/4 lbs monkfish",
    "cottage cheese, for garnish",
    "2/3 dash sun dried tomato",
    "4 pork shoulder",
    "1 1/2 ounce toasted barberry",
    "catfish, for garnish",
    "0.25 goat cheese",
    "0.5 dash kiwi",
    "2/3 beef liver",
    "3/8 banana",
    "tangerine, for garnish",
    "3 pounds fresh broccoli",
    "1 1/2 baby bok choy",
    "1/2 slices parsnip, room temperature and deveined",
    "2 grams shiitake mushroom, plus more for garnish and diced",
    "5 fennel",
    "2.5 teaspoon diced abalone",
    "brussels sprout, for garnish",
    "2 1/2 lbs cream, peeled and bone-in",
    "1/3 duck breast",
    "1.5 kilogram pineapple, fresh",
    "8 lb bacon, finely chopped and julienned",
    "1/3 slice lentil sprout",
    "5 tsp dijon mustard, at room temperature",
    "3/4 avocado",
    "10 teaspoons macadamia, optional",
    "chard, for garnish",
    "0.25 dragon fruit",
    "6 tamarind",
    "3 teaspoons jicama, cut into 1-inch pieces",
    "duck breast, for garnish",
    "2/3 handful chopped spinach",
    "0.25 sprig sunflower seed",
    "melon, for garnish",
    "2/3 slices chestnut, diced and fresh",
    "1/4 sprig apricot kernel, diced and deboned",
    "2 milliliter ricotta cheese, deboned and frozen",
    "3/4 stalk kumquat",
    "bean sprout, for garnish",
    "4 teaspoons brussels sprout, ground and minced",
    "8 bay scallop",
    "1 1/2 pinch sweet potato, divided",
    "10 oz beef liver, skin-on and crushed",
    "pecan, for garnish",
    "2 1/2 lbs truffle cheese",
    "duck breast, for garnish",
    "watercress, for garnish",
    "1 1/2 cup eggplant",
    "3 sprigs cucumber, grated",
    "0.25 bread",
    "12 oz gochujang",
    "1 1/2 silken tofu",
    "pomegranate, for garnish",
    "1 1/2 dash roughly chopped grapefruit",
    "3/8 sprig sun dried tomato, fresh",
    "2 1/2 teaspoon gooseberry, canned and or to taste",
    "10 brussels sprout",
    "2/3 radicchio",
    "6 ounces walnut, frozen",
    "honeydew melon, for garnish",
    "1 handful peeled cottage cheese",
    "1/3 pint arugula, toasted and deveined",
    "3/4 bunch roasted apricot kernel",
    "10 venison",
]


@pytest.mark.parametrize("ingredient", NOVEL_INGREDIENTS, ids=NOVEL_INGREDIENTS)
def test_novel_ingredient_structure(ingredient, llm_model, grammar, results_collector):
    raw = _extract(ingredient, llm_model, grammar)

    actual_qty = normalize_quantity(raw.get("quantity"))
    actual_unit = raw.get("unit") or ""
    actual_food = raw.get("food") or ""
    actual_note = raw.get("note") or ""

    structural_errors = []
    if actual_qty is not None and not isinstance(actual_qty, (int, float)):
        structural_errors.append("quantity not numeric")
    if not actual_food:
        structural_errors.append("food is empty")

    status = "fail" if structural_errors else "pass"
    actual_fmt = _fmt_fields(actual_qty, actual_unit, actual_food, actual_note)
    result_line = f'{status.upper()} "{ingredient}" → {actual_fmt}'
    if structural_errors:
        result_line += f" [{', '.join(structural_errors)}]"
    print(result_line)

    results_collector.append(
        {
            "suite": "novel",
            "input": ingredient,
            "actual": {
                "quantity": actual_qty,
                "unit": actual_unit,
                "food": actual_food,
                "note": actual_note,
            },
            "expected": None,
            "status": status,
            "jaccard": None,
            "mismatches": structural_errors,
        }
    )

    assert not structural_errors, f"Structural errors: {structural_errors}"
