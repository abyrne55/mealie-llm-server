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
from scripts.training_data import load_training_data

CSV_PATH = Path(__file__).resolve().parent / "integration" / "ingredients.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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
            model = Llama(model_path=model_id, n_ctx=4096, verbose=False)
        else:
            repo_id, filename = Settings.parse_model_id(model_id)
            cache_dir = os.environ.get(
                "MODEL_CACHE_DIR",
                str(Path(__file__).resolve().parent.parent / ".tmp" / "models"),
            )
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=4096,
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

    if actual_unit != exp_unit:
        mismatches.append("unit")

    if actual_food != exp_food:
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

from tests.integration.test_novel_ingredients import NOVEL_INGREDIENTS  # noqa: E402


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
