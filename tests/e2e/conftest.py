from __future__ import annotations

import asyncio
import html as html_mod
import json
import os
from pathlib import Path

import httpx
import pytest
from llama_cpp import LlamaGrammar

from mealie_local_ai.config import Settings
from mealie_local_ai.food_resolver import FoodResolver
from mealie_local_ai.handlers.ingredient_parsing import _STRUCTURE_SCHEMA
from mealie_local_ai.mealie_client import MealieClient
from mealie_local_ai.regex_parser import RegexParser


def _load_env_file() -> None:
    if os.environ.get("MEALIE_URL") and os.environ.get("MEALIE_API_KEY"):
        return
    env_file = Path(__file__).resolve().parents[2] / ".env.test"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


@pytest.fixture(scope="session")
def mealie_url() -> str:
    url = os.environ.get("MEALIE_URL")
    if not url:
        pytest.skip("MEALIE_URL not set")
    return url


@pytest.fixture(scope="session")
def mealie_api_key() -> str:
    key = os.environ.get("MEALIE_API_KEY")
    if not key:
        pytest.skip("MEALIE_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def mealie_client(mealie_url, mealie_api_key):
    try:
        resp = httpx.get(f"{mealie_url}/api/app/about", timeout=5)
        resp.raise_for_status()
    except httpx.ConnectError, httpx.HTTPStatusError:
        pytest.skip(f"Mealie not reachable at {mealie_url}")
    return MealieClient(mealie_url, mealie_api_key)


@pytest.fixture(scope="session")
def model_id() -> str:
    return os.environ.get("MODEL_INGREDIENT_EXTRACTOR", "abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser:q8_0")


@pytest.fixture(scope="session")
def llm_model(model_id):
    from llama_cpp import Llama

    try:
        if Settings.is_local_gguf(model_id):
            model = Llama(model_path=model_id, n_ctx=4096, verbose=False)
        else:
            repo_id, filename = Settings.parse_model_id(model_id)
            cache_dir = os.environ.get("MODEL_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".tmp" / "models"))
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
def food_resolver():
    return FoodResolver(model_name="minishlab/potion-retrieval-32M")


@pytest.fixture(scope="session")
def foods(mealie_client):
    return asyncio.run(mealie_client.get_foods())


@pytest.fixture(scope="session")
def unit_aliases(mealie_client):
    return asyncio.run(mealie_client.get_unit_aliases())


@pytest.fixture(scope="session")
def regex_parser(foods, unit_aliases):
    parser = RegexParser()
    all_units = [a for aliases in unit_aliases.values() for a in aliases]
    parser.build(foods, all_units)
    return parser


@pytest.fixture(scope="session")
def grammar():
    return LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))


_collected_results: list[dict] = []


@pytest.fixture(scope="session")
def results_collector():
    yield _collected_results
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / "e2e-report.json"
    with open(report_path, "w") as f:
        json.dump(_collected_results, f, indent=2, default=str)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report._e2e_result = getattr(item, "_e2e_result", None)


def pytest_html_results_table_header(cells):
    # default: Result(0) Test(1) Duration(2) Links(3)
    del cells[3]  # Links
    del cells[1]  # Test ID
    # now: Result(0) Duration(1)
    cells.insert(1, "<th>Input</th>")
    cells.insert(2, "<th>Expected</th>")
    cells.insert(3, "<th>Actual</th>")
    cells.insert(4, "<th>Trace</th>")
    # final: Result | Input | Expected | Actual | Trace | Duration


def _format_fields(data: dict) -> str:
    parts = []
    for field in ("quantity", "unit", "food", "note"):
        val = data.get(field)
        if val is not None and val != "":
            parts.append(f"<b>{field}:</b> {html_mod.escape(str(val))}")
    return "<br>".join(parts)


def pytest_html_results_table_row(report, cells):
    result = getattr(report, "_e2e_result", None)
    del cells[-1]  # Links
    del cells[1]  # Test ID
    if result is None:
        cells.insert(1, "<td>-</td>")
        cells.insert(2, "<td>-</td>")
        cells.insert(3, "<td>-</td>")
        cells.insert(4, "<td>-</td>")
        return

    input_text = html_mod.escape(result["input"])
    cells.insert(1, f"<td>{input_text}</td>")
    cells.insert(2, f"<td>{_format_fields(result['expected'])}</td>")
    cells.insert(3, f"<td>{_format_fields(result['actual'])}</td>")

    trace = result.get("trace", {})
    mismatches = result.get("mismatches", [])
    note_score = result.get("note_score")
    trace_parts = []
    for stage, value in trace.items():
        display = json.dumps(value, default=str) if isinstance(value, dict) else str(value)
        trace_parts.append(f"<b>{stage}:</b> {html_mod.escape(display)}")
    if note_score is not None:
        trace_parts.append(f"<b>note_similarity:</b> {note_score:.2f}")
    if mismatches:
        mismatch_strs = [f"{m['field']}: {m['expected']!r} → {m['actual']!r}" for m in mismatches]
        trace_parts.append(f"<b>mismatches:</b> {html_mod.escape('; '.join(mismatch_strs))}")
    cells.insert(4, f"<td>{'<br>'.join(trace_parts)}</td>")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _collected_results:
        return
    terminalreporter.section("E2E Ingredient Parsing Results")
    passed = sum(1 for r in _collected_results if r["status"] == "pass")
    total = len(_collected_results)
    for r in _collected_results:
        status = "PASS" if r["status"] == "pass" else "FAIL"
        line = f"  {status}: {r['input']}"
        if r["mismatches"]:
            details = ", ".join(f"{m['field']}: {m['expected']!r} != {m['actual']!r}" for m in r["mismatches"])
            line += f" [{details}]"
        terminalreporter.line(line)
    terminalreporter.line(f"\n  Pass rate: {passed}/{total} ({100 * passed / total:.0f}%)")
