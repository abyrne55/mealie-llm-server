from __future__ import annotations

import asyncio
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
    return os.environ.get("MODEL_INGREDIENT_EXTRACTOR", "DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K")


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
