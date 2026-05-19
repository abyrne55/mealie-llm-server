from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from mealie_llm_server.config import Settings
from mealie_llm_server.food_resolver import FoodResolver
from mealie_llm_server.mealie_client import MealieClient


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
    except (httpx.ConnectError, httpx.HTTPStatusError):
        pytest.skip(f"Mealie not reachable at {mealie_url}")
    return MealieClient(mealie_url, mealie_api_key)


@pytest.fixture(scope="session")
def model_id() -> str:
    return os.environ.get("MODEL_INGREDIENT_EXTRACTOR", "DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K")


@pytest.fixture(scope="session")
def llm_model(model_id):
    from llama_cpp import Llama

    repo_id, filename = Settings.parse_model_id(model_id)
    cache_dir = os.environ.get("MODEL_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".tmp" / "models"))
    try:
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
