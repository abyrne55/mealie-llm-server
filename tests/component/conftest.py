from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mealie_local_ai.config import Settings
from mealie_local_ai.handlers.ingredient_parsing import _STRUCTURE_SCHEMA


@pytest.fixture(scope="session")
def llm_model():
    from llama_cpp import Llama

    model_id = os.environ.get(
        "MODEL_INGREDIENT_EXTRACTOR",
        "DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K",
    )
    try:
        if Settings.is_local_gguf(model_id):
            model = Llama(model_path=model_id, n_ctx=512, verbose=False)
        else:
            repo_id, filename = Settings.parse_model_id(model_id)
            cache_dir = os.environ.get(
                "MODEL_CACHE_DIR",
                str(Path(__file__).resolve().parents[2] / ".tmp" / "models"),
            )
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=512,
                verbose=False,
                cache_dir=cache_dir,
            )
    except Exception as e:
        pytest.skip(f"Failed to load model {model_id}: {e}")
    yield model
    model.close()


@pytest.fixture(scope="session")
def grammar():
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_json_schema(json.dumps(_STRUCTURE_SCHEMA))


@pytest.fixture(scope="session")
def food_resolver():
    from mealie_local_ai.food_resolver import FoodResolver

    return FoodResolver(model_name="minishlab/potion-retrieval-32M")
