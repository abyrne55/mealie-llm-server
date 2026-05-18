from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


_FILE_SUFFIX = "_FILE"

_FILE_FIELDS = [
    "MEALIE_URL",
    "MEALIE_API_KEY",
    "MEALIE_CACHE_TTL",
    "UPSTREAM_URL",
    "UPSTREAM_API_KEY",
    "UPSTREAM_TIMEOUT",
    "MODEL_INGREDIENT_PARSING",
    "MODEL_GENERAL",
    "MODEL_LOADING_STRATEGY",
    "MODEL_CONTEXT_SIZE",
    "MODEL_THREADS",
    "MODEL_CACHE_DIR",
    "ROUTER_THRESHOLD",
    "LOG_LEVEL",
]


class Settings(BaseSettings):
    MEALIE_URL: str
    MEALIE_API_KEY: str = ""
    MEALIE_CACHE_TTL: int = 300

    UPSTREAM_URL: str | None = None
    UPSTREAM_API_KEY: str | None = None
    UPSTREAM_TIMEOUT: int = 300

    MODEL_INGREDIENT_PARSING: str = "openbmb/MiniCPM-V-4.6-gguf:Q6_K"
    MODEL_GENERAL: str | None = None
    MODEL_LOADING_STRATEGY: str = "all"
    MODEL_CONTEXT_SIZE: int = 4096
    MODEL_THREADS: int | None = None
    MODEL_CACHE_DIR: str = "/models"

    ROUTER_THRESHOLD: float = 0.6
    LOG_LEVEL: str = "info"

    MEALIE_URL_FILE: str | None = None
    MEALIE_API_KEY_FILE: str | None = None
    MEALIE_CACHE_TTL_FILE: str | None = None
    UPSTREAM_URL_FILE: str | None = None
    UPSTREAM_API_KEY_FILE: str | None = None
    UPSTREAM_TIMEOUT_FILE: str | None = None
    MODEL_INGREDIENT_PARSING_FILE: str | None = None
    MODEL_GENERAL_FILE: str | None = None
    MODEL_LOADING_STRATEGY_FILE: str | None = None
    MODEL_CONTEXT_SIZE_FILE: str | None = None
    MODEL_THREADS_FILE: str | None = None
    MODEL_CACHE_DIR_FILE: str | None = None
    ROUTER_THRESHOLD_FILE: str | None = None
    LOG_LEVEL_FILE: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_file_variants(cls, values: dict) -> dict:
        for field in _FILE_FIELDS:
            file_key = field + _FILE_SUFFIX
            file_path = values.get(file_key)
            if file_path:
                values[field] = Path(file_path).read_text().strip()
        return values

    @staticmethod
    def parse_model_id(model_str: str) -> tuple[str, str]:
        if ":" not in model_str:
            raise ValueError(f"Invalid model ID '{model_str}': expected 'repo:quant_tag' format")
        repo_id, quant_tag = model_str.rsplit(":", 1)
        return repo_id, f"*{quant_tag}.gguf"
