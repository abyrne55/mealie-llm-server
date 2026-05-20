from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from llama_cpp import Llama

from mealie_llm_server.config import Settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self,
        models: dict[str, str | None],
        strategy: str,
        n_ctx: int,
        n_threads: int | None,
        cache_dir: str,
    ):
        self._model_configs = models
        self._strategy = strategy
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._cache_dir = cache_dir
        self._loaded: dict[str, Llama] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._current_swap_key: str | None = None

    async def load_all(self) -> None:
        for key, model_id in self._model_configs.items():
            if model_id is None:
                continue
            self._load(key)

    def _load(self, key: str) -> None:
        model_id = self._model_configs[key]
        if model_id is None:
            return
        logger.info("Loading model %s (%s)", key, model_id)
        if Settings.is_local_gguf(model_id):
            model = Llama(
                model_path=model_id,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )
        else:
            repo_id, filename = Settings.parse_model_id(model_id)
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
                cache_dir=self._cache_dir,
            )
        self._loaded[key] = model
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

    @asynccontextmanager
    async def get_model(self, key: str) -> AsyncIterator[Llama]:
        if self._strategy == "swap":
            async with self._global_lock:
                if key not in self._loaded:
                    if self._current_swap_key and self._current_swap_key in self._loaded:
                        old = self._loaded.pop(self._current_swap_key)
                        old.close()
                    self._load(key)
                self._current_swap_key = key
                yield self._loaded[key]
        else:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            async with self._locks[key]:
                yield self._loaded[key]

    async def close(self) -> None:
        for model in self._loaded.values():
            model.close()
        self._loaded.clear()
