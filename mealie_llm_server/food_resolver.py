from __future__ import annotations

import logging

import numpy as np
from model2vec import StaticModel

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.65


class FoodResolver:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        logger.info("Loading embedding model %s", model_name)
        self._model = StaticModel.from_pretrained(model_name)
        self._food_names: list[str] = []
        self._food_names_lower: dict[str, str] = {}
        self._food_embeddings: np.ndarray | None = None
        self._food_hash: int = 0

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    def _ensure_embeddings(self, foods: list[str]) -> None:
        h = hash(tuple(foods))
        if h == self._food_hash and self._food_embeddings is not None:
            return
        logger.info("Embedding %d food names", len(foods))
        self._food_names = list(foods)
        self._food_names_lower = {f.lower(): f for f in foods}
        self._food_embeddings = self._embed(foods)
        self._food_hash = h

    def match(self, query: str, foods: list[str], threshold: float = _DEFAULT_THRESHOLD) -> str | None:
        if not query or not foods:
            return None
        self._ensure_embeddings(foods)

        exact = self._food_names_lower.get(query.lower().strip())
        if exact is not None:
            return exact

        query_vec = self._embed([query])
        similarities = (query_vec @ self._food_embeddings.T)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        logger.debug("Food match: %r -> %r (%.3f)", query, self._food_names[best_idx], best_score)
        if best_score >= threshold:
            return self._food_names[best_idx]
        return None
