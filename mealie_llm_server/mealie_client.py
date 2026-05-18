from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)


class MealieClient:
    def __init__(self, base_url: str, api_key: str, cache_ttl: int = 300):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._cache_ttl = cache_ttl
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        self._foods: list[str] | None = None
        self._units: list[str] | None = None
        self._unit_aliases: dict[str, list[str]] | None = None
        self._last_fetch: float = 0

    async def get_foods(self) -> list[str]:
        await self._ensure_cache()
        return self._foods if self._foods is not None else []

    async def get_units(self) -> list[str]:
        await self._ensure_cache()
        return self._units if self._units is not None else []

    async def get_unit_aliases(self) -> dict[str, list[str]]:
        await self._ensure_cache()
        return self._unit_aliases if self._unit_aliases is not None else {}

    async def clear_cache(self) -> None:
        self._last_fetch = 0

    async def _ensure_cache(self) -> None:
        if self._last_fetch and (time.monotonic() - self._last_fetch) < self._cache_ttl:
            return
        await self._refresh()

    async def _refresh(self) -> None:
        try:
            foods_resp = await self._client.get(f"{self._base_url}/api/foods", params={"perPage": "-1"})
            foods_resp.raise_for_status()
            units_resp = await self._client.get(f"{self._base_url}/api/units", params={"perPage": "-1"})
            units_resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if self._foods is not None:
                logger.warning("Failed to refresh Mealie cache, using stale data: %s", e)
                return
            logger.warning("Failed initial Mealie fetch, degrading to no enum constraints: %s", e)
            self._foods = []
            self._units = []
            self._unit_aliases = {}
            self._last_fetch = time.monotonic()
            return

        self._foods = self._extract_food_names(foods_resp.json()["items"])
        self._units, self._unit_aliases = self._extract_unit_names(units_resp.json()["items"])
        self._last_fetch = time.monotonic()

    @staticmethod
    def _extract_food_names(items: list[dict]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for item in items:
            for key in ("name", "pluralName"):
                val = item.get(key)
                if val and val not in seen:
                    names.append(val)
                    seen.add(val)
            for alias in item.get("aliases", []):
                val = alias.get("name")
                if val and val not in seen:
                    names.append(val)
                    seen.add(val)
        return names

    @staticmethod
    def _extract_unit_names(items: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
        names: list[str] = []
        seen: set[str] = set()
        alias_map: dict[str, list[str]] = {}

        for item in items:
            canonical = item.get("name", "")
            all_aliases: list[str] = []

            for key in ("name", "pluralName", "abbreviation", "pluralAbbreviation"):
                val = item.get(key)
                if val and val not in seen:
                    names.append(val)
                    seen.add(val)
                if val:
                    all_aliases.append(val)

            for alias in item.get("aliases", []):
                val = alias.get("name")
                if val and val not in seen:
                    names.append(val)
                    seen.add(val)
                if val:
                    all_aliases.append(val)

            if canonical:
                alias_map[canonical] = all_aliases

        return names, alias_map
