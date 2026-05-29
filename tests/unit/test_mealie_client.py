import time
import pytest
import httpx
import respx
from mealie_local_ai.mealie_client import MealieClient


FOODS_RESPONSE = {
    "page": 1,
    "per_page": -1,
    "total": 3,
    "total_pages": 1,
    "items": [
        {"name": "flour", "pluralName": "flours", "aliases": [{"name": "all-purpose flour"}]},
        {"name": "egg", "pluralName": "eggs", "aliases": []},
        {"name": "olive oil", "pluralName": None, "aliases": []},
    ],
}

UNITS_RESPONSE = {
    "page": 1,
    "per_page": -1,
    "total": 3,
    "total_pages": 1,
    "items": [
        {
            "name": "cup",
            "pluralName": "cups",
            "abbreviation": "c",
            "pluralAbbreviation": "",
            "aliases": [],
        },
        {
            "name": "tablespoon",
            "pluralName": "tablespoons",
            "abbreviation": "tbsp",
            "pluralAbbreviation": "tbsps",
            "aliases": [{"name": "T"}],
        },
        {
            "name": "teaspoon",
            "pluralName": "teaspoons",
            "abbreviation": "tsp",
            "pluralAbbreviation": "",
            "aliases": [],
        },
    ],
}


@pytest.fixture
def client():
    return MealieClient(base_url="http://mealie.test:9000", api_key="test-key", cache_ttl=300)


class TestFetchFoods:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_foods_extracts_all_name_variants(self, client):
        respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=FOODS_RESPONSE)
        )
        respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=UNITS_RESPONSE)
        )
        foods = await client.get_foods()
        assert "flour" in foods
        assert "flours" in foods
        assert "all-purpose flour" in foods
        assert "egg" in foods
        assert "eggs" in foods
        assert "olive oil" in foods
        assert len(foods) == len(set(foods))


class TestFetchUnits:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_units_extracts_abbreviations(self, client):
        respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=FOODS_RESPONSE)
        )
        respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=UNITS_RESPONSE)
        )
        units = await client.get_units()
        assert "cup" in units
        assert "cups" in units
        assert "c" in units
        assert "tablespoon" in units
        assert "tbsp" in units
        assert "tbsps" in units
        assert "T" in units
        assert "teaspoon" in units
        assert "tsp" in units
        assert len(units) == len(set(units))


class TestCache:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_returns_same_data_within_ttl(self, client):
        foods_route = respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=FOODS_RESPONSE)
        )
        respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=UNITS_RESPONSE)
        )
        await client.get_foods()
        await client.get_foods()
        assert foods_route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_refreshes_after_ttl(self, client):
        foods_route = respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=FOODS_RESPONSE)
        )
        respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=UNITS_RESPONSE)
        )
        await client.get_foods()
        client._last_fetch = time.monotonic() - 400
        await client.get_foods()
        assert foods_route.call_count == 2


class TestStaleOnError:
    @respx.mock
    @pytest.mark.asyncio
    async def test_stale_on_error(self, client):
        foods_route = respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"})
        units_route = respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"})

        foods_route.mock(return_value=httpx.Response(200, json=FOODS_RESPONSE))
        units_route.mock(return_value=httpx.Response(200, json=UNITS_RESPONSE))
        foods_first = await client.get_foods()
        assert len(foods_first) > 0

        client._last_fetch = time.monotonic() - 400
        foods_route.mock(return_value=httpx.Response(500))
        units_route.mock(return_value=httpx.Response(500))
        foods_stale = await client.get_foods()
        assert foods_stale == foods_first

    @respx.mock
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_first_fetch_failure(self, client):
        respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"}).mock(return_value=httpx.Response(500))
        respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"}).mock(return_value=httpx.Response(500))
        foods = await client.get_foods()
        units = await client.get_units()
        assert foods == []
        assert units == []


class TestUnitAliases:
    @respx.mock
    @pytest.mark.asyncio
    async def test_unit_alias_lookup(self, client):
        respx.get("http://mealie.test:9000/api/foods", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=FOODS_RESPONSE)
        )
        respx.get("http://mealie.test:9000/api/units", params={"perPage": "-1"}).mock(
            return_value=httpx.Response(200, json=UNITS_RESPONSE)
        )
        aliases = await client.get_unit_aliases()
        assert "tablespoon" in aliases
        assert "tbsp" in aliases["tablespoon"]
        assert "tablespoons" in aliases["tablespoon"]
        assert "T" in aliases["tablespoon"]
        assert "cup" in aliases
        assert "c" in aliases["cup"]
