import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch


@pytest.fixture
def mock_settings(tmp_path):
    with patch.dict("os.environ", {
        "MEALIE_URL": "http://mealie.test:9000",
        "MEALIE_API_KEY": "test-key",
        "MODEL_CACHE_DIR": str(tmp_path / "models"),
        "MODEL_LOADING_STRATEGY": "swap",
    }):
        yield


@pytest.fixture
async def client(mock_settings):
    from mealie_llm_server.app import app, lifespan
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCacheEndpoint:
    @pytest.mark.asyncio
    async def test_delete_cache(self, client):
        response = await client.delete("/v1/cache")
        assert response.status_code == 200


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_unmatched_prompt_returns_501_without_upstream(self, client):
        response = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        })
        assert response.status_code == 501
