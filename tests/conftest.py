import pytest
from mealie_local_ai.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        MEALIE_URL="http://mealie.test:9000",
        MEALIE_API_KEY="test-api-key",
        MODEL_CACHE_DIR=str(tmp_path / "models"),
    )
