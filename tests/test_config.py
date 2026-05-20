import pytest
from mealie_llm_server.config import Settings


class TestSettings:
    def test_required_fields(self):
        settings = Settings(MEALIE_URL="http://localhost", MEALIE_API_KEY="key")
        assert settings.MEALIE_URL == "http://localhost"
        assert settings.MEALIE_API_KEY == "key"

    def test_defaults(self):
        settings = Settings(MEALIE_URL="http://localhost", MEALIE_API_KEY="key")
        assert settings.MEALIE_CACHE_TTL == 300
        assert settings.MODEL_INGREDIENT_EXTRACTOR == "abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser:q8_0"
        assert settings.MODEL_INGREDIENT_RESOLVER == "minishlab/potion-retrieval-32M"
        assert settings.MODEL_LOADING_STRATEGY == "all"
        assert settings.MODEL_CONTEXT_SIZE == 4096
        assert settings.MODEL_CACHE_DIR == "/models"
        assert settings.UPSTREAM_TIMEOUT == 300
        assert settings.ROUTER_THRESHOLD == 0.6
        assert settings.LOG_LEVEL == "info"

    def test_file_variant_reads_from_file(self, tmp_path):
        secret_file = tmp_path / "api_key"
        secret_file.write_text("secret-from-file")
        settings = Settings(
            MEALIE_URL="http://localhost",
            MEALIE_API_KEY_FILE=str(secret_file),
        )
        assert settings.MEALIE_API_KEY == "secret-from-file"

    def test_file_variant_takes_precedence(self, tmp_path):
        secret_file = tmp_path / "api_key"
        secret_file.write_text("from-file")
        settings = Settings(
            MEALIE_URL="http://localhost",
            MEALIE_API_KEY="from-env",
            MEALIE_API_KEY_FILE=str(secret_file),
        )
        assert settings.MEALIE_API_KEY == "from-file"

    def test_file_variant_strips_whitespace(self, tmp_path):
        secret_file = tmp_path / "url"
        secret_file.write_text("http://mealie.local\n")
        settings = Settings(
            MEALIE_URL_FILE=str(secret_file),
            MEALIE_API_KEY="key",
        )
        assert settings.MEALIE_URL == "http://mealie.local"

    def test_parse_model_id(self):
        settings = Settings(MEALIE_URL="http://localhost", MEALIE_API_KEY="key")
        repo_id, filename = settings.parse_model_id(settings.MODEL_INGREDIENT_EXTRACTOR)
        assert repo_id == "abyrne55/nuextract-1.5-tiny-mealie-ingredient-parser"
        assert filename == "*q8_0.gguf"

    def test_parse_model_id_missing_colon(self):
        settings = Settings(MEALIE_URL="http://localhost", MEALIE_API_KEY="key")
        with pytest.raises(ValueError):
            settings.parse_model_id("invalid-no-colon")

    def test_file_variant_for_numeric_field(self, tmp_path):
        ttl_file = tmp_path / "ttl"
        ttl_file.write_text("600\n")
        settings = Settings(
            MEALIE_URL="http://localhost",
            MEALIE_API_KEY="key",
            MEALIE_CACHE_TTL_FILE=str(ttl_file),
        )
        assert settings.MEALIE_CACHE_TTL == 600

    def test_file_variant_for_float_field(self, tmp_path):
        threshold_file = tmp_path / "threshold"
        threshold_file.write_text("0.8\n")
        settings = Settings(
            MEALIE_URL="http://localhost",
            MEALIE_API_KEY="key",
            ROUTER_THRESHOLD_FILE=str(threshold_file),
        )
        assert settings.ROUTER_THRESHOLD == 0.8

    def test_is_local_gguf_absolute_path(self):
        assert Settings.is_local_gguf("/models/model.gguf") is True

    def test_is_local_gguf_relative_path(self):
        assert Settings.is_local_gguf("./models/model.gguf") is True

    def test_is_local_gguf_repo_format(self):
        assert Settings.is_local_gguf("DevQuasar-3/numind.NuExtract-tiny-v1.5-GGUF:Q6_K") is False

    def test_is_local_gguf_plain_filename(self):
        assert Settings.is_local_gguf("model.gguf") is False
