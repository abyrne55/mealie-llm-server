import pytest
from unittest.mock import MagicMock, patch
from mealie_llm_server.model_manager import ModelManager


def _mock_from_pretrained(**kwargs):
    model = MagicMock()
    model.close = MagicMock()
    return model


@pytest.fixture
def models_config():
    return {
        "ingredient_extractor": "numind/NuExtract-2.0-2B-GGUF:Q6_K",
        "general": None,
    }


class TestModelManager:
    @patch("mealie_llm_server.model_manager.Llama")
    def test_load_model_parses_id_correctly(self, mock_llama, models_config, tmp_path):
        mock_llama.from_pretrained = MagicMock(side_effect=_mock_from_pretrained)
        manager = ModelManager(
            models=models_config, strategy="all", n_ctx=4096, n_threads=None, cache_dir=str(tmp_path)
        )
        manager._load("ingredient_extractor")
        mock_llama.from_pretrained.assert_called_once()
        call_kwargs = mock_llama.from_pretrained.call_args
        assert call_kwargs.kwargs["repo_id"] == "numind/NuExtract-2.0-2B-GGUF"
        assert call_kwargs.kwargs["filename"] == "*Q6_K.gguf"

    @patch("mealie_llm_server.model_manager.Llama")
    def test_load_model_calls_from_pretrained(self, mock_llama, models_config, tmp_path):
        mock_llama.from_pretrained = MagicMock(side_effect=_mock_from_pretrained)
        manager = ModelManager(models=models_config, strategy="all", n_ctx=4096, n_threads=2, cache_dir=str(tmp_path))
        manager._load("ingredient_extractor")
        call_kwargs = mock_llama.from_pretrained.call_args.kwargs
        assert call_kwargs["n_ctx"] == 4096
        assert call_kwargs["n_threads"] == 2
        assert call_kwargs["verbose"] is False

    @patch("mealie_llm_server.model_manager.Llama")
    @pytest.mark.asyncio
    async def test_all_strategy_loads_at_startup(self, mock_llama, tmp_path):
        mock_llama.from_pretrained = MagicMock(side_effect=_mock_from_pretrained)
        models = {
            "ingredient_extractor": "numind/NuExtract-2.0-2B-GGUF:Q6_K",
            "general": "some/Model-GGUF:Q4_K_M",
        }
        manager = ModelManager(models=models, strategy="all", n_ctx=4096, n_threads=None, cache_dir=str(tmp_path))
        await manager.load_all()
        assert mock_llama.from_pretrained.call_count == 2

    @patch("mealie_llm_server.model_manager.Llama")
    @pytest.mark.asyncio
    async def test_swap_strategy_loads_on_demand(self, mock_llama, tmp_path):
        mock_llama.from_pretrained = MagicMock(side_effect=_mock_from_pretrained)
        models = {
            "ingredient_extractor": "numind/NuExtract-2.0-2B-GGUF:Q6_K",
            "general": "some/Model-GGUF:Q4_K_M",
        }
        manager = ModelManager(models=models, strategy="swap", n_ctx=4096, n_threads=None, cache_dir=str(tmp_path))
        assert mock_llama.from_pretrained.call_count == 0

        async with manager.get_model("ingredient_extractor") as model:
            assert model is not None
        assert mock_llama.from_pretrained.call_count == 1

        async with manager.get_model("general") as model:
            assert model is not None
        assert mock_llama.from_pretrained.call_count == 2

    @patch("mealie_llm_server.model_manager.Llama")
    @pytest.mark.asyncio
    async def test_get_model_returns_lock_context(self, mock_llama, models_config, tmp_path):
        mock_llama.from_pretrained = MagicMock(side_effect=_mock_from_pretrained)
        manager = ModelManager(
            models=models_config, strategy="all", n_ctx=4096, n_threads=None, cache_dir=str(tmp_path)
        )
        await manager.load_all()
        async with manager.get_model("ingredient_extractor") as model:
            assert model is not None
