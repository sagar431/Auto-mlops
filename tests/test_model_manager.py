"""Tests for ModelManager LLM fallback chain functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.model_manager import DEFAULT_FALLBACK_CHAIN, ModelManager, get_model_manager


@pytest.fixture
def mock_config():
    """Test model configuration."""
    return {
        "default": "gemini",
        "fallback_chain": ["gemini", "gpt4", "gemini_flash"],
        "models": {
            "gemini": {
                "provider": "google",
                "model": "gemini-2.0-flash",
                "temperature": 0.7,
                "max_tokens": 8192,
            },
            "gemini_flash": {
                "provider": "google",
                "model": "gemini-1.5-flash",
                "temperature": 0.7,
                "max_tokens": 8192,
            },
            "gpt4": {
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        },
    }


@pytest.fixture
def model_manager(mock_config, tmp_path):
    """Create a ModelManager with test config."""
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(mock_config))
    return ModelManager(config_path=str(config_path))


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_loads_fallback_chain_from_config(self, model_manager):
        """Test that fallback chain is loaded from config."""
        assert model_manager.fallback_chain == ["gemini", "gpt4", "gemini_flash"]

    def test_default_fallback_chain_when_not_in_config(self, tmp_path):
        """Test that default fallback chain is used when not in config."""
        config = {
            "default": "gemini",
            "models": {
                "gemini": {"provider": "google", "model": "gemini-2.0-flash"},
            },
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))

        manager = ModelManager(config_path=str(config_path))
        assert manager.fallback_chain == DEFAULT_FALLBACK_CHAIN

    def test_default_fallback_chain_constant(self):
        """Test the default fallback chain constant value."""
        assert DEFAULT_FALLBACK_CHAIN == ["gemini", "gpt4", "gemini_flash"]


class TestBuildFallbackChain:
    """Tests for _build_fallback_chain method."""

    def test_starting_model_in_chain(self, model_manager):
        """Test chain building when starting model is in the fallback chain."""
        # Starting from first model
        chain = model_manager._build_fallback_chain("gemini")
        assert chain == ["gemini", "gpt4", "gemini_flash"]

        # Starting from second model
        chain = model_manager._build_fallback_chain("gpt4")
        assert chain == ["gpt4", "gemini_flash"]

        # Starting from last model
        chain = model_manager._build_fallback_chain("gemini_flash")
        assert chain == ["gemini_flash"]

    def test_starting_model_not_in_chain(self, model_manager):
        """Test chain building when starting model is not in the fallback chain."""
        # Add a custom model not in chain
        model_manager.models["custom"] = {"provider": "openai", "model": "custom-model"}

        chain = model_manager._build_fallback_chain("custom")
        assert chain == ["custom", "gemini", "gpt4", "gemini_flash"]


class TestGenerateWithFallback:
    """Tests for generate method with fallback chain."""

    @pytest.mark.asyncio
    async def test_successful_first_model(self, model_manager):
        """Test successful generation with first model (no fallback needed)."""
        with (
            patch.object(model_manager, "_get_client") as mock_get_client,
            patch.object(
                model_manager, "_generate_gemini", new_callable=AsyncMock
            ) as mock_generate,
        ):
            mock_get_client.return_value = MagicMock()
            mock_generate.return_value = "Generated text"

            result = await model_manager.generate("Test prompt")

            assert result == "Generated text"
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_first_model_failure(self, model_manager):
        """Test fallback to second model when first fails."""
        call_count = {"gemini": 0, "openai": 0}

        def mock_get_client(provider):
            if provider == "google":
                return MagicMock()
            elif provider == "openai":
                return MagicMock()
            return None

        async def mock_generate_gemini(*args, **kwargs):
            call_count["gemini"] += 1
            raise Exception("Gemini API error")

        async def mock_generate_openai(*args, **kwargs):
            call_count["openai"] += 1
            return "OpenAI response"

        with (
            patch.object(model_manager, "_get_client", side_effect=mock_get_client),
            patch.object(model_manager, "_generate_gemini", side_effect=mock_generate_gemini),
            patch.object(model_manager, "_generate_openai", side_effect=mock_generate_openai),
        ):
            result = await model_manager.generate("Test prompt")

            assert result == "OpenAI response"
            assert call_count["gemini"] == 1
            assert call_count["openai"] == 1

    @pytest.mark.asyncio
    async def test_fallback_to_third_model(self, model_manager):
        """Test fallback to third model when first two fail."""
        call_count = {"gemini": 0, "gemini_flash": 0, "openai": 0}

        def mock_get_client(provider):
            return MagicMock()

        async def mock_generate_gemini(client, model_id, *args, **kwargs):
            # gemini-1.5-flash is the third fallback model
            if model_id == "gemini-1.5-flash":
                call_count["gemini_flash"] += 1
                return "Gemini Flash response"
            # gemini-2.0-flash is the first model
            call_count["gemini"] += 1
            raise Exception("Gemini API error")

        async def mock_generate_openai(*args, **kwargs):
            call_count["openai"] += 1
            raise Exception("OpenAI API error")

        with (
            patch.object(model_manager, "_get_client", side_effect=mock_get_client),
            patch.object(model_manager, "_generate_gemini", side_effect=mock_generate_gemini),
            patch.object(model_manager, "_generate_openai", side_effect=mock_generate_openai),
        ):
            result = await model_manager.generate("Test prompt")

            assert result == "Gemini Flash response"
            assert call_count["gemini"] == 1
            assert call_count["openai"] == 1
            assert call_count["gemini_flash"] == 1

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_error(self, model_manager):
        """Test that RuntimeError is raised when all models fail."""

        def mock_get_client(provider):
            return MagicMock()

        async def mock_generate_fail(*args, **kwargs):
            raise Exception("API error")

        with (
            patch.object(model_manager, "_get_client", side_effect=mock_get_client),
            patch.object(model_manager, "_generate_gemini", side_effect=mock_generate_fail),
            patch.object(model_manager, "_generate_openai", side_effect=mock_generate_fail),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await model_manager.generate("Test prompt")

            assert "All models in fallback chain failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_disabled(self, model_manager):
        """Test that fallback can be disabled."""

        def mock_get_client(provider):
            return MagicMock()

        async def mock_generate_fail(*args, **kwargs):
            raise Exception("API error")

        with (
            patch.object(model_manager, "_get_client", side_effect=mock_get_client),
            patch.object(model_manager, "_generate_gemini", side_effect=mock_generate_fail),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await model_manager.generate("Test prompt", use_fallback=False)

            # Should only have tried one model
            error_msg = str(exc_info.value)
            assert "gemini" in error_msg
            assert "gpt4" not in error_msg

    @pytest.mark.asyncio
    async def test_skip_model_without_client(self, model_manager):
        """Test that models without valid clients are skipped."""

        def mock_get_client(provider):
            if provider == "google":
                return None  # Client init fails
            return MagicMock()

        async def mock_generate_openai(*args, **kwargs):
            return "OpenAI response"

        with (
            patch.object(model_manager, "_get_client", side_effect=mock_get_client),
            patch.object(model_manager, "_generate_openai", side_effect=mock_generate_openai),
        ):
            result = await model_manager.generate("Test prompt")

            assert result == "OpenAI response"

    @pytest.mark.asyncio
    async def test_skip_missing_model_config(self, model_manager):
        """Test that missing model configs are skipped."""
        # Use a fallback chain with a non-existent model
        model_manager.fallback_chain = ["nonexistent", "gpt4"]

        def mock_get_client(provider):
            return MagicMock()

        async def mock_generate_openai(*args, **kwargs):
            return "OpenAI response"

        with (
            patch.object(model_manager, "_get_client", side_effect=mock_get_client),
            patch.object(model_manager, "_generate_openai", side_effect=mock_generate_openai),
        ):
            result = await model_manager.generate("Test prompt", model_name="nonexistent")

            assert result == "OpenAI response"


class TestGenerateTextWithFallback:
    """Tests for generate_text method with fallback chain."""

    @pytest.mark.asyncio
    async def test_generate_text_uses_fallback_by_default(self, model_manager):
        """Test that generate_text uses fallback by default."""
        with patch.object(model_manager, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Generated text"

            await model_manager.generate_text("Test prompt")

            mock_generate.assert_called_once_with(
                prompt="Test prompt",
                model_name=None,
                system_prompt=None,
                response_format="text",
                use_fallback=True,
            )

    @pytest.mark.asyncio
    async def test_generate_text_can_disable_fallback(self, model_manager):
        """Test that generate_text can disable fallback."""
        with patch.object(model_manager, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Generated text"

            await model_manager.generate_text("Test prompt", use_fallback=False)

            mock_generate.assert_called_once_with(
                prompt="Test prompt",
                model_name=None,
                system_prompt=None,
                response_format="text",
                use_fallback=False,
            )


class TestGenerateJsonWithFallback:
    """Tests for generate_json method with fallback chain."""

    @pytest.mark.asyncio
    async def test_generate_json_uses_fallback_by_default(self, model_manager):
        """Test that generate_json uses fallback by default."""
        with patch.object(model_manager, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = '{"key": "value"}'

            await model_manager.generate_json("Test prompt")

            mock_generate.assert_called_once_with(
                prompt="Test prompt",
                model_name=None,
                system_prompt=None,
                response_format="json",
                use_fallback=True,
            )

    @pytest.mark.asyncio
    async def test_generate_json_can_disable_fallback(self, model_manager):
        """Test that generate_json can disable fallback."""
        with patch.object(model_manager, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = '{"key": "value"}'

            await model_manager.generate_json("Test prompt", use_fallback=False)

            mock_generate.assert_called_once_with(
                prompt="Test prompt",
                model_name=None,
                system_prompt=None,
                response_format="json",
                use_fallback=False,
            )


class TestSingletonModelManager:
    """Tests for get_model_manager singleton."""

    def test_singleton_returns_same_instance(self):
        """Test that get_model_manager returns the same instance."""
        # Reset the singleton
        import agent.model_manager as mm

        mm._model_manager = None

        manager1 = get_model_manager()
        manager2 = get_model_manager()

        assert manager1 is manager2

        # Clean up
        mm._model_manager = None
