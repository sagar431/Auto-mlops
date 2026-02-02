#!/usr/bin/env python3
"""
Tests for .env.example file completeness.

Run with: pytest test_env_example.py -v
"""

from pathlib import Path

import pytest


@pytest.fixture
def env_example_content() -> str:
    """Load .env.example file content."""
    env_example_path = Path(__file__).parent / ".env.example"
    return env_example_path.read_text()


class TestEnvExample:
    """Tests for .env.example file."""

    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        env_example_path = Path(__file__).parent / ".env.example"
        assert env_example_path.exists(), ".env.example file should exist"

    def test_database_url_present(self, env_example_content: str):
        """Test that DATABASE_URL is defined in .env.example."""
        assert (
            "DATABASE_URL=" in env_example_content
        ), "DATABASE_URL should be defined in .env.example"

    def test_database_url_has_sqlite_default(self, env_example_content: str):
        """Test that DATABASE_URL has SQLite as default for development."""
        assert (
            "sqlite:///" in env_example_content
        ), "DATABASE_URL should have SQLite default for development"

    def test_database_url_has_postgresql_example(self, env_example_content: str):
        """Test that DATABASE_URL has PostgreSQL example for production."""
        assert (
            "postgresql://" in env_example_content
        ), "DATABASE_URL should have PostgreSQL example for production"

    def test_llm_api_keys_present(self, env_example_content: str):
        """Test that LLM API keys are defined."""
        assert "OPENAI_API_KEY=" in env_example_content
        assert "GOOGLE_API_KEY=" in env_example_content

    def test_mlflow_config_present(self, env_example_content: str):
        """Test that MLflow configuration is defined."""
        assert "MLFLOW_TRACKING_URI=" in env_example_content
        assert "MLFLOW_EXPERIMENT_NAME=" in env_example_content

    def test_dvc_config_present(self, env_example_content: str):
        """Test that DVC configuration is defined."""
        assert "DVC_REMOTE_URL=" in env_example_content

    def test_docker_config_present(self, env_example_content: str):
        """Test that Docker configuration is defined."""
        assert "DOCKER_REGISTRY=" in env_example_content
        assert "DOCKER_USERNAME=" in env_example_content

    def test_agent_settings_present(self, env_example_content: str):
        """Test that agent settings are defined."""
        assert "MAX_IMPROVEMENT_ATTEMPTS=" in env_example_content
        assert "DEFAULT_ACCURACY_THRESHOLD=" in env_example_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
