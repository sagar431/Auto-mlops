#!/usr/bin/env python3
"""
Tests for configurable CORS origins in api_server.

Run with: pytest test_cors_config.py -v
"""

import os

import pytest


class TestGetCorsOrigins:
    """Tests for get_cors_origins function."""

    def test_default_returns_wildcard(self):
        """Test that default (unset env var) returns ['*']."""
        # Ensure env var is not set
        os.environ.pop("CORS_ORIGINS", None)

        # Need to reload the function after changing env
        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["*"]

    def test_empty_string_returns_wildcard(self):
        """Test that empty string returns ['*']."""
        os.environ["CORS_ORIGINS"] = ""

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["*"]

    def test_whitespace_only_returns_wildcard(self):
        """Test that whitespace-only string returns ['*']."""
        os.environ["CORS_ORIGINS"] = "   "

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["*"]

    def test_single_origin(self):
        """Test parsing a single origin."""
        os.environ["CORS_ORIGINS"] = "http://localhost:3000"

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["http://localhost:3000"]

    def test_multiple_origins(self):
        """Test parsing multiple comma-separated origins."""
        os.environ["CORS_ORIGINS"] = "http://localhost:3000,https://example.com"

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["http://localhost:3000", "https://example.com"]

    def test_origins_with_whitespace(self):
        """Test that whitespace around origins is trimmed."""
        os.environ["CORS_ORIGINS"] = "  http://localhost:3000 , https://example.com  "

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["http://localhost:3000", "https://example.com"]

    def test_origins_with_trailing_comma(self):
        """Test that trailing comma doesn't add empty string."""
        os.environ["CORS_ORIGINS"] = "http://localhost:3000,https://example.com,"

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["http://localhost:3000", "https://example.com"]

    def test_origins_with_empty_items(self):
        """Test that empty items between commas are filtered out."""
        os.environ["CORS_ORIGINS"] = "http://localhost:3000,,https://example.com"

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == ["http://localhost:3000", "https://example.com"]

    def test_multiple_origins_mixed_protocols(self):
        """Test parsing origins with different protocols and ports."""
        os.environ["CORS_ORIGINS"] = (
            "http://localhost:3000,https://api.example.com,http://192.168.1.100:8080"
        )

        from api_server import get_cors_origins

        result = get_cors_origins()
        assert result == [
            "http://localhost:3000",
            "https://api.example.com",
            "http://192.168.1.100:8080",
        ]

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Clean up environment variable after each test."""
        yield
        os.environ.pop("CORS_ORIGINS", None)


class TestCorsMiddlewareConfiguration:
    """Tests that CORS middleware is properly configured."""

    def test_cors_middleware_is_configured(self):
        """Test that CORSMiddleware is added to the app."""
        # Clear any cached imports
        os.environ.pop("CORS_ORIGINS", None)

        from fastapi.middleware.cors import CORSMiddleware

        from api_server import app

        # Check that CORSMiddleware is in the middleware stack
        cors_middleware_found = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware_found = True
                break

        assert cors_middleware_found, "CORSMiddleware should be configured"

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Clean up environment variable after each test."""
        yield
        os.environ.pop("CORS_ORIGINS", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
