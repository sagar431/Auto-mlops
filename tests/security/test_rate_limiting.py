#!/usr/bin/env python3
"""
Tests for the security module's RateLimiter class.

Tests verify:
1. RateLimiter initialization with default and custom configs
2. Client ID extraction from requests (direct IP and X-Forwarded-For)
3. Rate limit enforcement and tracking
4. Request cleanup of expired entries
5. get_remaining() functionality
6. FastAPI dependency integration
7. RateLimitExceeded exception handling
8. Edge cases and concurrent request scenarios

Run with: pytest tests/security/test_rate_limiting.py -v
"""

import time
from unittest.mock import MagicMock

import pytest

from security import (
    RateLimiter,
    RateLimitExceeded,
    SecurityConfig,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.100"
    return request


@pytest.fixture
def mock_request_with_forwarded():
    """Create a mock request with X-Forwarded-For header (behind proxy)."""
    request = MagicMock()
    request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1, 172.16.0.1"}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def mock_request_no_client():
    """Create a mock request with no client info."""
    request = MagicMock()
    request.headers = {}
    request.client = None
    return request


@pytest.fixture
def rate_limit_config(monkeypatch):
    """Create a SecurityConfig with rate limiting enabled."""
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    return SecurityConfig()


@pytest.fixture
def rate_limit_disabled_config(monkeypatch):
    """Create a SecurityConfig with rate limiting disabled."""
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "false")
    return SecurityConfig()


@pytest.fixture
def tight_rate_limit_config(monkeypatch):
    """Create a SecurityConfig with very tight rate limits for testing."""
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "1")
    return SecurityConfig()


@pytest.fixture
def rate_limiter(rate_limit_config):
    """Create a RateLimiter instance with test config."""
    return RateLimiter(config=rate_limit_config)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestRateLimiterInitialization:
    """Tests for RateLimiter initialization."""

    def test_init_with_default_config(self):
        """Test RateLimiter initializes with default SecurityConfig."""
        limiter = RateLimiter()
        assert limiter.config is not None
        assert hasattr(limiter, "_requests")
        assert hasattr(limiter, "_last_cleanup")
        assert limiter._cleanup_interval == 60

    def test_init_with_custom_config(self, rate_limit_config):
        """Test RateLimiter initializes with custom config."""
        limiter = RateLimiter(config=rate_limit_config)
        assert limiter.config == rate_limit_config
        assert limiter.config.rate_limit_requests == 5
        assert limiter.config.rate_limit_window_seconds == 60

    def test_init_creates_empty_request_tracking(self, rate_limit_config):
        """Test RateLimiter starts with empty request tracking."""
        limiter = RateLimiter(config=rate_limit_config)
        assert len(limiter._requests) == 0

    def test_init_sets_last_cleanup_time(self, rate_limit_config):
        """Test RateLimiter sets initial cleanup timestamp."""
        before = time.time()
        limiter = RateLimiter(config=rate_limit_config)
        after = time.time()
        assert before <= limiter._last_cleanup <= after


# ============================================================================
# Client ID Extraction Tests
# ============================================================================


class TestClientIdExtraction:
    """Tests for client ID extraction from requests."""

    def test_get_client_id_from_direct_ip(self, rate_limiter, mock_request):
        """Test extracting client ID from direct client IP."""
        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "192.168.1.100"

    def test_get_client_id_from_x_forwarded_for(self, rate_limiter, mock_request_with_forwarded):
        """Test extracting client ID from X-Forwarded-For header."""
        client_id = rate_limiter._get_client_id(mock_request_with_forwarded)
        # Should use the first IP in the X-Forwarded-For chain
        assert client_id == "10.0.0.1"

    def test_get_client_id_with_no_client(self, rate_limiter, mock_request_no_client):
        """Test extracting client ID when client info is missing."""
        client_id = rate_limiter._get_client_id(mock_request_no_client)
        assert client_id == "unknown"

    def test_get_client_id_x_forwarded_for_with_spaces(self, rate_limiter):
        """Test X-Forwarded-For header with extra whitespace."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "  10.0.0.5  ,  192.168.1.1  "}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        client_id = rate_limiter._get_client_id(request)
        assert client_id == "10.0.0.5"

    def test_get_client_id_single_forwarded_ip(self, rate_limiter):
        """Test X-Forwarded-For with single IP."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "203.0.113.50"}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        client_id = rate_limiter._get_client_id(request)
        assert client_id == "203.0.113.50"


# ============================================================================
# Rate Limit Enforcement Tests
# ============================================================================


class TestRateLimitEnforcement:
    """Tests for rate limit enforcement logic."""

    def test_allows_requests_within_limit(self, rate_limiter, mock_request):
        """Test that requests within limit are allowed."""
        # Config allows 5 requests per window
        for i in range(5):
            result = rate_limiter.check_rate_limit(mock_request)
            assert result is True, f"Request {i + 1} should be allowed"

    def test_blocks_requests_exceeding_limit(self, rate_limiter, mock_request):
        """Test that requests exceeding limit are blocked."""
        # Exhaust the limit (5 requests)
        for _ in range(5):
            rate_limiter.check_rate_limit(mock_request)

        # 6th request should be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            rate_limiter.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

    def test_rate_limit_includes_retry_after(self, rate_limiter, mock_request):
        """Test that RateLimitExceeded includes Retry-After header."""
        # Exhaust the limit
        for _ in range(5):
            rate_limiter.check_rate_limit(mock_request)

        # Check the exception
        with pytest.raises(RateLimitExceeded) as exc_info:
            rate_limiter.check_rate_limit(mock_request)

        assert "Retry-After" in exc_info.value.headers
        retry_after = int(exc_info.value.headers["Retry-After"])
        assert retry_after > 0

    def test_rate_limit_disabled_allows_all(self, rate_limit_disabled_config, mock_request):
        """Test that disabled rate limiting allows all requests."""
        limiter = RateLimiter(config=rate_limit_disabled_config)

        # Should allow many more requests than the limit
        for _ in range(100):
            result = limiter.check_rate_limit(mock_request)
            assert result is True

    def test_different_clients_have_separate_limits(self, rate_limiter):
        """Test that different clients have independent rate limits."""
        request1 = MagicMock()
        request1.headers = {}
        request1.client = MagicMock()
        request1.client.host = "192.168.1.1"

        request2 = MagicMock()
        request2.headers = {}
        request2.client = MagicMock()
        request2.client.host = "192.168.1.2"

        # Exhaust limit for client 1
        for _ in range(5):
            rate_limiter.check_rate_limit(request1)

        # Client 1 should be blocked
        with pytest.raises(RateLimitExceeded):
            rate_limiter.check_rate_limit(request1)

        # Client 2 should still be allowed
        result = rate_limiter.check_rate_limit(request2)
        assert result is True

    def test_requests_track_timestamps(self, rate_limiter, mock_request):
        """Test that requests are tracked with timestamps."""
        before = time.time()
        rate_limiter.check_rate_limit(mock_request)
        after = time.time()

        client_id = rate_limiter._get_client_id(mock_request)
        assert len(rate_limiter._requests[client_id]) == 1
        timestamp = rate_limiter._requests[client_id][0]
        assert before <= timestamp <= after


# ============================================================================
# Request Cleanup Tests
# ============================================================================


class TestRequestCleanup:
    """Tests for cleanup of expired request entries."""

    def test_cleanup_removes_expired_requests(self, tight_rate_limit_config):
        """Test that expired requests are cleaned up."""
        limiter = RateLimiter(config=tight_rate_limit_config)
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Make requests
        limiter.check_rate_limit(request)
        limiter.check_rate_limit(request)

        # Should be at limit now (2 requests)
        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limit(request)

        # Wait for window to expire
        time.sleep(1.1)

        # Force cleanup by setting last cleanup time in the past
        limiter._last_cleanup = time.time() - 61

        # Now request should be allowed again
        result = limiter.check_rate_limit(request)
        assert result is True

    def test_cleanup_removes_empty_client_entries(self, monkeypatch):
        """Test that clients with no recent requests are removed."""
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "1")
        config = SecurityConfig()
        limiter = RateLimiter(config=config)

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Make a request
        limiter.check_rate_limit(request)
        client_id = limiter._get_client_id(request)
        assert client_id in limiter._requests

        # Wait for window to expire
        time.sleep(1.1)

        # Force cleanup
        limiter._last_cleanup = time.time() - 61
        limiter._cleanup_old_requests()

        # Client entry should be removed
        assert client_id not in limiter._requests

    def test_cleanup_respects_interval(self, rate_limiter, mock_request):
        """Test that cleanup only runs at configured interval."""
        # Make a request to populate data
        rate_limiter.check_rate_limit(mock_request)

        # Set cleanup time to recent past (within interval)
        rate_limiter._last_cleanup = time.time() - 30

        # Add an expired timestamp manually
        client_id = rate_limiter._get_client_id(mock_request)
        rate_limiter._requests[client_id].insert(0, time.time() - 120)

        # Cleanup should not run yet
        rate_limiter._cleanup_old_requests()

        # Old entry should still be there (cleanup didn't run)
        assert len(rate_limiter._requests[client_id]) == 2


# ============================================================================
# Get Remaining Tests
# ============================================================================


class TestGetRemaining:
    """Tests for get_remaining() functionality."""

    def test_get_remaining_full_quota(self, rate_limiter, mock_request):
        """Test get_remaining returns full quota for new client."""
        remaining = rate_limiter.get_remaining(mock_request)
        assert remaining == 5

    def test_get_remaining_after_requests(self, rate_limiter, mock_request):
        """Test get_remaining decreases after requests."""
        # Make 3 requests
        for _ in range(3):
            rate_limiter.check_rate_limit(mock_request)

        remaining = rate_limiter.get_remaining(mock_request)
        assert remaining == 2

    def test_get_remaining_at_limit(self, rate_limiter, mock_request):
        """Test get_remaining returns 0 at limit."""
        # Exhaust the limit
        for _ in range(5):
            rate_limiter.check_rate_limit(mock_request)

        remaining = rate_limiter.get_remaining(mock_request)
        assert remaining == 0

    def test_get_remaining_never_negative(self, rate_limiter, mock_request):
        """Test get_remaining never returns negative value."""
        # Exhaust the limit
        for _ in range(5):
            rate_limiter.check_rate_limit(mock_request)

        # Try to exceed (will raise exception but we catch it)
        try:
            rate_limiter.check_rate_limit(mock_request)
        except RateLimitExceeded:
            pass

        remaining = rate_limiter.get_remaining(mock_request)
        assert remaining == 0

    def test_get_remaining_excludes_expired(self, tight_rate_limit_config):
        """Test get_remaining excludes expired requests."""
        limiter = RateLimiter(config=tight_rate_limit_config)
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Make requests
        limiter.check_rate_limit(request)
        limiter.check_rate_limit(request)

        # Should have 0 remaining
        assert limiter.get_remaining(request) == 0

        # Wait for window to expire
        time.sleep(1.1)

        # Should have full quota again
        assert limiter.get_remaining(request) == 2


# ============================================================================
# FastAPI Dependency Tests
# ============================================================================


class TestFastAPIDependency:
    """Tests for RateLimiter as FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_callable_allows_within_limit(self, rate_limiter, mock_request):
        """Test __call__ allows requests within limit."""
        result = await rate_limiter(mock_request)
        assert result is True

    @pytest.mark.asyncio
    async def test_callable_blocks_exceeding_limit(self, rate_limiter, mock_request):
        """Test __call__ blocks requests exceeding limit."""
        # Exhaust limit
        for _ in range(5):
            await rate_limiter(mock_request)

        # Should raise on next request
        with pytest.raises(RateLimitExceeded):
            await rate_limiter(mock_request)

    @pytest.mark.asyncio
    async def test_callable_is_async(self, rate_limiter, mock_request):
        """Test that __call__ is awaitable."""
        import asyncio

        result = rate_limiter(mock_request)
        assert asyncio.iscoroutine(result)
        # Clean up the coroutine
        await result


# ============================================================================
# RateLimitExceeded Exception Tests
# ============================================================================


class TestRateLimitExceededException:
    """Tests for RateLimitExceeded exception."""

    def test_exception_status_code(self):
        """Test RateLimitExceeded has correct status code."""
        exc = RateLimitExceeded(retry_after=30)
        assert exc.status_code == 429

    def test_exception_detail_message(self):
        """Test RateLimitExceeded has descriptive message."""
        exc = RateLimitExceeded(retry_after=45)
        assert "Rate limit exceeded" in exc.detail
        assert "45" in exc.detail

    def test_exception_retry_after_header(self):
        """Test RateLimitExceeded includes Retry-After header."""
        exc = RateLimitExceeded(retry_after=60)
        assert exc.headers["Retry-After"] == "60"

    def test_exception_different_retry_values(self):
        """Test RateLimitExceeded with various retry values."""
        for retry_after in [1, 30, 60, 300]:
            exc = RateLimitExceeded(retry_after=retry_after)
            assert exc.headers["Retry-After"] == str(retry_after)


# ============================================================================
# Edge Cases and Concurrent Scenarios
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_x_forwarded_for(self, rate_limiter):
        """Test handling of empty X-Forwarded-For header."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": ""}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Empty string is falsy, should fall back to client host
        client_id = rate_limiter._get_client_id(request)
        assert client_id == "192.168.1.100"

    def test_rate_limit_boundary(self, rate_limiter, mock_request):
        """Test exact boundary of rate limit."""
        # Make exactly limit - 1 requests
        for _ in range(4):
            result = rate_limiter.check_rate_limit(mock_request)
            assert result is True

        # Last allowed request
        result = rate_limiter.check_rate_limit(mock_request)
        assert result is True

        # First blocked request
        with pytest.raises(RateLimitExceeded):
            rate_limiter.check_rate_limit(mock_request)

    def test_multiple_clients_concurrent(self, rate_limiter):
        """Test multiple clients making requests concurrently."""
        clients = []
        for i in range(10):
            request = MagicMock()
            request.headers = {}
            request.client = MagicMock()
            request.client.host = f"192.168.1.{i}"
            clients.append(request)

        # Each client makes 3 requests
        for client in clients:
            for _ in range(3):
                result = rate_limiter.check_rate_limit(client)
                assert result is True

        # All clients should have 2 remaining
        for client in clients:
            assert rate_limiter.get_remaining(client) == 2

    def test_rate_limit_window_expiry(self, tight_rate_limit_config):
        """Test that rate limit resets after window expires."""
        limiter = RateLimiter(config=tight_rate_limit_config)
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Exhaust limit
        limiter.check_rate_limit(request)
        limiter.check_rate_limit(request)

        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limit(request)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        result = limiter.check_rate_limit(request)
        assert result is True

    def test_ipv6_client(self, rate_limiter):
        """Test rate limiting with IPv6 client address."""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

        result = rate_limiter.check_rate_limit(request)
        assert result is True

        client_id = rate_limiter._get_client_id(request)
        assert client_id == "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

    def test_localhost_client(self, rate_limiter):
        """Test rate limiting for localhost."""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        for _ in range(5):
            result = rate_limiter.check_rate_limit(request)
            assert result is True

        with pytest.raises(RateLimitExceeded):
            rate_limiter.check_rate_limit(request)


# ============================================================================
# Configuration Validation Tests
# ============================================================================


class TestConfigurationValidation:
    """Tests for rate limiter configuration handling."""

    def test_high_rate_limit(self, monkeypatch):
        """Test with high rate limit configuration."""
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10000")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
        config = SecurityConfig()
        limiter = RateLimiter(config=config)

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Should allow many requests
        for _ in range(100):
            result = limiter.check_rate_limit(request)
            assert result is True

    def test_short_window(self, monkeypatch):
        """Test with very short time window."""
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "1")
        config = SecurityConfig()
        limiter = RateLimiter(config=config)

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        # Quick burst
        limiter.check_rate_limit(request)
        limiter.check_rate_limit(request)

        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limit(request)

        # Wait for window
        time.sleep(1.1)

        # Should work again
        result = limiter.check_rate_limit(request)
        assert result is True

    def test_default_config_values(self):
        """Test RateLimiter with default config values."""
        limiter = RateLimiter()
        # Default should have rate_limit_requests and rate_limit_window_seconds set
        assert limiter.config.rate_limit_requests > 0
        assert limiter.config.rate_limit_window_seconds > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
