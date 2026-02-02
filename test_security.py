#!/usr/bin/env python3
"""
Tests for the security module.

Run with: pytest test_security.py -v
"""

import os
import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from security import (
    APIKeyAuth,
    InputValidator,
    JWTAuth,
    PathTraversalError,
    RateLimiter,
    RateLimitExceeded,
    SecurityConfig,
    SecurityHeaders,
    SessionOwnership,
    require_rate_limit,
    security_headers_middleware,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def security_config_with_keys():
    """Create a security config with valid API keys."""
    os.environ["VALID_API_KEYS"] = "test-key-1,test-key-2"
    os.environ["ENABLE_API_KEY_AUTH"] = "true"
    config = SecurityConfig()
    yield config
    del os.environ["VALID_API_KEYS"]
    del os.environ["ENABLE_API_KEY_AUTH"]


@pytest.fixture
def security_config_jwt():
    """Create a security config with JWT enabled."""
    os.environ["ENABLE_JWT_AUTH"] = "true"
    os.environ["JWT_SECRET"] = "test-secret-key-for-testing"
    config = SecurityConfig()
    yield config
    del os.environ["ENABLE_JWT_AUTH"]
    del os.environ["JWT_SECRET"]


# ============================================================================
# SecurityConfig Tests
# ============================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SecurityConfig()
        assert config.api_key_header_name == "X-API-Key"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration_hours == 24
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window_seconds == 60

    def test_env_override(self):
        """Test environment variable overrides."""
        os.environ["API_KEY_HEADER"] = "X-Custom-Key"
        os.environ["RATE_LIMIT_REQUESTS"] = "50"
        try:
            config = SecurityConfig()
            assert config.api_key_header_name == "X-Custom-Key"
            assert config.rate_limit_requests == 50
        finally:
            del os.environ["API_KEY_HEADER"]
            del os.environ["RATE_LIMIT_REQUESTS"]

    def test_api_keys_parsing(self):
        """Test parsing of comma-separated API keys."""
        os.environ["VALID_API_KEYS"] = "key1, key2 , key3"
        try:
            config = SecurityConfig()
            assert "key1" in config.valid_api_keys
            assert "key2" in config.valid_api_keys
            assert "key3" in config.valid_api_keys
        finally:
            del os.environ["VALID_API_KEYS"]


# ============================================================================
# APIKeyAuth Tests
# ============================================================================


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_generate_api_key(self):
        """Test API key generation."""
        key = APIKeyAuth.generate_api_key()
        assert len(key) > 20
        assert isinstance(key, str)

    def test_validate_key_success(self, security_config_with_keys):
        """Test successful key validation."""
        auth = APIKeyAuth(config=security_config_with_keys)
        assert auth.validate_key("test-key-1") is True
        assert auth.validate_key("test-key-2") is True

    def test_validate_key_failure(self, security_config_with_keys):
        """Test failed key validation."""
        auth = APIKeyAuth(config=security_config_with_keys)
        assert auth.validate_key("invalid-key") is False
        assert auth.validate_key("") is False
        assert auth.validate_key(None) is False

    def test_add_remove_key(self):
        """Test adding and removing keys."""
        auth = APIKeyAuth()
        new_key = "my-new-key"

        auth.add_key(new_key)
        assert auth.validate_key(new_key) is True

        auth.remove_key(new_key)
        assert auth.validate_key(new_key) is False

    @pytest.mark.asyncio
    async def test_call_missing_key(self, mock_request, security_config_with_keys):
        """Test authentication failure when key is missing."""
        auth = APIKeyAuth(config=security_config_with_keys)
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_call_invalid_key(self, mock_request, security_config_with_keys):
        """Test authentication failure with invalid key."""
        auth = APIKeyAuth(config=security_config_with_keys)
        mock_request.headers = {"X-API-Key": "wrong-key"}

        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_call_valid_key(self, mock_request, security_config_with_keys):
        """Test successful authentication with valid key."""
        auth = APIKeyAuth(config=security_config_with_keys)
        mock_request.headers = {"X-API-Key": "test-key-1"}

        result = await auth(mock_request)
        assert result == "test-key-1"

    @pytest.mark.asyncio
    async def test_call_disabled(self, mock_request):
        """Test that auth is skipped when disabled."""
        os.environ["ENABLE_API_KEY_AUTH"] = "false"
        try:
            config = SecurityConfig()
            auth = APIKeyAuth(config=config)
            result = await auth(mock_request)
            assert result is None
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]


# ============================================================================
# JWTAuth Tests
# ============================================================================


class TestJWTAuth:
    """Tests for JWT authentication."""

    def test_create_token(self, security_config_jwt):
        """Test JWT token creation."""
        auth = JWTAuth(config=security_config_jwt)
        token = auth.create_token(user_id="user123", roles=["admin"])

        assert token is not None
        assert len(token.split(".")) == 3

    def test_verify_token_success(self, security_config_jwt):
        """Test successful token verification."""
        auth = JWTAuth(config=security_config_jwt)
        token = auth.create_token(
            user_id="user123", roles=["admin", "user"], session_ids=["session-1"]
        )

        payload = auth.verify_token(token)
        assert payload is not None
        assert payload.sub == "user123"
        assert "admin" in payload.roles
        assert "user" in payload.roles
        assert "session-1" in payload.session_ids

    def test_verify_token_invalid(self, security_config_jwt):
        """Test verification of invalid token."""
        auth = JWTAuth(config=security_config_jwt)

        assert auth.verify_token("invalid.token.here") is None
        assert auth.verify_token("") is None
        assert auth.verify_token(None) is None

    def test_verify_token_tampered(self, security_config_jwt):
        """Test verification of tampered token."""
        auth = JWTAuth(config=security_config_jwt)
        token = auth.create_token(user_id="user123")

        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1][:-4] + "XXXX"  # Modify payload
        tampered = ".".join(parts)

        assert auth.verify_token(tampered) is None

    def test_verify_token_expired(self, security_config_jwt):
        """Test verification of expired token."""
        auth = JWTAuth(config=security_config_jwt)
        # Create token that expires immediately
        token = auth.create_token(user_id="user123", expires_in_hours=0)

        # Wait a tiny bit
        time.sleep(0.1)

        # Should be expired
        assert auth.verify_token(token) is None

    def test_revoke_token(self, security_config_jwt):
        """Test token revocation."""
        auth = JWTAuth(config=security_config_jwt)
        token = auth.create_token(user_id="user123")

        # Token should be valid initially
        assert auth.verify_token(token) is not None

        # Revoke token
        auth.revoke_token(token)

        # Token should no longer be valid
        assert auth.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_call_missing_token(self, mock_request, security_config_jwt):
        """Test authentication failure when token is missing."""
        auth = JWTAuth(config=security_config_jwt)
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_call_invalid_token(self, mock_request, security_config_jwt):
        """Test authentication failure with invalid token."""
        auth = JWTAuth(config=security_config_jwt)
        mock_request.headers = {"Authorization": "Bearer invalid-token"}

        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_call_valid_token(self, mock_request, security_config_jwt):
        """Test successful authentication with valid token."""
        auth = JWTAuth(config=security_config_jwt)
        token = auth.create_token(user_id="user123")
        mock_request.headers = {"Authorization": f"Bearer {token}"}

        payload = await auth(mock_request)
        assert payload.sub == "user123"


# ============================================================================
# RateLimiter Tests
# ============================================================================


class TestRateLimiter:
    """Tests for rate limiting."""

    def test_check_rate_limit_allowed(self, mock_request):
        """Test that requests within limit are allowed."""
        os.environ["RATE_LIMIT_REQUESTS"] = "10"
        os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"
        os.environ["ENABLE_RATE_LIMITING"] = "true"
        try:
            config = SecurityConfig()
            limiter = RateLimiter(config=config)

            # First request should be allowed
            assert limiter.check_rate_limit(mock_request) is True
        finally:
            del os.environ["RATE_LIMIT_REQUESTS"]
            del os.environ["RATE_LIMIT_WINDOW_SECONDS"]
            del os.environ["ENABLE_RATE_LIMITING"]

    def test_check_rate_limit_exceeded(self, mock_request):
        """Test that rate limit is enforced."""
        os.environ["RATE_LIMIT_REQUESTS"] = "5"
        os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"
        os.environ["ENABLE_RATE_LIMITING"] = "true"
        try:
            config = SecurityConfig()
            limiter = RateLimiter(config=config)

            # Make requests up to the limit
            for _ in range(5):
                limiter.check_rate_limit(mock_request)

            # Next request should be rate limited
            with pytest.raises(RateLimitExceeded) as exc_info:
                limiter.check_rate_limit(mock_request)
            assert exc_info.value.status_code == 429
        finally:
            del os.environ["RATE_LIMIT_REQUESTS"]
            del os.environ["RATE_LIMIT_WINDOW_SECONDS"]
            del os.environ["ENABLE_RATE_LIMITING"]

    def test_get_remaining(self, mock_request):
        """Test getting remaining requests."""
        os.environ["RATE_LIMIT_REQUESTS"] = "10"
        os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"
        os.environ["ENABLE_RATE_LIMITING"] = "true"
        try:
            config = SecurityConfig()
            limiter = RateLimiter(config=config)

            assert limiter.get_remaining(mock_request) == 10

            limiter.check_rate_limit(mock_request)
            assert limiter.get_remaining(mock_request) == 9

            limiter.check_rate_limit(mock_request)
            assert limiter.get_remaining(mock_request) == 8
        finally:
            del os.environ["RATE_LIMIT_REQUESTS"]
            del os.environ["RATE_LIMIT_WINDOW_SECONDS"]
            del os.environ["ENABLE_RATE_LIMITING"]

    def test_rate_limit_disabled(self, mock_request):
        """Test that rate limiting can be disabled."""
        os.environ["ENABLE_RATE_LIMITING"] = "false"
        try:
            config = SecurityConfig()
            limiter = RateLimiter(config=config)

            # Should always return True when disabled
            for _ in range(1000):
                assert limiter.check_rate_limit(mock_request) is True
        finally:
            del os.environ["ENABLE_RATE_LIMITING"]

    def test_client_id_from_forwarded_header(self):
        """Test client ID extraction from X-Forwarded-For header."""
        request = MagicMock(spec=Request)
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        limiter = RateLimiter()
        client_id = limiter._get_client_id(request)
        assert client_id == "192.168.1.1"


# ============================================================================
# InputValidator Tests
# ============================================================================


class TestInputValidator:
    """Tests for input validation."""

    def test_validate_path_success(self):
        """Test successful path validation."""
        validator = InputValidator()
        assert validator.validate_path("/home/user/project") == "/home/user/project"
        assert validator.validate_path("relative/path") == "relative/path"
        assert validator.validate_path("./relative/path") == "relative/path"

    def test_validate_path_traversal(self):
        """Test path traversal detection."""
        validator = InputValidator()

        with pytest.raises(PathTraversalError):
            validator.validate_path("/home/user/../etc/passwd")

        with pytest.raises(PathTraversalError):
            validator.validate_path("../../../etc/passwd")

    def test_validate_path_with_base(self, tmp_path):
        """Test path validation with base path restriction."""
        validator = InputValidator()
        base = str(tmp_path)

        # Valid path within base
        valid_path = str(tmp_path / "subdir" / "file.txt")
        result = validator.validate_path(valid_path, base_path=base)
        assert base in result

        # Path outside base should fail
        with pytest.raises(PathTraversalError):
            validator.validate_path("/etc/passwd", base_path=base)

    def test_validate_path_empty(self):
        """Test empty path validation."""
        validator = InputValidator()

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_path("")
        assert exc_info.value.status_code == 400

    def test_sanitize_string(self):
        """Test string sanitization."""
        validator = InputValidator()

        # Normal string should pass through
        assert validator.sanitize_string("Hello World") == "Hello World"

        # Null bytes should be removed
        assert validator.sanitize_string("Hello\x00World") == "HelloWorld"

        # XSS should be removed
        result = validator.sanitize_string("Hello <script>alert('xss')</script>")
        assert "<script" not in result.lower()

    def test_sanitize_string_length(self):
        """Test string length truncation."""
        validator = InputValidator(max_string_length=10)
        result = validator.sanitize_string("This is a very long string")
        assert len(result) == 10

    def test_sanitize_string_html_allowed(self):
        """Test string sanitization with HTML allowed."""
        validator = InputValidator()
        result = validator.sanitize_string("<b>Bold</b>", allow_html=True)
        assert "<b>" in result

    def test_validate_query_success(self):
        """Test successful query validation."""
        validator = InputValidator()
        assert validator.validate_query("Set up MLOps pipeline") == "Set up MLOps pipeline"
        assert validator.validate_query("  trimmed  ") == "trimmed"

    def test_validate_query_empty(self):
        """Test empty query validation."""
        validator = InputValidator()

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_query("")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_query("   ")
        assert exc_info.value.status_code == 400

    def test_validate_session_id_success(self):
        """Test successful session ID validation."""
        validator = InputValidator()
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = validator.validate_session_id(valid_uuid)
        assert result == valid_uuid

    def test_validate_session_id_invalid(self):
        """Test invalid session ID validation."""
        validator = InputValidator()

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_session_id("not-a-uuid")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            validator.validate_session_id("")
        assert exc_info.value.status_code == 400


# ============================================================================
# SessionOwnership Tests
# ============================================================================


class TestSessionOwnership:
    """Tests for session ownership."""

    def test_register_session(self):
        """Test session registration."""
        ownership = SessionOwnership()
        ownership.register_session("session-1", "user-1")

        assert ownership.get_owner("session-1") == "user-1"
        assert "session-1" in ownership.get_user_sessions("user-1")

    def test_unregister_session(self):
        """Test session unregistration."""
        ownership = SessionOwnership()
        ownership.register_session("session-1", "user-1")
        ownership.unregister_session("session-1")

        assert ownership.get_owner("session-1") is None
        assert "session-1" not in ownership.get_user_sessions("user-1")

    def test_check_access_owner(self):
        """Test access check for owner."""
        ownership = SessionOwnership()
        ownership.register_session("session-1", "user-1")

        assert ownership.check_access("session-1", "user-1") is True
        assert ownership.check_access("session-1", "user-2") is False

    def test_check_access_untracked(self):
        """Test access check for untracked session."""
        ownership = SessionOwnership()

        # Untracked sessions allow public access
        assert ownership.check_access("unknown-session", "any-user") is True

    def test_verify_access_success(self):
        """Test verify access success."""
        ownership = SessionOwnership()
        ownership.register_session("session-1", "user-1")

        # Should not raise
        ownership.verify_access("session-1", "user-1")

    def test_verify_access_denied(self):
        """Test verify access denial."""
        ownership = SessionOwnership()
        ownership.register_session("session-1", "user-1")

        with pytest.raises(HTTPException) as exc_info:
            ownership.verify_access("session-1", "user-2")
        assert exc_info.value.status_code == 403

    def test_get_user_sessions(self):
        """Test getting all sessions for a user."""
        ownership = SessionOwnership()
        ownership.register_session("session-1", "user-1")
        ownership.register_session("session-2", "user-1")
        ownership.register_session("session-3", "user-2")

        user1_sessions = ownership.get_user_sessions("user-1")
        assert len(user1_sessions) == 2
        assert "session-1" in user1_sessions
        assert "session-2" in user1_sessions
        assert "session-3" not in user1_sessions


# ============================================================================
# SecurityHeaders Tests
# ============================================================================


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_default_headers(self):
        """Test default security headers."""
        headers = SecurityHeaders()
        h = headers.get_headers()

        assert "X-Content-Type-Options" in h
        assert h["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in h
        assert "Strict-Transport-Security" in h
        assert "Content-Security-Policy" in h

    def test_custom_headers(self):
        """Test custom headers."""
        headers = SecurityHeaders(custom_headers={"X-Custom-Header": "custom-value"})
        h = headers.get_headers()

        assert h["X-Custom-Header"] == "custom-value"
        # Default headers should still be present
        assert "X-Content-Type-Options" in h

    def test_header_override(self):
        """Test overriding default headers."""
        headers = SecurityHeaders(custom_headers={"X-Frame-Options": "SAMEORIGIN"})
        h = headers.get_headers()

        assert h["X-Frame-Options"] == "SAMEORIGIN"


# ============================================================================
# Middleware Tests
# ============================================================================


class TestMiddleware:
    """Tests for security middleware."""

    @pytest.mark.asyncio
    async def test_security_headers_middleware(self, mock_request):
        """Test security headers middleware."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {}

        async def mock_call_next(request):
            return mock_response

        response = await security_headers_middleware(mock_request, mock_call_next)

        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers


# ============================================================================
# Decorator Tests
# ============================================================================


class TestDecorators:
    """Tests for security decorators."""

    @pytest.mark.asyncio
    async def test_require_rate_limit_decorator(self):
        """Test require_rate_limit decorator."""
        os.environ["ENABLE_RATE_LIMITING"] = "true"
        os.environ["RATE_LIMIT_REQUESTS"] = "5"
        os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"
        try:
            config = SecurityConfig()
            limiter = RateLimiter(config=config)

            # Create a fresh mock request with unique client
            test_request = MagicMock(spec=Request)
            test_request.headers = {}
            test_request.client = MagicMock()
            test_request.client.host = "192.168.99.99"  # Unique IP

            @require_rate_limit(rate_limiter=limiter)
            async def test_endpoint(request: Request):
                return {"status": "ok"}

            # First few requests should work
            for _ in range(5):
                result = await test_endpoint(request=test_request)
                assert result == {"status": "ok"}

            # Next request should be rate limited
            with pytest.raises(RateLimitExceeded):
                await test_endpoint(request=test_request)
        finally:
            del os.environ["ENABLE_RATE_LIMITING"]
            del os.environ["RATE_LIMIT_REQUESTS"]
            del os.environ["RATE_LIMIT_WINDOW_SECONDS"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple security components."""

    @pytest.mark.asyncio
    async def test_full_auth_flow(self, mock_request, security_config_jwt):
        """Test complete authentication flow."""
        jwt_auth = JWTAuth(config=security_config_jwt)

        # Create token with session access
        token = jwt_auth.create_token(
            user_id="user-123", roles=["user"], session_ids=["session-abc"]
        )

        # Verify token
        payload = jwt_auth.verify_token(token)
        assert payload is not None
        assert payload.sub == "user-123"

        # Check session ownership
        ownership = SessionOwnership()
        ownership.register_session("session-abc", "user-123")

        # User should have access to their session
        assert ownership.check_access("session-abc", payload.sub) is True

        # But not to another user's session
        ownership.register_session("session-xyz", "user-456")
        assert ownership.check_access("session-xyz", payload.sub) is False

    def test_input_validation_pipeline(self):
        """Test input validation pipeline."""
        validator = InputValidator()

        # Validate query
        query = validator.validate_query("  Set up MLOps pipeline  ")
        assert query == "Set up MLOps pipeline"

        # Sanitize query
        sanitized = validator.sanitize_string(query)
        assert sanitized == query

        # Validate path
        path = validator.validate_path("/home/user/ml-project")
        assert path == "/home/user/ml-project"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
