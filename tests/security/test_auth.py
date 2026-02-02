#!/usr/bin/env python3
"""
Unit tests for security authentication components.

Tests for:
- APIKeyAuth: API key-based authentication
- JWTAuth: JWT token authentication
- RateLimiter: Request rate limiting
- InputValidator: Input sanitization
- SessionOwnership: Session access control
- APIKeyManager: API key lifecycle management
- Middleware: get_current_user dependency

Run with: pytest tests/security/test_auth.py -v
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from security import (
    APIKeyAuth,
    InputValidator,
    JWTAuth,
    JWTPayload,
    PathTraversalError,
    RateLimiter,
    RateLimitExceeded,
    SecurityConfig,
    SecurityHeaders,
    SessionOwnership,
)
from security.api_keys import APIKeyInfo, APIKeyManager, GeneratedKey
from security.middleware import (
    AuthenticationError,
    AuthorizationError,
    CurrentUser,
)

# ============================================================================
# SecurityConfig Tests
# ============================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig class."""

    def test_default_config_values(self, monkeypatch):
        """Test SecurityConfig loads default values when env vars not set."""
        monkeypatch.delenv("API_KEY_HEADER", raising=False)
        monkeypatch.delenv("JWT_EXPIRATION_HOURS", raising=False)
        monkeypatch.delenv("RATE_LIMIT_REQUESTS", raising=False)
        monkeypatch.delenv("ENABLE_API_KEY_AUTH", raising=False)

        config = SecurityConfig()

        assert config.api_key_header_name == "X-API-Key"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration_hours == 24
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window_seconds == 60
        assert config.enable_api_key_auth is False
        assert config.enable_jwt_auth is False
        assert config.enable_rate_limiting is True

    def test_config_from_env_vars(self, monkeypatch):
        """Test SecurityConfig loads values from environment variables."""
        monkeypatch.setenv("API_KEY_HEADER", "X-Custom-Key")
        monkeypatch.setenv("JWT_EXPIRATION_HOURS", "48")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "200")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "120")
        monkeypatch.setenv("ENABLE_API_KEY_AUTH", "true")
        monkeypatch.setenv("ENABLE_JWT_AUTH", "true")
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "false")

        config = SecurityConfig()

        assert config.api_key_header_name == "X-Custom-Key"
        assert config.jwt_expiration_hours == 48
        assert config.rate_limit_requests == 200
        assert config.rate_limit_window_seconds == 120
        assert config.enable_api_key_auth is True
        assert config.enable_jwt_auth is True
        assert config.enable_rate_limiting is False

    def test_valid_api_keys_parsing(self, monkeypatch):
        """Test VALID_API_KEYS environment variable parsing."""
        monkeypatch.setenv("VALID_API_KEYS", "key1, key2, key3")

        config = SecurityConfig()

        assert len(config.valid_api_keys) == 3
        assert "key1" in config.valid_api_keys
        assert "key2" in config.valid_api_keys
        assert "key3" in config.valid_api_keys

    def test_empty_valid_api_keys(self, monkeypatch):
        """Test empty VALID_API_KEYS returns empty set."""
        monkeypatch.setenv("VALID_API_KEYS", "")

        config = SecurityConfig()

        assert config.valid_api_keys == set()

    def test_allowed_origins_parsing(self, monkeypatch):
        """Test ALLOWED_ORIGINS environment variable parsing."""
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,https://example.com")

        config = SecurityConfig()

        assert len(config.allowed_origins) == 2
        assert "http://localhost:3000" in config.allowed_origins
        assert "https://example.com" in config.allowed_origins


# ============================================================================
# APIKeyAuth Tests
# ============================================================================


class TestAPIKeyAuth:
    """Tests for APIKeyAuth class."""

    @pytest.fixture
    def config_with_keys(self, monkeypatch):
        """Create config with valid API keys."""
        monkeypatch.setenv("VALID_API_KEYS", "test-key-123,another-key-456")
        monkeypatch.setenv("ENABLE_API_KEY_AUTH", "true")
        return SecurityConfig()

    @pytest.fixture
    def api_key_auth(self, config_with_keys):
        """Create APIKeyAuth instance with test keys."""
        return APIKeyAuth(config=config_with_keys)

    def test_generate_api_key(self):
        """Test API key generation produces valid format."""
        key = APIKeyAuth.generate_api_key()

        assert isinstance(key, str)
        assert len(key) == 43  # Base64url encoding of 32 bytes

    def test_generate_unique_keys(self):
        """Test generated keys are unique."""
        keys = [APIKeyAuth.generate_api_key() for _ in range(100)]

        assert len(set(keys)) == 100

    def test_validate_key_valid(self, api_key_auth):
        """Test validating a valid API key."""
        assert api_key_auth.validate_key("test-key-123") is True

    def test_validate_key_invalid(self, api_key_auth):
        """Test validating an invalid API key."""
        assert api_key_auth.validate_key("invalid-key") is False

    def test_validate_key_empty(self, api_key_auth):
        """Test validating empty API key."""
        assert api_key_auth.validate_key("") is False
        assert api_key_auth.validate_key(None) is False

    def test_add_key(self, api_key_auth):
        """Test adding a new API key."""
        new_key = "new-dynamic-key"
        api_key_auth.add_key(new_key)

        assert api_key_auth.validate_key(new_key) is True

    def test_remove_key(self, api_key_auth):
        """Test removing an API key."""
        api_key_auth.remove_key("test-key-123")

        assert api_key_auth.validate_key("test-key-123") is False

    def test_hash_key_consistency(self):
        """Test key hashing is consistent."""
        key = "test-key"
        hash1 = APIKeyAuth._hash_key(key)
        hash2 = APIKeyAuth._hash_key(key)

        assert hash1 == hash2

    def test_hash_key_different_keys(self):
        """Test different keys produce different hashes."""
        hash1 = APIKeyAuth._hash_key("key1")
        hash2 = APIKeyAuth._hash_key("key2")

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_call_missing_key_raises(self, api_key_auth):
        """Test __call__ raises HTTPException when key is missing."""
        request = MagicMock()
        request.headers.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await api_key_auth(request)

        assert exc_info.value.status_code == 401
        assert "API key required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_invalid_key_raises(self, api_key_auth):
        """Test __call__ raises HTTPException when key is invalid."""
        request = MagicMock()
        request.headers.get.return_value = "invalid-key"

        with pytest.raises(HTTPException) as exc_info:
            await api_key_auth(request)

        assert exc_info.value.status_code == 403
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_valid_key_returns_key(self, api_key_auth):
        """Test __call__ returns API key when valid."""
        request = MagicMock()
        request.headers.get.return_value = "test-key-123"

        result = await api_key_auth(request)

        assert result == "test-key-123"

    @pytest.mark.asyncio
    async def test_call_auth_disabled_returns_none(self, monkeypatch):
        """Test __call__ returns None when auth is disabled."""
        monkeypatch.setenv("ENABLE_API_KEY_AUTH", "false")
        config = SecurityConfig()
        auth = APIKeyAuth(config=config)

        request = MagicMock()
        request.headers.get.return_value = None

        result = await auth(request)

        assert result is None


# ============================================================================
# JWTAuth Tests
# ============================================================================


class TestJWTAuth:
    """Tests for JWTAuth class."""

    @pytest.fixture
    def jwt_config(self, monkeypatch):
        """Create config for JWT testing."""
        monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-testing")
        monkeypatch.setenv("JWT_EXPIRATION_HOURS", "24")
        monkeypatch.setenv("ENABLE_JWT_AUTH", "true")
        return SecurityConfig()

    @pytest.fixture
    def jwt_auth(self, jwt_config):
        """Create JWTAuth instance for testing."""
        return JWTAuth(config=jwt_config)

    def test_create_token(self, jwt_auth):
        """Test creating a JWT token."""
        token = jwt_auth.create_token(user_id="user123")

        assert isinstance(token, str)
        assert token.count(".") == 2  # JWT has 3 parts

    def test_create_token_with_roles(self, jwt_auth):
        """Test creating token with roles."""
        token = jwt_auth.create_token(user_id="user123", roles=["admin", "editor"])

        payload = jwt_auth.verify_token(token)

        assert payload is not None
        assert "admin" in payload.roles
        assert "editor" in payload.roles

    def test_create_token_with_session_ids(self, jwt_auth):
        """Test creating token with session IDs."""
        token = jwt_auth.create_token(user_id="user123", session_ids=["session1", "session2"])

        payload = jwt_auth.verify_token(token)

        assert payload is not None
        assert "session1" in payload.session_ids
        assert "session2" in payload.session_ids

    def test_create_token_custom_expiration(self, jwt_auth):
        """Test creating token with custom expiration."""
        token = jwt_auth.create_token(user_id="user123", expires_in_hours=1)

        payload = jwt_auth.verify_token(token)

        assert payload is not None
        assert payload.exp < time.time() + 3700  # Less than 1 hour + 100s buffer

    def test_verify_token_valid(self, jwt_auth):
        """Test verifying a valid token."""
        token = jwt_auth.create_token(user_id="user123")

        payload = jwt_auth.verify_token(token)

        assert payload is not None
        assert payload.sub == "user123"

    def test_verify_token_expired(self, jwt_auth):
        """Test verifying an expired token returns None."""
        # Create token that expires immediately
        token = jwt_auth.create_token(user_id="user123", expires_in_hours=0)

        # Wait a moment for expiration
        time.sleep(0.1)

        payload = jwt_auth.verify_token(token)

        assert payload is None

    def test_verify_token_invalid_format(self, jwt_auth):
        """Test verifying invalid token format returns None."""
        assert jwt_auth.verify_token("invalid") is None
        assert jwt_auth.verify_token("invalid.token") is None
        assert jwt_auth.verify_token("a.b.c.d") is None

    def test_verify_token_tampered_signature(self, jwt_auth):
        """Test verifying tampered token returns None."""
        token = jwt_auth.create_token(user_id="user123")
        # Tamper with the signature
        parts = token.split(".")
        parts[2] = "tampered_signature"
        tampered_token = ".".join(parts)

        payload = jwt_auth.verify_token(tampered_token)

        assert payload is None

    def test_verify_token_empty(self, jwt_auth):
        """Test verifying empty token returns None."""
        assert jwt_auth.verify_token("") is None
        assert jwt_auth.verify_token(None) is None

    def test_revoke_token(self, jwt_auth):
        """Test revoking a token."""
        token = jwt_auth.create_token(user_id="user123")

        # Token should be valid initially
        assert jwt_auth.verify_token(token) is not None

        # Revoke the token
        jwt_auth.revoke_token(token)

        # Token should now be invalid
        assert jwt_auth.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_call_missing_bearer_token_raises(self, jwt_auth):
        """Test __call__ raises HTTPException when Bearer token missing."""
        request = MagicMock()
        request.headers.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await jwt_auth(request)

        assert exc_info.value.status_code == 401
        assert "Bearer token required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_invalid_bearer_prefix_raises(self, jwt_auth):
        """Test __call__ raises HTTPException for invalid Authorization header."""
        request = MagicMock()
        request.headers.get.return_value = "Basic invalidtoken"

        with pytest.raises(HTTPException) as exc_info:
            await jwt_auth(request)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_call_invalid_token_raises(self, jwt_auth):
        """Test __call__ raises HTTPException for invalid token."""
        request = MagicMock()
        request.headers.get.return_value = "Bearer invalid.token.here"

        with pytest.raises(HTTPException) as exc_info:
            await jwt_auth(request)

        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_valid_token_returns_payload(self, jwt_auth):
        """Test __call__ returns payload for valid token."""
        token = jwt_auth.create_token(user_id="user123", roles=["admin"])

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"

        payload = await jwt_auth(request)

        assert payload is not None
        assert payload.sub == "user123"
        assert "admin" in payload.roles

    @pytest.mark.asyncio
    async def test_call_auth_disabled_returns_none(self, monkeypatch):
        """Test __call__ returns None when JWT auth is disabled."""
        monkeypatch.setenv("ENABLE_JWT_AUTH", "false")
        config = SecurityConfig()
        auth = JWTAuth(config=config)

        request = MagicMock()
        request.headers.get.return_value = None

        result = await auth(request)

        assert result is None


class TestJWTPayload:
    """Tests for JWTPayload Pydantic model."""

    def test_jwt_payload_creation(self):
        """Test creating JWTPayload with all fields."""
        payload = JWTPayload(
            sub="user123",
            exp=time.time() + 3600,
            iat=time.time(),
            roles=["admin"],
            session_ids=["session1"],
        )

        assert payload.sub == "user123"
        assert "admin" in payload.roles
        assert "session1" in payload.session_ids

    def test_jwt_payload_defaults(self):
        """Test JWTPayload default values."""
        payload = JWTPayload(sub="user123", exp=time.time() + 3600, iat=time.time())

        assert payload.roles == []
        assert payload.session_ids == []


# ============================================================================
# RateLimiter Tests
# ============================================================================


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture
    def rate_config(self, monkeypatch):
        """Create config for rate limiting testing."""
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
        return SecurityConfig()

    @pytest.fixture
    def rate_limiter(self, rate_config):
        """Create RateLimiter instance for testing."""
        return RateLimiter(config=rate_config)

    @pytest.fixture
    def mock_request(self):
        """Create mock request with client IP."""
        request = MagicMock()
        request.headers.get.return_value = None  # No X-Forwarded-For
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        return request

    def test_check_rate_limit_allows_under_limit(self, rate_limiter, mock_request):
        """Test rate limiter allows requests under limit."""
        for _ in range(5):
            assert rate_limiter.check_rate_limit(mock_request) is True

    def test_check_rate_limit_blocks_over_limit(self, rate_limiter, mock_request):
        """Test rate limiter blocks requests over limit."""
        # Use up the limit
        for _ in range(5):
            rate_limiter.check_rate_limit(mock_request)

        # Next request should be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            rate_limiter.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    def test_get_remaining_requests(self, rate_limiter, mock_request):
        """Test getting remaining requests count."""
        initial = rate_limiter.get_remaining(mock_request)
        assert initial == 5

        rate_limiter.check_rate_limit(mock_request)

        after_one = rate_limiter.get_remaining(mock_request)
        assert after_one == 4

    def test_rate_limit_per_client(self, rate_limiter):
        """Test rate limiting is per-client."""
        request1 = MagicMock()
        request1.headers.get.return_value = None
        request1.client = MagicMock()
        request1.client.host = "192.168.1.1"

        request2 = MagicMock()
        request2.headers.get.return_value = None
        request2.client = MagicMock()
        request2.client.host = "192.168.1.2"

        # Use up limit for client 1
        for _ in range(5):
            rate_limiter.check_rate_limit(request1)

        # Client 2 should still have quota
        assert rate_limiter.check_rate_limit(request2) is True
        assert rate_limiter.get_remaining(request2) == 4

    def test_x_forwarded_for_header(self, rate_limiter):
        """Test X-Forwarded-For header is used for client identification."""
        request = MagicMock()
        request.headers.get.return_value = "10.0.0.1, 10.0.0.2"
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        rate_limiter.check_rate_limit(request)

        # Should track against first IP in X-Forwarded-For
        assert rate_limiter.get_remaining(request) == 4

    def test_rate_limit_disabled(self, monkeypatch):
        """Test rate limiting can be disabled."""
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "false")
        config = SecurityConfig()
        limiter = RateLimiter(config=config)

        request = MagicMock()
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        # Should always allow when disabled
        for _ in range(100):
            assert limiter.check_rate_limit(request) is True

    @pytest.mark.asyncio
    async def test_call_as_dependency(self, rate_limiter, mock_request):
        """Test __call__ can be used as FastAPI dependency."""
        result = await rate_limiter(mock_request)
        assert result is True


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_rate_limit_exceeded_status_code(self):
        """Test RateLimitExceeded has correct status code."""
        exc = RateLimitExceeded(retry_after=30)
        assert exc.status_code == 429

    def test_rate_limit_exceeded_retry_after_header(self):
        """Test RateLimitExceeded has Retry-After header."""
        exc = RateLimitExceeded(retry_after=30)
        assert exc.headers["Retry-After"] == "30"

    def test_rate_limit_exceeded_detail(self):
        """Test RateLimitExceeded has meaningful detail."""
        exc = RateLimitExceeded(retry_after=30)
        assert "30" in exc.detail
        assert "Rate limit" in exc.detail


# ============================================================================
# InputValidator Tests
# ============================================================================


class TestInputValidator:
    """Tests for InputValidator class."""

    @pytest.fixture
    def validator(self):
        """Create InputValidator instance."""
        return InputValidator()

    def test_validate_path_valid(self, validator):
        """Test validating a valid path."""
        result = validator.validate_path("/home/user/project")
        assert result == "/home/user/project"

    def test_validate_path_traversal_detected(self, validator):
        """Test path traversal detection."""
        with pytest.raises(PathTraversalError):
            validator.validate_path("/home/user/../../../etc/passwd")

    def test_validate_path_empty_raises(self, validator):
        """Test empty path raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validator.validate_path("")

        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()

    def test_validate_path_with_base_path(self, validator, tmp_path):
        """Test path validation with base path restriction."""
        base = str(tmp_path)
        valid_path = str(tmp_path / "subdir")

        result = validator.validate_path(valid_path, base_path=base)
        assert result is not None

    def test_validate_path_outside_base_raises(self, validator, tmp_path):
        """Test path outside base path raises error."""
        base = str(tmp_path / "allowed")
        outside_path = "/etc/passwd"

        with pytest.raises(PathTraversalError):
            validator.validate_path(outside_path, base_path=base)

    def test_sanitize_string_removes_null_bytes(self, validator):
        """Test sanitize_string removes null bytes."""
        result = validator.sanitize_string("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"

    def test_sanitize_string_truncates_long_strings(self):
        """Test sanitize_string truncates strings over limit."""
        validator = InputValidator(max_string_length=10)
        result = validator.sanitize_string("a" * 100)
        assert len(result) == 10

    def test_sanitize_string_removes_xss_patterns(self, validator):
        """Test sanitize_string removes XSS patterns."""
        result = validator.sanitize_string("<script>alert('xss')</script>")
        assert "<script" not in result.lower()

    def test_sanitize_string_removes_javascript_protocol(self, validator):
        """Test sanitize_string removes javascript: protocol."""
        result = validator.sanitize_string("javascript:alert('xss')")
        assert "javascript:" not in result.lower()

    def test_sanitize_string_removes_event_handlers(self, validator):
        """Test sanitize_string removes event handlers."""
        result = validator.sanitize_string('<img onerror="alert(1)">')
        assert "onerror" not in result.lower()

    def test_sanitize_string_allows_html_when_enabled(self, validator):
        """Test sanitize_string allows HTML when allow_html=True."""
        result = validator.sanitize_string("<b>bold</b>", allow_html=True)
        assert "<b>" in result

    def test_sanitize_string_empty_returns_empty(self, validator):
        """Test sanitize_string returns empty for empty input."""
        assert validator.sanitize_string("") == ""
        assert validator.sanitize_string(None) is None

    def test_validate_query_valid(self, validator):
        """Test validate_query with valid input."""
        result = validator.validate_query("Set up MLOps pipeline")
        assert result == "Set up MLOps pipeline"

    def test_validate_query_strips_whitespace(self, validator):
        """Test validate_query strips whitespace."""
        result = validator.validate_query("  query with spaces  ")
        assert result == "query with spaces"

    def test_validate_query_empty_raises(self, validator):
        """Test validate_query raises on empty input."""
        with pytest.raises(HTTPException) as exc_info:
            validator.validate_query("")

        assert exc_info.value.status_code == 400

    def test_validate_query_whitespace_only_raises(self, validator):
        """Test validate_query raises on whitespace-only input."""
        with pytest.raises(HTTPException):
            validator.validate_query("   ")

    def test_validate_session_id_valid_uuid(self, validator):
        """Test validate_session_id with valid UUID."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = validator.validate_session_id(uuid)
        assert result == uuid.lower()

    def test_validate_session_id_invalid_format(self, validator):
        """Test validate_session_id raises on invalid format."""
        with pytest.raises(HTTPException) as exc_info:
            validator.validate_session_id("not-a-uuid")

        assert exc_info.value.status_code == 400
        assert "session ID format" in exc_info.value.detail

    def test_validate_session_id_empty_raises(self, validator):
        """Test validate_session_id raises on empty input."""
        with pytest.raises(HTTPException):
            validator.validate_session_id("")

    def test_validate_session_id_normalizes_case(self, validator):
        """Test validate_session_id normalizes to lowercase."""
        uuid_upper = "550E8400-E29B-41D4-A716-446655440000"
        result = validator.validate_session_id(uuid_upper)
        assert result == uuid_upper.lower()


class TestPathTraversalError:
    """Tests for PathTraversalError exception."""

    def test_path_traversal_error_status_code(self):
        """Test PathTraversalError has correct status code."""
        exc = PathTraversalError("../etc/passwd")
        assert exc.status_code == 400

    def test_path_traversal_error_detail(self):
        """Test PathTraversalError has meaningful detail."""
        exc = PathTraversalError("../etc/passwd")
        assert "path traversal" in exc.detail.lower()


# ============================================================================
# SessionOwnership Tests
# ============================================================================


class TestSessionOwnership:
    """Tests for SessionOwnership class."""

    @pytest.fixture
    def ownership(self):
        """Create SessionOwnership instance."""
        return SessionOwnership()

    def test_register_session(self, ownership):
        """Test registering a session."""
        ownership.register_session("session1", "user1")

        assert ownership.get_owner("session1") == "user1"

    def test_unregister_session(self, ownership):
        """Test unregistering a session."""
        ownership.register_session("session1", "user1")
        ownership.unregister_session("session1")

        assert ownership.get_owner("session1") is None

    def test_get_user_sessions(self, ownership):
        """Test getting all sessions for a user."""
        ownership.register_session("session1", "user1")
        ownership.register_session("session2", "user1")
        ownership.register_session("session3", "user2")

        user1_sessions = ownership.get_user_sessions("user1")

        assert len(user1_sessions) == 2
        assert "session1" in user1_sessions
        assert "session2" in user1_sessions
        assert "session3" not in user1_sessions

    def test_check_access_owner(self, ownership):
        """Test owner has access to their session."""
        ownership.register_session("session1", "user1")

        assert ownership.check_access("session1", "user1") is True

    def test_check_access_non_owner(self, ownership):
        """Test non-owner does not have access."""
        ownership.register_session("session1", "user1")

        assert ownership.check_access("session1", "user2") is False

    def test_check_access_untracked_session(self, ownership):
        """Test untracked session is considered public."""
        assert ownership.check_access("unknown-session", "any-user") is True

    def test_verify_access_owner_passes(self, ownership):
        """Test verify_access passes for owner."""
        ownership.register_session("session1", "user1")

        # Should not raise
        ownership.verify_access("session1", "user1")

    def test_verify_access_non_owner_raises(self, ownership):
        """Test verify_access raises for non-owner."""
        ownership.register_session("session1", "user1")

        with pytest.raises(HTTPException) as exc_info:
            ownership.verify_access("session1", "user2")

        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail

    def test_unregister_updates_user_sessions(self, ownership):
        """Test unregistering removes from user's session list."""
        ownership.register_session("session1", "user1")
        ownership.register_session("session2", "user1")
        ownership.unregister_session("session1")

        sessions = ownership.get_user_sessions("user1")
        assert "session1" not in sessions
        assert "session2" in sessions


# ============================================================================
# SecurityHeaders Tests
# ============================================================================


class TestSecurityHeaders:
    """Tests for SecurityHeaders class."""

    def test_default_headers(self):
        """Test default security headers are present."""
        headers = SecurityHeaders()
        h = headers.get_headers()

        assert "X-Content-Type-Options" in h
        assert h["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in h
        assert "X-XSS-Protection" in h
        assert "Strict-Transport-Security" in h
        assert "Content-Security-Policy" in h
        assert "Referrer-Policy" in h
        assert "Permissions-Policy" in h

    def test_custom_headers_override(self):
        """Test custom headers override defaults."""
        custom = {"X-Custom-Header": "custom-value", "X-Frame-Options": "SAMEORIGIN"}
        headers = SecurityHeaders(custom_headers=custom)
        h = headers.get_headers()

        assert h["X-Custom-Header"] == "custom-value"
        assert h["X-Frame-Options"] == "SAMEORIGIN"  # Overridden

    def test_get_headers_returns_copy(self):
        """Test get_headers returns a copy, not original."""
        headers = SecurityHeaders()
        h1 = headers.get_headers()
        h1["new-header"] = "new-value"
        h2 = headers.get_headers()

        assert "new-header" not in h2


# ============================================================================
# APIKeyManager Tests
# ============================================================================


class TestAPIKeyManager:
    """Tests for APIKeyManager class."""

    @pytest.fixture
    def manager(self):
        """Create fresh APIKeyManager instance."""
        return APIKeyManager()

    def test_generate_creates_key(self, manager):
        """Test generate creates a key with metadata."""
        result = manager.generate(name="Test Key", user_id="user123")

        assert isinstance(result, GeneratedKey)
        assert isinstance(result.raw_key, str)
        assert len(result.raw_key) == 43
        assert result.key_info.name == "Test Key"
        assert result.key_info.user_id == "user123"
        assert result.key_info.is_active is True

    def test_generate_with_expiration(self, manager):
        """Test generate with expiration time."""
        result = manager.generate(name="Expiring Key", expires_in_days=30)

        assert result.key_info.expires_at is not None
        expected = datetime.utcnow() + timedelta(days=30)
        # Allow 1 second tolerance
        assert abs((result.key_info.expires_at - expected).total_seconds()) < 1

    def test_generate_with_scopes(self, manager):
        """Test generate with scopes."""
        result = manager.generate(name="Scoped Key", scopes=["read", "write"])

        assert "read" in result.key_info.scopes
        assert "write" in result.key_info.scopes

    def test_generate_unique_keys(self, manager):
        """Test generate creates unique keys."""
        keys = [manager.generate(name=f"Key {i}") for i in range(10)]
        raw_keys = [k.raw_key for k in keys]

        assert len(set(raw_keys)) == 10

    def test_verify_valid_key(self, manager):
        """Test verify returns info for valid key."""
        result = manager.generate(name="Test Key", user_id="user123")

        info = manager.verify(result.raw_key)

        assert info is not None
        assert info.name == "Test Key"
        assert info.user_id == "user123"

    def test_verify_invalid_key(self, manager):
        """Test verify returns None for invalid key."""
        info = manager.verify("invalid-key")
        assert info is None

    def test_verify_empty_key(self, manager):
        """Test verify returns None for empty key."""
        assert manager.verify("") is None
        assert manager.verify(None) is None

    def test_verify_revoked_key(self, manager):
        """Test verify returns None for revoked key."""
        result = manager.generate(name="Test Key")
        manager.revoke(result.key_info.key_id)

        info = manager.verify(result.raw_key)
        assert info is None

    def test_verify_expired_key(self, manager):
        """Test verify returns None for expired key."""
        result = manager.generate(name="Expired Key", expires_in_days=0)
        # Manually set expiration to past
        for key_info in manager._keys.values():
            if key_info.key_id == result.key_info.key_id:
                key_info.expires_at = datetime.utcnow() - timedelta(days=1)

        info = manager.verify(result.raw_key)
        assert info is None

    def test_verify_with_required_scope_present(self, manager):
        """Test verify with required scope that is present."""
        result = manager.generate(name="Scoped Key", scopes=["read", "write"])

        info = manager.verify(result.raw_key, required_scope="read")
        assert info is not None

    def test_verify_with_required_scope_missing(self, manager):
        """Test verify with required scope that is missing."""
        result = manager.generate(name="Limited Key", scopes=["read"])

        info = manager.verify(result.raw_key, required_scope="admin")
        assert info is None

    def test_verify_updates_last_used(self, manager):
        """Test verify updates last_used_at timestamp."""
        result = manager.generate(name="Test Key")
        assert result.key_info.last_used_at is None

        manager.verify(result.raw_key)

        info = manager.get_key_info(result.key_info.key_id)
        assert info.last_used_at is not None

    def test_revoke_by_id(self, manager):
        """Test revoking key by ID."""
        result = manager.generate(name="Test Key")

        success = manager.revoke(result.key_info.key_id)

        assert success is True
        assert manager.verify(result.raw_key) is None

    def test_revoke_nonexistent_key(self, manager):
        """Test revoking nonexistent key returns False."""
        success = manager.revoke("nonexistent-id")
        assert success is False

    def test_revoke_already_revoked(self, manager):
        """Test revoking already revoked key returns False."""
        result = manager.generate(name="Test Key")
        manager.revoke(result.key_info.key_id)

        success = manager.revoke(result.key_info.key_id)
        assert success is False

    def test_revoke_by_raw_key(self, manager):
        """Test revoking key by raw key string."""
        result = manager.generate(name="Test Key")

        success = manager.revoke_by_raw_key(result.raw_key)

        assert success is True
        assert manager.verify(result.raw_key) is None

    def test_revoke_by_raw_key_empty(self, manager):
        """Test revoking empty raw key returns False."""
        assert manager.revoke_by_raw_key("") is False

    def test_get_key_info(self, manager):
        """Test getting key info by ID."""
        result = manager.generate(name="Test Key")

        info = manager.get_key_info(result.key_info.key_id)

        assert info is not None
        assert info.name == "Test Key"

    def test_get_key_info_nonexistent(self, manager):
        """Test getting info for nonexistent key returns None."""
        info = manager.get_key_info("nonexistent-id")
        assert info is None

    def test_list_keys_all(self, manager):
        """Test listing all keys."""
        manager.generate(name="Key 1", user_id="user1")
        manager.generate(name="Key 2", user_id="user2")

        keys = manager.list_keys()

        assert len(keys) == 2

    def test_list_keys_by_user(self, manager):
        """Test listing keys filtered by user."""
        manager.generate(name="Key 1", user_id="user1")
        manager.generate(name="Key 2", user_id="user1")
        manager.generate(name="Key 3", user_id="user2")

        keys = manager.list_keys(user_id="user1")

        assert len(keys) == 2
        assert all(k.user_id == "user1" for k in keys)

    def test_list_keys_excludes_revoked(self, manager):
        """Test listing keys excludes revoked by default."""
        result = manager.generate(name="Revoked Key")
        manager.generate(name="Active Key")
        manager.revoke(result.key_info.key_id)

        keys = manager.list_keys()

        assert len(keys) == 1
        assert keys[0].name == "Active Key"

    def test_list_keys_include_revoked(self, manager):
        """Test listing keys can include revoked."""
        result = manager.generate(name="Revoked Key")
        manager.generate(name="Active Key")
        manager.revoke(result.key_info.key_id)

        keys = manager.list_keys(include_revoked=True)

        assert len(keys) == 2

    def test_revoke_all_for_user(self, manager):
        """Test revoking all keys for a user."""
        manager.generate(name="Key 1", user_id="user1")
        manager.generate(name="Key 2", user_id="user1")
        manager.generate(name="Key 3", user_id="user2")

        count = manager.revoke_all_for_user("user1")

        assert count == 2
        assert len(manager.list_keys(user_id="user1")) == 0
        assert len(manager.list_keys(user_id="user2")) == 1

    def test_cleanup_expired(self, manager):
        """Test cleaning up expired keys."""
        result = manager.generate(name="Expired Key", expires_in_days=0)
        manager.generate(name="Active Key")

        # Manually expire the key
        for key_info in manager._keys.values():
            if key_info.key_id == result.key_info.key_id:
                key_info.expires_at = datetime.utcnow() - timedelta(days=1)

        count = manager.cleanup_expired()

        assert count == 1
        assert len(manager.list_keys()) == 1

    def test_export_import_state(self, manager):
        """Test exporting and importing state."""
        manager.generate(name="Key 1", user_id="user1", scopes=["read"])
        result = manager.generate(name="Key 2")
        manager.revoke(result.key_info.key_id)

        state = manager.export_state()

        # Create new manager and import
        new_manager = APIKeyManager()
        new_manager.import_state(state)

        # Verify state was imported
        keys = new_manager.list_keys(include_revoked=True)
        assert len(keys) == 2


class TestAPIKeyInfo:
    """Tests for APIKeyInfo Pydantic model."""

    def test_api_key_info_creation(self):
        """Test creating APIKeyInfo with all fields."""
        info = APIKeyInfo(
            key_id="key123",
            name="Test Key",
            key_prefix="abc12345",
            user_id="user123",
            is_active=True,
            scopes=["read", "write"],
        )

        assert info.key_id == "key123"
        assert info.name == "Test Key"
        assert info.is_active is True

    def test_api_key_info_defaults(self):
        """Test APIKeyInfo default values."""
        info = APIKeyInfo(key_id="key123", name="Test Key", key_prefix="abc12345")

        assert info.user_id is None
        assert info.is_active is True
        assert info.expires_at is None
        assert info.last_used_at is None
        assert info.scopes == []


# ============================================================================
# Middleware Tests
# ============================================================================


class TestCurrentUser:
    """Tests for CurrentUser model."""

    def test_current_user_creation(self):
        """Test creating CurrentUser with all fields."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
            scopes=["read", "write"],
            session_ids=["session1"],
        )

        assert user.user_id == "user123"
        assert user.is_authenticated is True
        assert user.auth_method == "jwt"
        assert "admin" in user.roles

    def test_current_user_defaults(self):
        """Test CurrentUser default values."""
        user = CurrentUser(user_id="user123", auth_method="anonymous")

        assert user.is_authenticated is True
        assert user.roles == []
        assert user.scopes == []
        assert user.session_ids == []
        assert user.api_key_id is None
        assert user.api_key_name is None


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_authentication_error_defaults(self):
        """Test AuthenticationError with default message."""
        exc = AuthenticationError()

        assert exc.status_code == 401
        assert "Authentication required" in exc.detail
        assert "WWW-Authenticate" in exc.headers

    def test_authentication_error_custom_detail(self):
        """Test AuthenticationError with custom message."""
        exc = AuthenticationError(detail="Token expired")

        assert "Token expired" in exc.detail


class TestAuthorizationError:
    """Tests for AuthorizationError exception."""

    def test_authorization_error_defaults(self):
        """Test AuthorizationError with default message."""
        exc = AuthorizationError()

        assert exc.status_code == 403
        assert "Insufficient permissions" in exc.detail

    def test_authorization_error_custom_detail(self):
        """Test AuthorizationError with custom message."""
        exc = AuthorizationError(detail="Admin role required")

        assert "Admin role required" in exc.detail


# ============================================================================
# Integration Tests
# ============================================================================


class TestAuthenticationIntegration:
    """Integration tests for authentication flow."""

    @pytest.fixture
    def full_config(self, monkeypatch):
        """Create full configuration for integration testing."""
        monkeypatch.setenv("JWT_SECRET", "integration-test-secret")
        monkeypatch.setenv("ENABLE_API_KEY_AUTH", "true")
        monkeypatch.setenv("ENABLE_JWT_AUTH", "true")
        monkeypatch.setenv("VALID_API_KEYS", "integration-test-key")
        return SecurityConfig()

    def test_api_key_auth_flow(self, full_config):
        """Test complete API key authentication flow."""
        api_auth = APIKeyAuth(config=full_config)

        # Generate a new key
        new_key = APIKeyAuth.generate_api_key()
        api_auth.add_key(new_key)

        # Validate the key
        assert api_auth.validate_key(new_key) is True

        # Remove the key
        api_auth.remove_key(new_key)
        assert api_auth.validate_key(new_key) is False

    def test_jwt_auth_flow(self, full_config):
        """Test complete JWT authentication flow."""
        jwt_auth = JWTAuth(config=full_config)

        # Create token
        token = jwt_auth.create_token(
            user_id="test-user", roles=["admin"], session_ids=["session-123"]
        )

        # Verify token
        payload = jwt_auth.verify_token(token)
        assert payload is not None
        assert payload.sub == "test-user"
        assert "admin" in payload.roles

        # Revoke token
        jwt_auth.revoke_token(token)
        assert jwt_auth.verify_token(token) is None

    def test_api_key_manager_full_lifecycle(self):
        """Test API key manager complete lifecycle."""
        manager = APIKeyManager()

        # Generate key
        result = manager.generate(
            name="Production Key",
            user_id="user123",
            expires_in_days=90,
            scopes=["read", "write", "admin"],
        )

        # Verify key
        info = manager.verify(result.raw_key)
        assert info is not None
        assert info.name == "Production Key"

        # Check scope
        info_with_scope = manager.verify(result.raw_key, required_scope="admin")
        assert info_with_scope is not None

        # Missing scope should fail
        info_missing_scope = manager.verify(result.raw_key, required_scope="superadmin")
        assert info_missing_scope is None

        # List keys
        keys = manager.list_keys(user_id="user123")
        assert len(keys) == 1

        # Revoke key
        manager.revoke(result.key_info.key_id)
        assert manager.verify(result.raw_key) is None

    def test_session_ownership_flow(self):
        """Test session ownership complete flow."""
        ownership = SessionOwnership()

        # Register sessions
        ownership.register_session("session1", "user1")
        ownership.register_session("session2", "user1")
        ownership.register_session("session3", "user2")

        # Check access
        assert ownership.check_access("session1", "user1") is True
        assert ownership.check_access("session1", "user2") is False

        # Get user sessions
        user1_sessions = ownership.get_user_sessions("user1")
        assert len(user1_sessions) == 2

        # Unregister session
        ownership.unregister_session("session1")
        assert ownership.get_owner("session1") is None
        assert len(ownership.get_user_sessions("user1")) == 1

    def test_input_validation_security(self):
        """Test input validation prevents security issues."""
        validator = InputValidator()

        # Path traversal
        with pytest.raises(PathTraversalError):
            validator.validate_path("../../etc/passwd")

        # XSS prevention
        clean = validator.sanitize_string('<script>alert("xss")</script>')
        assert "<script" not in clean.lower()

        # Session ID validation
        with pytest.raises(HTTPException):
            validator.validate_session_id("invalid-session")

        # Valid UUID passes
        valid_uuid = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
        result = validator.validate_session_id(valid_uuid)
        assert result == valid_uuid
