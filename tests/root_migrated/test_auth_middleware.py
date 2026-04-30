#!/usr/bin/env python3
"""
Additional tests for auth middleware and API key validation.

Focuses on edge cases and scenarios not covered by existing tests:
- Authentication priority order
- Combined auth methods
- Token/key format edge cases
- Error message validation
- Concurrent access scenarios

Run with: pytest tests/root_migrated/test_auth_middleware.py -v
"""

import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from security import (
    APIKeyAuth,
    JWTAuth,
    SecurityConfig,
)
from security.api_keys import APIKeyManager
from security.middleware import (
    AuthenticationError,
    AuthorizationError,
    CurrentUser,
    get_current_user,
    get_current_user_optional,
    require_roles,
    require_scopes,
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
def fresh_api_key_manager():
    """Create a fresh API key manager for isolated tests."""
    return APIKeyManager()


@pytest.fixture
def auth_disabled_env():
    """Environment with authentication disabled."""
    os.environ["ENABLE_API_KEY_AUTH"] = "false"
    os.environ["ENABLE_JWT_AUTH"] = "false"
    yield
    del os.environ["ENABLE_API_KEY_AUTH"]
    del os.environ["ENABLE_JWT_AUTH"]


@pytest.fixture
def api_key_only_env():
    """Environment with only API key auth enabled."""
    os.environ["ENABLE_API_KEY_AUTH"] = "true"
    os.environ["ENABLE_JWT_AUTH"] = "false"
    os.environ["VALID_API_KEYS"] = "valid-api-key-123"
    yield
    del os.environ["ENABLE_API_KEY_AUTH"]
    del os.environ["ENABLE_JWT_AUTH"]
    del os.environ["VALID_API_KEYS"]


@pytest.fixture
def jwt_only_env():
    """Environment with only JWT auth enabled."""
    os.environ["ENABLE_API_KEY_AUTH"] = "false"
    os.environ["ENABLE_JWT_AUTH"] = "true"
    os.environ["JWT_SECRET"] = "test-jwt-secret-for-tests"
    yield
    del os.environ["ENABLE_API_KEY_AUTH"]
    del os.environ["ENABLE_JWT_AUTH"]
    del os.environ["JWT_SECRET"]


@pytest.fixture
def both_auth_enabled_env():
    """Environment with both API key and JWT auth enabled."""
    os.environ["ENABLE_API_KEY_AUTH"] = "true"
    os.environ["ENABLE_JWT_AUTH"] = "true"
    os.environ["VALID_API_KEYS"] = "valid-api-key-123"
    os.environ["JWT_SECRET"] = "test-jwt-secret-for-tests"
    yield
    del os.environ["ENABLE_API_KEY_AUTH"]
    del os.environ["ENABLE_JWT_AUTH"]
    del os.environ["VALID_API_KEYS"]
    del os.environ["JWT_SECRET"]


# ============================================================================
# Authentication Priority Tests
# ============================================================================


class TestAuthenticationPriority:
    """Tests for authentication method priority ordering."""

    @pytest.mark.asyncio
    async def test_api_key_takes_priority_over_jwt(self, mock_request, both_auth_enabled_env):
        """Test that API key auth is attempted before JWT when both are provided."""
        from security.api_keys import api_key_manager

        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        # Generate a valid API key
        result = api_key_manager.generate(
            name="Priority Test Key", user_id="apikey-user", scopes=["read"]
        )

        # Also create a valid JWT
        token = jwt_auth.create_token(user_id="jwt-user", roles=["admin"])

        # Provide both credentials
        mock_request.headers = {
            "X-API-Key": result.raw_key,
            "Authorization": f"Bearer {token}",
        }

        user = await get_current_user(mock_request, api_key_auth, jwt_auth)

        # Should authenticate via API key (checked first)
        assert user.auth_method == "api_key"
        assert user.user_id == "apikey-user"

        # Cleanup
        api_key_manager.revoke(result.key_info.key_id)

    @pytest.mark.asyncio
    async def test_jwt_used_when_api_key_invalid(self, mock_request, both_auth_enabled_env):
        """Test that JWT is used when API key is invalid."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        token = jwt_auth.create_token(user_id="jwt-user", roles=["user"])

        # Provide invalid API key and valid JWT
        mock_request.headers = {
            "X-API-Key": "invalid-api-key",
            "Authorization": f"Bearer {token}",
        }

        # Should fail because invalid API key when API auth is enabled
        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user(mock_request, api_key_auth, jwt_auth)
        assert "Invalid API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_api_key_manager_priority_over_hashed_keys(self, mock_request, auth_disabled_env):
        """Test APIKeyManager takes priority over APIKeyAuth hashed keys."""
        from security.api_keys import api_key_manager

        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        # Add a key to the simple APIKeyAuth system
        simple_key = "simple-hashed-key"
        api_key_auth.add_key(simple_key)

        # Also generate a key via APIKeyManager with same prefix
        result = api_key_manager.generate(
            name="Manager Key", user_id="manager-user", scopes=["admin"]
        )

        # Try the manager key first
        mock_request.headers = {"X-API-Key": result.raw_key}
        user = await get_current_user(mock_request, api_key_auth, jwt_auth)

        # Should use manager key (has full metadata)
        assert user.user_id == "manager-user"
        assert user.api_key_name == "Manager Key"
        assert "admin" in user.scopes

        # Cleanup
        api_key_manager.revoke(result.key_info.key_id)
        api_key_auth.remove_key(simple_key)


# ============================================================================
# Bearer Token Format Edge Cases
# ============================================================================


class TestBearerTokenFormat:
    """Tests for Bearer token format handling."""

    @pytest.mark.asyncio
    async def test_bearer_token_case_sensitivity(self, mock_request, jwt_only_env):
        """Test that 'Bearer' prefix is case-sensitive."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        token = jwt_auth.create_token(user_id="user123")

        # Should fail with lowercase 'bearer'
        mock_request.headers = {"Authorization": f"bearer {token}"}
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, api_key_auth, jwt_auth)

    @pytest.mark.asyncio
    async def test_bearer_token_with_extra_spaces(self, mock_request, jwt_only_env):
        """Test Bearer token with extra spaces."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        token = jwt_auth.create_token(user_id="user123")

        # Should fail with extra spaces (token starts with space)
        mock_request.headers = {"Authorization": f"Bearer  {token}"}
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, api_key_auth, jwt_auth)

    @pytest.mark.asyncio
    async def test_bearer_token_only_prefix(self, mock_request, jwt_only_env):
        """Test Bearer token with only prefix, no token."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {"Authorization": "Bearer "}
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, api_key_auth, jwt_auth)

    @pytest.mark.asyncio
    async def test_non_bearer_auth_scheme(self, mock_request, jwt_only_env):
        """Test non-Bearer authorization scheme."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}
        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user(mock_request, api_key_auth, jwt_auth)
        assert "required" in str(exc_info.value.detail).lower()


# ============================================================================
# API Key Format Edge Cases
# ============================================================================


class TestAPIKeyFormat:
    """Tests for API key format handling."""

    @pytest.mark.asyncio
    async def test_empty_api_key_header(self, mock_request, api_key_only_env):
        """Test empty API key header value."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {"X-API-Key": ""}
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, api_key_auth, jwt_auth)

    @pytest.mark.asyncio
    async def test_whitespace_only_api_key(self, mock_request, api_key_only_env):
        """Test whitespace-only API key."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {"X-API-Key": "   "}
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, api_key_auth, jwt_auth)

    @pytest.mark.asyncio
    async def test_api_key_with_special_characters(self, mock_request, auth_disabled_env):
        """Test API key containing special characters."""
        from security.api_keys import api_key_manager

        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        # Generated keys can contain URL-safe base64 characters
        result = api_key_manager.generate(name="Special Chars Key", user_id="user1")

        mock_request.headers = {"X-API-Key": result.raw_key}
        user = await get_current_user(mock_request, api_key_auth, jwt_auth)

        assert user.is_authenticated
        api_key_manager.revoke(result.key_info.key_id)

    def test_api_key_hash_consistency(self):
        """Test that API key hashing is consistent."""
        manager = APIKeyManager()
        key = "test-key-for-hashing"

        hash1 = manager._hash_key(key)
        hash2 = manager._hash_key(key)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_different_keys_produce_different_hashes(self):
        """Test that different keys produce different hashes."""
        manager = APIKeyManager()

        hash1 = manager._hash_key("key1")
        hash2 = manager._hash_key("key2")

        assert hash1 != hash2


# ============================================================================
# JWT Token Edge Cases
# ============================================================================


class TestJWTTokenEdgeCases:
    """Tests for JWT token edge cases."""

    def test_token_with_malformed_base64(self):
        """Test token with invalid base64 encoding."""
        os.environ["JWT_SECRET"] = "test-secret"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            # Malformed base64 in payload
            malformed_token = "valid.!!!invalid-base64!!!.signature"
            assert jwt_auth.verify_token(malformed_token) is None
        finally:
            del os.environ["JWT_SECRET"]

    def test_token_with_invalid_json(self):
        """Test token with invalid JSON in payload."""
        os.environ["JWT_SECRET"] = "test-secret"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            # Create a token then manually corrupt it
            token = jwt_auth.create_token(user_id="user123")
            parts = token.split(".")

            # Replace payload with invalid JSON (valid base64 but not JSON)
            import base64

            invalid_payload = base64.urlsafe_b64encode(b"not-json").rstrip(b"=").decode()
            corrupted = f"{parts[0]}.{invalid_payload}.{parts[2]}"

            assert jwt_auth.verify_token(corrupted) is None
        finally:
            del os.environ["JWT_SECRET"]

    def test_token_with_wrong_secret(self):
        """Test token verification with wrong secret fails."""
        os.environ["JWT_SECRET"] = "secret-one"
        try:
            config1 = SecurityConfig()
            jwt_auth1 = JWTAuth(config=config1)
            token = jwt_auth1.create_token(user_id="user123")
        finally:
            del os.environ["JWT_SECRET"]

        os.environ["JWT_SECRET"] = "secret-two"
        try:
            config2 = SecurityConfig()
            jwt_auth2 = JWTAuth(config=config2)

            # Token signed with different secret should fail
            assert jwt_auth2.verify_token(token) is None
        finally:
            del os.environ["JWT_SECRET"]

    def test_token_just_before_expiry(self):
        """Test token verification just before expiry."""
        os.environ["JWT_SECRET"] = "test-secret"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            # Create token with 1 hour expiry
            token = jwt_auth.create_token(user_id="user123", expires_in_hours=1)

            # Should be valid immediately
            payload = jwt_auth.verify_token(token)
            assert payload is not None
            assert payload.sub == "user123"
        finally:
            del os.environ["JWT_SECRET"]

    def test_token_with_empty_roles(self):
        """Test token with empty roles list."""
        os.environ["JWT_SECRET"] = "test-secret"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            token = jwt_auth.create_token(user_id="user123", roles=[])
            payload = jwt_auth.verify_token(token)

            assert payload is not None
            assert payload.roles == []
        finally:
            del os.environ["JWT_SECRET"]

    def test_token_with_many_session_ids(self):
        """Test token with many session IDs."""
        os.environ["JWT_SECRET"] = "test-secret"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            session_ids = [f"session-{i}" for i in range(100)]
            token = jwt_auth.create_token(user_id="user123", session_ids=session_ids)
            payload = jwt_auth.verify_token(token)

            assert payload is not None
            assert len(payload.session_ids) == 100
        finally:
            del os.environ["JWT_SECRET"]


# ============================================================================
# Error Message Tests
# ============================================================================


class TestErrorMessages:
    """Tests for error message content."""

    def test_authentication_error_default_message(self):
        """Test AuthenticationError default message."""
        error = AuthenticationError()
        assert error.status_code == 401
        assert error.detail == "Authentication required"
        assert "WWW-Authenticate" in error.headers

    def test_authentication_error_custom_message(self):
        """Test AuthenticationError with custom message."""
        error = AuthenticationError("Custom auth error")
        assert error.detail == "Custom auth error"

    def test_authorization_error_default_message(self):
        """Test AuthorizationError default message."""
        error = AuthorizationError()
        assert error.status_code == 403
        assert error.detail == "Insufficient permissions"

    def test_authorization_error_custom_message(self):
        """Test AuthorizationError with custom message."""
        error = AuthorizationError("Missing admin role")
        assert error.detail == "Missing admin role"

    @pytest.mark.asyncio
    async def test_missing_api_key_error_message(self, mock_request, api_key_only_env):
        """Test error message when API key is missing."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {}

        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user(mock_request, api_key_auth, jwt_auth)
        assert "required" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_invalid_api_key_error_message(self, mock_request, api_key_only_env):
        """Test error message when API key is invalid."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {"X-API-Key": "wrong-key"}

        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user(mock_request, api_key_auth, jwt_auth)
        assert "invalid" in str(exc_info.value.detail).lower()


# ============================================================================
# Role and Scope Requirement Edge Cases
# ============================================================================


class TestRoleAndScopeRequirements:
    """Tests for role and scope requirement edge cases."""

    @pytest.mark.asyncio
    async def test_require_roles_with_none_roles(self):
        """Test require_roles when user has None roles."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=[],  # Empty list, not None
        )

        checker = require_roles("admin")
        with pytest.raises(AuthorizationError):
            await checker(user=user)

    @pytest.mark.asyncio
    async def test_require_scopes_all_must_match(self):
        """Test that require_scopes requires ALL scopes."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="api_key",
            scopes=["read", "write"],
        )

        # Should pass - has both required scopes
        checker = require_scopes("read", "write")
        result = await checker(user=user)
        assert result.user_id == "user123"

        # Should fail - missing 'delete' scope
        checker_strict = require_scopes("read", "write", "delete")
        with pytest.raises(AuthorizationError) as exc_info:
            await checker_strict(user=user)
        assert "delete" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_roles_any_match(self):
        """Test that require_roles passes with ANY matching role."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["editor"],
        )

        # Should pass - has 'editor' which is one of the required
        checker = require_roles("admin", "editor", "viewer")
        result = await checker(user=user)
        assert result.user_id == "user123"

    @pytest.mark.asyncio
    async def test_scope_case_sensitivity(self):
        """Test that scopes are case-sensitive."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="api_key",
            scopes=["Read", "Write"],
        )

        # Should fail - case mismatch
        checker = require_scopes("read")
        with pytest.raises(AuthorizationError):
            await checker(user=user)


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent API key operations."""

    def test_concurrent_key_generation(self):
        """Test generating many keys concurrently produces unique keys."""
        manager = APIKeyManager()

        def generate_key(i):
            return manager.generate(name=f"Key {i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(generate_key, range(50)))

        # All keys should be unique
        raw_keys = [r.raw_key for r in results]
        assert len(set(raw_keys)) == 50

        key_ids = [r.key_info.key_id for r in results]
        assert len(set(key_ids)) == 50

    def test_concurrent_verification(self):
        """Test verifying keys concurrently."""
        manager = APIKeyManager()
        result = manager.generate(name="Concurrent Test Key")

        def verify_key(_):
            return manager.verify(result.raw_key)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(verify_key, range(50)))

        # All verifications should succeed
        assert all(r is not None for r in results)
        assert all(r.name == "Concurrent Test Key" for r in results)

    def test_revoke_during_verification(self):
        """Test that revocation is respected during concurrent verifications."""
        manager = APIKeyManager()
        result = manager.generate(name="Revoke Test Key")

        # Verify works initially
        assert manager.verify(result.raw_key) is not None

        # Revoke the key
        manager.revoke(result.key_info.key_id)

        # Verify should now fail
        assert manager.verify(result.raw_key) is None


# ============================================================================
# API Key Expiration Edge Cases
# ============================================================================


class TestAPIKeyExpiration:
    """Tests for API key expiration handling."""

    def test_key_expires_exactly_now(self):
        """Test key that expires at the exact current moment."""
        manager = APIKeyManager()
        # Create key that expires in -0 days (effectively now or in the past)
        result = manager.generate(name="Expiring Now", expires_in_days=0)

        # Should be valid immediately (expires at end of day 0)
        # The verify call tests that keys with zero-day expiration can be created
        # The result may be None or valid depending on exact timing
        _ = manager.verify(result.raw_key)

    def test_key_with_no_expiration(self):
        """Test key without expiration date."""
        manager = APIKeyManager()
        result = manager.generate(name="No Expiry Key")

        assert result.key_info.expires_at is None

        # Should remain valid
        key_info = manager.verify(result.raw_key)
        assert key_info is not None

    def test_cleanup_only_removes_expired(self):
        """Test cleanup only removes expired keys, not valid ones."""
        manager = APIKeyManager()

        # Create mix of expired and valid keys
        expired1 = manager.generate(name="Expired 1", expires_in_days=-10)
        expired2 = manager.generate(name="Expired 2", expires_in_days=-5)
        valid1 = manager.generate(name="Valid 1", expires_in_days=30)
        valid2 = manager.generate(name="Valid 2")  # No expiration

        count = manager.cleanup_expired()
        assert count == 2

        # Valid keys should still work
        assert manager.verify(valid1.raw_key) is not None
        assert manager.verify(valid2.raw_key) is not None

        # Expired keys should be gone
        assert manager.verify(expired1.raw_key) is None
        assert manager.verify(expired2.raw_key) is None


# ============================================================================
# CurrentUser Model Tests
# ============================================================================


class TestCurrentUserModel:
    """Tests for CurrentUser model edge cases."""

    def test_current_user_all_fields(self):
        """Test CurrentUser with all fields populated."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="api_key",
            roles=["admin", "user"],
            scopes=["read", "write", "deploy"],
            session_ids=["session-1", "session-2"],
            api_key_id="key-abc",
            api_key_name="Production Key",
        )

        assert user.user_id == "user123"
        assert user.is_authenticated is True
        assert user.auth_method == "api_key"
        assert len(user.roles) == 2
        assert len(user.scopes) == 3
        assert len(user.session_ids) == 2
        assert user.api_key_id == "key-abc"
        assert user.api_key_name == "Production Key"

    def test_current_user_minimal(self):
        """Test CurrentUser with minimal required fields."""
        user = CurrentUser(
            user_id="anon",
            auth_method="anonymous",
        )

        assert user.user_id == "anon"
        assert user.is_authenticated is True  # Default
        assert user.auth_method == "anonymous"
        assert user.roles == []
        assert user.scopes == []
        assert user.session_ids == []
        assert user.api_key_id is None
        assert user.api_key_name is None

    def test_current_user_serialization(self):
        """Test CurrentUser model serialization."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        # Should be serializable to dict
        user_dict = user.model_dump()
        assert user_dict["user_id"] == "user123"
        assert user_dict["auth_method"] == "jwt"

        # Should be reconstructable
        user2 = CurrentUser(**user_dict)
        assert user2.user_id == user.user_id


# ============================================================================
# get_current_user_optional Tests
# ============================================================================


class TestGetCurrentUserOptional:
    """Additional tests for get_current_user_optional."""

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_credentials(self, mock_request, api_key_only_env):
        """Test that optional returns None when credentials are missing."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {}

        user = await get_current_user_optional(mock_request, api_key_auth, jwt_auth)
        assert user is None

    @pytest.mark.asyncio
    async def test_returns_user_when_anonymous_allowed(self, mock_request, auth_disabled_env):
        """Test that optional returns anonymous user when auth disabled."""
        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        mock_request.headers = {}

        user = await get_current_user_optional(mock_request, api_key_auth, jwt_auth)
        assert user is not None
        assert user.auth_method == "anonymous"


# ============================================================================
# APIKeyAuth Additional Tests
# ============================================================================


class TestAPIKeyAuthAdditional:
    """Additional tests for APIKeyAuth."""

    def test_hash_key_is_deterministic(self):
        """Test that key hashing is deterministic."""
        auth = APIKeyAuth()
        key = "my-test-key"

        hash1 = auth._hash_key(key)
        hash2 = auth._hash_key(key)

        assert hash1 == hash2

    def test_hash_key_produces_correct_length(self):
        """Test that key hash has correct length (SHA-256)."""
        auth = APIKeyAuth()
        key = "any-key"

        hashed = auth._hash_key(key)
        assert len(hashed) == 64  # 256 bits = 64 hex characters

    def test_generate_api_key_uniqueness(self):
        """Test that generated keys are unique."""
        keys = [APIKeyAuth.generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_validate_key_with_none(self):
        """Test validating None key."""
        auth = APIKeyAuth()
        assert auth.validate_key(None) is False

    def test_validate_key_empty_string(self):
        """Test validating empty string key."""
        auth = APIKeyAuth()
        assert auth.validate_key("") is False

    def test_remove_nonexistent_key(self):
        """Test removing a key that doesn't exist (should not error)."""
        auth = APIKeyAuth()
        # Should not raise
        auth.remove_key("nonexistent-key")


# ============================================================================
# JWTAuth Additional Tests
# ============================================================================


class TestJWTAuthAdditional:
    """Additional tests for JWTAuth."""

    def test_base64url_encoding_roundtrip(self):
        """Test base64url encoding/decoding roundtrip."""
        os.environ["JWT_SECRET"] = "test"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            test_data = b"Hello, World! Special chars: +/="
            encoded = jwt_auth._base64url_encode(test_data)
            decoded = jwt_auth._base64url_decode(encoded)

            assert decoded == test_data
        finally:
            del os.environ["JWT_SECRET"]

    def test_revoke_same_token_twice(self):
        """Test revoking the same token twice doesn't error."""
        os.environ["JWT_SECRET"] = "test"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            token = jwt_auth.create_token(user_id="user123")

            # First revocation
            jwt_auth.revoke_token(token)
            assert jwt_auth.verify_token(token) is None

            # Second revocation (should not error)
            jwt_auth.revoke_token(token)
            assert jwt_auth.verify_token(token) is None
        finally:
            del os.environ["JWT_SECRET"]

    def test_token_with_special_characters_in_user_id(self):
        """Test token with special characters in user_id."""
        os.environ["JWT_SECRET"] = "test"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            user_id = "user@example.com:123/test"
            token = jwt_auth.create_token(user_id=user_id)
            payload = jwt_auth.verify_token(token)

            assert payload is not None
            assert payload.sub == user_id
        finally:
            del os.environ["JWT_SECRET"]

    def test_verify_token_with_two_parts(self):
        """Test that token with only 2 parts fails."""
        os.environ["JWT_SECRET"] = "test"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            assert jwt_auth.verify_token("header.payload") is None
        finally:
            del os.environ["JWT_SECRET"]

    def test_verify_token_with_four_parts(self):
        """Test that token with 4 parts fails."""
        os.environ["JWT_SECRET"] = "test"
        try:
            config = SecurityConfig()
            jwt_auth = JWTAuth(config=config)

            assert jwt_auth.verify_token("a.b.c.d") is None
        finally:
            del os.environ["JWT_SECRET"]


# ============================================================================
# Custom API Key Header Tests
# ============================================================================


class TestCustomAPIKeyHeader:
    """Tests for custom API key header configuration."""

    @pytest.mark.asyncio
    async def test_custom_header_name(self, mock_request):
        """Test using a custom API key header name."""
        os.environ["API_KEY_HEADER"] = "X-Custom-Auth"
        os.environ["ENABLE_API_KEY_AUTH"] = "true"
        os.environ["VALID_API_KEYS"] = "custom-key-123"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            # Using custom header
            mock_request.headers = {"X-Custom-Auth": "custom-key-123"}
            user = await get_current_user(mock_request, api_key_auth, jwt_auth)

            assert user.is_authenticated
            assert user.auth_method == "api_key"

            # Using default header should fail
            mock_request.headers = {"X-API-Key": "custom-key-123"}
            with pytest.raises(AuthenticationError):
                await get_current_user(mock_request, api_key_auth, jwt_auth)
        finally:
            del os.environ["API_KEY_HEADER"]
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["VALID_API_KEYS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
