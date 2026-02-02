#!/usr/bin/env python3
"""
Tests for the security middleware module.

Run with: pytest test_middleware.py -v
"""

import os
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from security import SecurityConfig
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
def api_key_manager():
    """Create a fresh API key manager for testing."""
    return APIKeyManager()


@pytest.fixture
def auth_disabled_env():
    """Environment with authentication disabled."""
    os.environ["ENABLE_API_KEY_AUTH"] = "false"
    os.environ["ENABLE_JWT_AUTH"] = "false"
    yield
    del os.environ["ENABLE_API_KEY_AUTH"]
    del os.environ["ENABLE_JWT_AUTH"]


# ============================================================================
# CurrentUser Model Tests
# ============================================================================


class TestCurrentUser:
    """Tests for CurrentUser model."""

    def test_create_authenticated_user(self):
        """Test creating an authenticated user."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin", "user"],
        )
        assert user.user_id == "user123"
        assert user.is_authenticated is True
        assert user.auth_method == "jwt"
        assert "admin" in user.roles

    def test_create_anonymous_user(self):
        """Test creating an anonymous user."""
        user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )
        assert user.user_id == "anonymous"
        assert user.is_authenticated is False
        assert user.auth_method == "anonymous"

    def test_create_api_key_user(self):
        """Test creating a user authenticated via API key."""
        user = CurrentUser(
            user_id="apikey:user123",
            is_authenticated=True,
            auth_method="api_key",
            scopes=["read", "write"],
            api_key_id="key123",
            api_key_name="My API Key",
        )
        assert user.user_id == "apikey:user123"
        assert user.auth_method == "api_key"
        assert "read" in user.scopes
        assert user.api_key_id == "key123"


# ============================================================================
# get_current_user Tests
# ============================================================================


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_anonymous_when_auth_disabled(self, mock_request, auth_disabled_env):
        """Test anonymous user is returned when auth is disabled."""
        from security import APIKeyAuth, JWTAuth

        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        user = await get_current_user(mock_request, api_key_auth, jwt_auth)

        assert user.user_id == "anonymous"
        assert user.is_authenticated is False
        assert user.auth_method == "anonymous"

    @pytest.mark.asyncio
    async def test_api_key_auth_success(self, mock_request):
        """Test successful API key authentication."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "true"
        os.environ["ENABLE_JWT_AUTH"] = "false"
        os.environ["VALID_API_KEYS"] = "test-api-key-123"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            mock_request.headers = {"X-API-Key": "test-api-key-123"}

            user = await get_current_user(mock_request, api_key_auth, jwt_auth)

            assert user.is_authenticated is True
            assert user.auth_method == "api_key"
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["VALID_API_KEYS"]

    @pytest.mark.asyncio
    async def test_api_key_auth_failure(self, mock_request):
        """Test failed API key authentication."""
        from security import APIKeyAuth, JWTAuth

        # Set up environment before creating config
        os.environ["ENABLE_API_KEY_AUTH"] = "true"
        os.environ["ENABLE_JWT_AUTH"] = "false"
        os.environ["VALID_API_KEYS"] = "test-api-key-123"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            mock_request.headers = {"X-API-Key": "invalid-key"}

            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, api_key_auth, jwt_auth)
            assert exc_info.value.status_code == 401
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["VALID_API_KEYS"]

    @pytest.mark.asyncio
    async def test_jwt_auth_success(self, mock_request):
        """Test successful JWT authentication."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "false"
        os.environ["ENABLE_JWT_AUTH"] = "true"
        os.environ["JWT_SECRET"] = "test-jwt-secret-key"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            token = jwt_auth.create_token(
                user_id="user123", roles=["admin"], session_ids=["session-1"]
            )
            mock_request.headers = {"Authorization": f"Bearer {token}"}

            user = await get_current_user(mock_request, api_key_auth, jwt_auth)

            assert user.user_id == "user123"
            assert user.is_authenticated is True
            assert user.auth_method == "jwt"
            assert "admin" in user.roles
            assert "session-1" in user.session_ids
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["JWT_SECRET"]

    @pytest.mark.asyncio
    async def test_jwt_auth_failure(self, mock_request):
        """Test failed JWT authentication."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "false"
        os.environ["ENABLE_JWT_AUTH"] = "true"
        os.environ["JWT_SECRET"] = "test-jwt-secret-key"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            mock_request.headers = {"Authorization": "Bearer invalid-token"}

            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, api_key_auth, jwt_auth)
            assert exc_info.value.status_code == 401
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["JWT_SECRET"]

    @pytest.mark.asyncio
    async def test_missing_credentials_when_auth_required(self, mock_request):
        """Test error when credentials missing and auth required."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "true"
        os.environ["ENABLE_JWT_AUTH"] = "false"
        os.environ["VALID_API_KEYS"] = "test-api-key-123"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            mock_request.headers = {}

            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, api_key_auth, jwt_auth)
            assert exc_info.value.status_code == 401
            assert "Authentication required" in str(exc_info.value.detail)
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["VALID_API_KEYS"]

    @pytest.mark.asyncio
    async def test_api_key_manager_auth(self, mock_request, auth_disabled_env):
        """Test authentication via APIKeyManager."""
        from security import APIKeyAuth, JWTAuth
        from security.api_keys import api_key_manager

        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        # Generate a key using the manager
        result = api_key_manager.generate(
            name="Test Key", user_id="user456", scopes=["read", "write"]
        )

        mock_request.headers = {"X-API-Key": result.raw_key}

        user = await get_current_user(mock_request, api_key_auth, jwt_auth)

        assert user.is_authenticated is True
        assert user.auth_method == "api_key"
        assert user.user_id == "user456"
        assert "read" in user.scopes
        assert "write" in user.scopes
        assert user.api_key_name == "Test Key"

        # Cleanup
        api_key_manager.revoke(result.key_info.key_id)


# ============================================================================
# get_current_user_optional Tests
# ============================================================================


class TestGetCurrentUserOptional:
    """Tests for get_current_user_optional dependency."""

    @pytest.mark.asyncio
    async def test_returns_none_when_auth_fails(self, mock_request):
        """Test None is returned instead of exception when auth fails."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "true"
        os.environ["ENABLE_JWT_AUTH"] = "false"
        os.environ["VALID_API_KEYS"] = "test-api-key-123"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            mock_request.headers = {"X-API-Key": "invalid-key"}

            user = await get_current_user_optional(mock_request, api_key_auth, jwt_auth)

            assert user is None
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["VALID_API_KEYS"]

    @pytest.mark.asyncio
    async def test_returns_user_when_auth_succeeds(self, mock_request):
        """Test user is returned when auth succeeds."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "false"
        os.environ["ENABLE_JWT_AUTH"] = "true"
        os.environ["JWT_SECRET"] = "test-jwt-secret-key"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            token = jwt_auth.create_token(user_id="user123")
            mock_request.headers = {"Authorization": f"Bearer {token}"}

            user = await get_current_user_optional(mock_request, api_key_auth, jwt_auth)

            assert user is not None
            assert user.user_id == "user123"
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["JWT_SECRET"]


# ============================================================================
# require_roles Tests
# ============================================================================


class TestRequireRoles:
    """Tests for require_roles factory."""

    @pytest.mark.asyncio
    async def test_role_check_success(self):
        """Test successful role check."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin", "user"],
        )

        # The require_roles factory returns an async function, call it directly
        checker = require_roles("admin")
        # Use the inner function directly by simulating the dependency injection

        async def fake_get_user():
            return user

        # Call checker with the user directly
        result = await _call_role_checker(checker, user)
        assert result.user_id == "user123"

    @pytest.mark.asyncio
    async def test_role_check_any_match(self):
        """Test role check passes with any matching role."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["user"],
        )

        checker = require_roles("admin", "user")
        result = await _call_role_checker(checker, user)
        assert result.user_id == "user123"

    @pytest.mark.asyncio
    async def test_role_check_failure(self):
        """Test failed role check."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["user"],
        )

        checker = require_roles("admin")
        with pytest.raises(AuthorizationError) as exc_info:
            await _call_role_checker(checker, user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_role_check_unauthenticated(self):
        """Test role check fails for unauthenticated user."""
        user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )

        checker = require_roles("admin")
        with pytest.raises(AuthenticationError) as exc_info:
            await _call_role_checker(checker, user)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_roles_requirement(self):
        """Test that empty roles requirement passes."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="jwt",
            roles=[],
        )

        checker = require_roles()
        result = await _call_role_checker(checker, user)
        assert result.user_id == "user123"


async def _call_role_checker(checker, user: CurrentUser) -> CurrentUser:
    """Helper to call role/scope checker with a user directly."""
    # The checker function expects a CurrentUser as its first positional arg
    # In the actual middleware, this is injected via Depends(get_current_user)
    return await checker(user=user)


# ============================================================================
# require_scopes Tests
# ============================================================================


class TestRequireScopes:
    """Tests for require_scopes factory."""

    @pytest.mark.asyncio
    async def test_scope_check_success(self):
        """Test successful scope check."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="api_key",
            scopes=["read", "write", "deploy"],
        )

        checker = require_scopes("read", "write")
        result = await _call_scope_checker(checker, user)
        assert result.user_id == "user123"

    @pytest.mark.asyncio
    async def test_scope_check_failure_missing_scope(self):
        """Test failed scope check when scope is missing."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="api_key",
            scopes=["read"],
        )

        checker = require_scopes("read", "write")
        with pytest.raises(AuthorizationError) as exc_info:
            await _call_scope_checker(checker, user)
        assert exc_info.value.status_code == 403
        assert "write" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_scope_check_unauthenticated(self):
        """Test scope check fails for unauthenticated user."""
        user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )

        checker = require_scopes("read")
        with pytest.raises(AuthenticationError) as exc_info:
            await _call_scope_checker(checker, user)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_scopes_requirement(self):
        """Test that empty scopes requirement passes."""
        user = CurrentUser(
            user_id="user123",
            is_authenticated=True,
            auth_method="api_key",
            scopes=[],
        )

        checker = require_scopes()
        result = await _call_scope_checker(checker, user)
        assert result.user_id == "user123"


async def _call_scope_checker(checker, user: CurrentUser) -> CurrentUser:
    """Helper to call scope checker with a user directly."""
    return await checker(user=user)


# ============================================================================
# Exception Tests
# ============================================================================


class TestExceptions:
    """Tests for custom exception classes."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Custom message")
        assert error.status_code == 401
        assert error.detail == "Custom message"
        assert "WWW-Authenticate" in error.headers

    def test_authentication_error_default_message(self):
        """Test AuthenticationError with default message."""
        error = AuthenticationError()
        assert error.detail == "Authentication required"

    def test_authorization_error(self):
        """Test AuthorizationError exception."""
        error = AuthorizationError("Custom message")
        assert error.status_code == 403
        assert error.detail == "Custom message"

    def test_authorization_error_default_message(self):
        """Test AuthorizationError with default message."""
        error = AuthorizationError()
        assert error.detail == "Insufficient permissions"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for middleware with real FastAPI components."""

    @pytest.mark.asyncio
    async def test_full_auth_flow_api_key(self, mock_request, auth_disabled_env):
        """Test complete auth flow with API key manager."""
        from security import APIKeyAuth, JWTAuth
        from security.api_keys import api_key_manager

        config = SecurityConfig()
        api_key_auth = APIKeyAuth(config=config)
        jwt_auth = JWTAuth(config=config)

        # Generate key with specific scopes
        result = api_key_manager.generate(
            name="Deploy Key",
            user_id="deployer",
            scopes=["deploy:read", "deploy:write"],
        )

        mock_request.headers = {"X-API-Key": result.raw_key}

        # Authenticate
        user = await get_current_user(mock_request, api_key_auth, jwt_auth)
        assert user.is_authenticated
        assert user.user_id == "deployer"

        # Check scopes
        scope_checker = require_scopes("deploy:read")
        result_user = await _call_scope_checker(scope_checker, user)
        assert result_user.user_id == "deployer"

        # Cleanup
        api_key_manager.revoke(result.key_info.key_id)

    @pytest.mark.asyncio
    async def test_full_auth_flow_jwt(self, mock_request):
        """Test complete auth flow with JWT."""
        from security import APIKeyAuth, JWTAuth

        os.environ["ENABLE_API_KEY_AUTH"] = "false"
        os.environ["ENABLE_JWT_AUTH"] = "true"
        os.environ["JWT_SECRET"] = "test-jwt-secret-key"

        try:
            config = SecurityConfig()
            api_key_auth = APIKeyAuth(config=config)
            jwt_auth = JWTAuth(config=config)

            # Create token with roles
            token = jwt_auth.create_token(
                user_id="admin-user",
                roles=["admin", "user"],
                session_ids=["session-123"],
            )

            mock_request.headers = {"Authorization": f"Bearer {token}"}

            # Authenticate
            user = await get_current_user(mock_request, api_key_auth, jwt_auth)
            assert user.is_authenticated
            assert user.user_id == "admin-user"
            assert user.auth_method == "jwt"

            # Check roles
            role_checker = require_roles("admin")
            result_user = await _call_role_checker(role_checker, user)
            assert result_user.user_id == "admin-user"
        finally:
            del os.environ["ENABLE_API_KEY_AUTH"]
            del os.environ["ENABLE_JWT_AUTH"]
            del os.environ["JWT_SECRET"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
