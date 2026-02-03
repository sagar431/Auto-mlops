"""
Unit tests for security middleware module.

Tests CurrentUser model, authentication dependencies, and authorization checks.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import status

from security.middleware import (
    AuthenticationError,
    AuthorizationError,
    CurrentUser,
    get_current_user,
    get_current_user_optional,
    require_roles,
    require_scopes,
)


class TestCurrentUser:
    """Tests for CurrentUser model."""

    def test_create_current_user_minimal(self):
        """Test creating CurrentUser with minimal fields."""
        user = CurrentUser(
            user_id="user-123",
            auth_method="api_key",
        )
        assert user.user_id == "user-123"
        assert user.auth_method == "api_key"
        assert user.is_authenticated is True
        assert user.roles == []
        assert user.scopes == []
        assert user.session_ids == []
        assert user.api_key_id is None

    def test_create_current_user_full(self):
        """Test creating CurrentUser with all fields."""
        user = CurrentUser(
            user_id="user-123",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin", "user"],
            scopes=["read", "write"],
            session_ids=["session-1", "session-2"],
            api_key_id="key-001",
            api_key_name="My API Key",
        )
        assert user.user_id == "user-123"
        assert user.is_authenticated is True
        assert user.auth_method == "jwt"
        assert "admin" in user.roles
        assert "write" in user.scopes
        assert len(user.session_ids) == 2
        assert user.api_key_id == "key-001"

    def test_create_anonymous_user(self):
        """Test creating anonymous user."""
        user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )
        assert user.user_id == "anonymous"
        assert user.is_authenticated is False
        assert user.auth_method == "anonymous"


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = AuthenticationError()
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Authentication required"
        assert "WWW-Authenticate" in error.headers

    def test_custom_message(self):
        """Test custom error message."""
        error = AuthenticationError("Invalid token")
        assert error.detail == "Invalid token"


class TestAuthorizationError:
    """Tests for AuthorizationError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = AuthorizationError()
        assert error.status_code == status.HTTP_403_FORBIDDEN
        assert error.detail == "Insufficient permissions"

    def test_custom_message(self):
        """Test custom error message."""
        error = AuthorizationError("Admin role required")
        assert error.detail == "Admin role required"


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        return request

    @pytest.fixture
    def mock_api_key_auth(self):
        """Create a mock API key authenticator."""
        auth = MagicMock()
        auth.config = MagicMock()
        auth.config.enable_api_key_auth = False
        auth.config.api_key_header_name = "X-API-Key"
        auth.validate_key = MagicMock(return_value=False)
        return auth

    @pytest.fixture
    def mock_jwt_auth(self):
        """Create a mock JWT authenticator."""
        auth = MagicMock()
        auth.config = MagicMock()
        auth.config.enable_jwt_auth = False
        auth.verify_token = MagicMock(return_value=None)
        return auth

    @pytest.mark.asyncio
    async def test_anonymous_when_auth_disabled(
        self, mock_request, mock_api_key_auth, mock_jwt_auth
    ):
        """Test returns anonymous user when auth is disabled."""
        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user.user_id == "anonymous"
        assert user.is_authenticated is False
        assert user.auth_method == "anonymous"

    @pytest.mark.asyncio
    async def test_api_key_auth_via_manager(self, mock_request, mock_api_key_auth, mock_jwt_auth):
        """Test API key authentication via APIKeyManager."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "valid-key" if h == "X-API-Key" else None
        )

        mock_key_info = MagicMock()
        mock_key_info.key_id = "key-001"
        mock_key_info.user_id = "user-123"
        mock_key_info.name = "Test Key"
        mock_key_info.scopes = ["read", "write"]

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = mock_key_info

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user.user_id == "user-123"
        assert user.is_authenticated is True
        assert user.auth_method == "api_key"
        assert user.api_key_id == "key-001"
        assert "read" in user.scopes

    @pytest.mark.asyncio
    async def test_api_key_auth_via_legacy_system(
        self, mock_request, mock_api_key_auth, mock_jwt_auth
    ):
        """Test API key authentication via legacy APIKeyAuth."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "valid-key" if h == "X-API-Key" else None
        )
        mock_api_key_auth.validate_key.return_value = True

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user.is_authenticated is True
        assert user.auth_method == "api_key"
        assert user.user_id.startswith("apikey:")

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_error(
        self, mock_request, mock_api_key_auth, mock_jwt_auth
    ):
        """Test invalid API key raises AuthenticationError."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "invalid-key" if h == "X-API-Key" else None
        )
        mock_api_key_auth.config.enable_api_key_auth = True
        mock_api_key_auth.validate_key.return_value = False

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert "Invalid API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_jwt_auth_success(self, mock_request, mock_api_key_auth, mock_jwt_auth):
        """Test JWT authentication success."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "Bearer valid.jwt.token" if h == "Authorization" else None
        )

        mock_payload = MagicMock()
        mock_payload.sub = "user-456"
        mock_payload.roles = ["admin"]
        mock_payload.session_ids = ["session-1"]
        mock_jwt_auth.verify_token.return_value = mock_payload

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user.user_id == "user-456"
        assert user.is_authenticated is True
        assert user.auth_method == "jwt"
        assert "admin" in user.roles

    @pytest.mark.asyncio
    async def test_invalid_jwt_raises_error(self, mock_request, mock_api_key_auth, mock_jwt_auth):
        """Test invalid JWT raises AuthenticationError."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "Bearer invalid.jwt.token" if h == "Authorization" else None
        )
        mock_jwt_auth.config.enable_jwt_auth = True
        mock_jwt_auth.verify_token.return_value = None

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert "Invalid or expired token" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_auth_required_no_credentials(
        self, mock_request, mock_api_key_auth, mock_jwt_auth
    ):
        """Test error when auth required but no credentials provided."""
        mock_api_key_auth.config.enable_api_key_auth = True

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert "Authentication required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_api_key_with_no_user_id(self, mock_request, mock_api_key_auth, mock_jwt_auth):
        """Test API key without user_id uses key_id."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "valid-key" if h == "X-API-Key" else None
        )

        mock_key_info = MagicMock()
        mock_key_info.key_id = "key-001"
        mock_key_info.user_id = None  # No user_id
        mock_key_info.name = "Test Key"
        mock_key_info.scopes = []

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = mock_key_info

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user.user_id == "apikey:key-001"


class TestGetCurrentUserOptional:
    """Tests for get_current_user_optional dependency."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        return request

    @pytest.fixture
    def mock_api_key_auth(self):
        """Create a mock API key authenticator."""
        auth = MagicMock()
        auth.config = MagicMock()
        auth.config.enable_api_key_auth = True
        auth.config.api_key_header_name = "X-API-Key"
        auth.validate_key = MagicMock(return_value=False)
        return auth

    @pytest.fixture
    def mock_jwt_auth(self):
        """Create a mock JWT authenticator."""
        auth = MagicMock()
        auth.config = MagicMock()
        auth.config.enable_jwt_auth = False
        auth.verify_token = MagicMock(return_value=None)
        return auth

    @pytest.mark.asyncio
    async def test_returns_user_when_authenticated(
        self, mock_request, mock_api_key_auth, mock_jwt_auth
    ):
        """Test returns user when authentication succeeds."""
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: "valid-key" if h == "X-API-Key" else None
        )

        mock_key_info = MagicMock()
        mock_key_info.key_id = "key-001"
        mock_key_info.user_id = "user-123"
        mock_key_info.name = "Test"
        mock_key_info.scopes = []

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = mock_key_info

            user = await get_current_user_optional(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user is not None
        assert user.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_returns_none_when_auth_fails(
        self, mock_request, mock_api_key_auth, mock_jwt_auth
    ):
        """Test returns None when authentication fails."""
        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            user = await get_current_user_optional(mock_request, mock_api_key_auth, mock_jwt_auth)

        assert user is None


class TestRequireRoles:
    """Tests for require_roles dependency factory."""

    @pytest.mark.asyncio
    async def test_user_with_required_role_passes(self):
        """Test user with required role passes."""
        checker = require_roles("admin")
        user = CurrentUser(
            user_id="user-123",
            auth_method="jwt",
            roles=["admin", "user"],
        )

        result = await checker(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_without_required_role_fails(self):
        """Test user without required role fails."""
        checker = require_roles("admin")
        user = CurrentUser(
            user_id="user-123",
            auth_method="jwt",
            roles=["user"],
        )

        with pytest.raises(AuthorizationError) as exc_info:
            await checker(user)

        assert "admin" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_multiple_roles_any_match(self):
        """Test user needs any of multiple required roles."""
        checker = require_roles("admin", "moderator")
        user = CurrentUser(
            user_id="user-123",
            auth_method="jwt",
            roles=["moderator"],  # Has one of the required roles
        )

        result = await checker(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_unauthenticated_user_fails(self):
        """Test unauthenticated user fails."""
        checker = require_roles("admin")
        user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )

        with pytest.raises(AuthenticationError):
            await checker(user)

    @pytest.mark.asyncio
    async def test_no_required_roles_passes(self):
        """Test no required roles allows any authenticated user."""
        checker = require_roles()  # No roles required
        user = CurrentUser(
            user_id="user-123",
            auth_method="jwt",
            roles=[],
        )

        result = await checker(user)
        assert result == user


class TestRequireScopes:
    """Tests for require_scopes dependency factory."""

    @pytest.mark.asyncio
    async def test_user_with_required_scopes_passes(self):
        """Test user with required scopes passes."""
        checker = require_scopes("read", "write")
        user = CurrentUser(
            user_id="user-123",
            auth_method="api_key",
            scopes=["read", "write", "delete"],
        )

        result = await checker(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_missing_scope_fails(self):
        """Test user missing required scope fails."""
        checker = require_scopes("read", "write", "admin")
        user = CurrentUser(
            user_id="user-123",
            auth_method="api_key",
            scopes=["read", "write"],  # Missing "admin"
        )

        with pytest.raises(AuthorizationError) as exc_info:
            await checker(user)

        assert "admin" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_unauthenticated_user_fails(self):
        """Test unauthenticated user fails."""
        checker = require_scopes("read")
        user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )

        with pytest.raises(AuthenticationError):
            await checker(user)

    @pytest.mark.asyncio
    async def test_no_required_scopes_passes(self):
        """Test no required scopes allows any authenticated user."""
        checker = require_scopes()
        user = CurrentUser(
            user_id="user-123",
            auth_method="api_key",
            scopes=[],
        )

        result = await checker(user)
        assert result == user


class TestMiddlewareEdgeCases:
    """Tests for edge cases in middleware."""

    def test_current_user_model_validation(self):
        """Test CurrentUser model validates fields."""
        # Should work with valid data
        user = CurrentUser(user_id="test", auth_method="jwt")
        assert user.user_id == "test"

    def test_authentication_error_headers(self):
        """Test AuthenticationError includes proper headers."""
        error = AuthenticationError()
        assert "Bearer" in error.headers["WWW-Authenticate"]
        assert "ApiKey" in error.headers["WWW-Authenticate"]

    @pytest.mark.asyncio
    async def test_bearer_prefix_stripped(self):
        """Test that 'Bearer ' prefix is properly stripped from token."""
        mock_request = MagicMock()
        mock_request.headers = MagicMock()

        def get_header(name):
            if name == "Authorization":
                return "Bearer mytoken123"
            return None

        mock_request.headers.get = get_header

        mock_api_key_auth = MagicMock()
        mock_api_key_auth.config = MagicMock()
        mock_api_key_auth.config.enable_api_key_auth = False
        mock_api_key_auth.config.api_key_header_name = "X-API-Key"

        mock_jwt_auth = MagicMock()
        mock_jwt_auth.config = MagicMock()
        mock_jwt_auth.config.enable_jwt_auth = True

        mock_payload = MagicMock()
        mock_payload.sub = "user-123"
        mock_payload.roles = []
        mock_payload.session_ids = []
        mock_jwt_auth.verify_token = MagicMock(return_value=mock_payload)

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        # Verify token was called with stripped value
        mock_jwt_auth.verify_token.assert_called_once_with("mytoken123")
        assert user.auth_method == "jwt"

    @pytest.mark.asyncio
    async def test_non_bearer_auth_header_ignored(self):
        """Test non-Bearer Authorization header is ignored."""
        mock_request = MagicMock()
        mock_request.headers = MagicMock()

        def get_header(name):
            if name == "Authorization":
                return "Basic dXNlcjpwYXNz"  # Basic auth, not Bearer
            return None

        mock_request.headers.get = get_header

        mock_api_key_auth = MagicMock()
        mock_api_key_auth.config = MagicMock()
        mock_api_key_auth.config.enable_api_key_auth = False
        mock_api_key_auth.config.api_key_header_name = "X-API-Key"

        mock_jwt_auth = MagicMock()
        mock_jwt_auth.config = MagicMock()
        mock_jwt_auth.config.enable_jwt_auth = False

        with patch("security.middleware.api_key_manager") as mock_manager:
            mock_manager.verify.return_value = None

            user = await get_current_user(mock_request, mock_api_key_auth, mock_jwt_auth)

        # Should return anonymous since Basic auth is not supported
        assert user.auth_method == "anonymous"
