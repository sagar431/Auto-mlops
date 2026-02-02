#!/usr/bin/env python3
"""
Tests for admin API endpoints.

Tests verify that:
1. Admin endpoints require authentication
2. Admin endpoints require admin privileges
3. POST /admin/users creates users correctly
4. POST /admin/keys creates API keys correctly
5. DELETE /admin/keys/{id} revokes API keys correctly

Run with: pytest test_admin_endpoints.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

# Set auth disabled before importing api_server
os.environ["ENABLE_API_KEY_AUTH"] = "false"
os.environ["ENABLE_JWT_AUTH"] = "false"

from api_server import app, user_store
from security.api_keys import api_key_manager


@pytest.fixture
def client():
    """Create a test client with auth disabled (anonymous access allowed)."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset user store and api key manager before each test."""
    user_store._users.clear()
    user_store._next_id = 1
    api_key_manager._keys.clear()
    api_key_manager._revoked_hashes.clear()
    yield


class TestAdminEndpointSignatures:
    """Tests that verify admin endpoints have correct dependencies."""

    def test_create_user_has_current_user_dependency(self):
        """Verify POST /admin/users endpoint has current_user dependency."""
        import inspect

        from api_server import create_user

        sig = inspect.signature(create_user)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_create_api_key_has_current_user_dependency(self):
        """Verify POST /admin/keys endpoint has current_user dependency."""
        import inspect

        from api_server import create_api_key

        sig = inspect.signature(create_api_key)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_revoke_api_key_has_current_user_dependency(self):
        """Verify DELETE /admin/keys/{id} endpoint has current_user dependency."""
        import inspect

        from api_server import revoke_api_key

        sig = inspect.signature(revoke_api_key)
        params = list(sig.parameters.keys())
        assert "current_user" in params


class TestAdminEndpointsWithoutAuth:
    """Tests that admin endpoints fail without proper auth when auth is enabled."""

    def test_create_user_requires_admin(self, client):
        """Test POST /admin/users fails for non-admin users."""
        # With auth disabled, anonymous users are allowed but not admin
        response = client.post(
            "/admin/users",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "testpassword123",
            },
        )
        # Should fail with 403 because anonymous users are not admin
        assert response.status_code == 403

    def test_create_api_key_requires_admin(self, client):
        """Test POST /admin/keys fails for non-admin users."""
        response = client.post(
            "/admin/keys",
            json={"name": "Test Key"},
        )
        assert response.status_code == 403

    def test_revoke_api_key_requires_admin(self, client):
        """Test DELETE /admin/keys/{id} fails for non-admin users."""
        response = client.delete("/admin/keys/some-key-id")
        assert response.status_code == 403


class TestCreateUserEndpoint:
    """Tests for POST /admin/users endpoint."""

    @pytest.fixture
    def admin_api_key(self):
        """Create an admin API key for testing."""
        result = api_key_manager.generate(
            name="Admin Key",
            user_id="admin-user",
            scopes=["admin"],
        )
        # Create an admin user
        user_store.create_user(
            username="admin",
            email="admin@example.com",
            password="adminpass123",
            is_admin=True,
        )
        return result.raw_key

    @pytest.fixture
    def admin_headers(self, admin_api_key):
        """Headers with admin API key."""
        return {"X-API-Key": admin_api_key}

    @pytest.mark.asyncio
    async def test_create_user_success(self):
        """Test creating a user with admin privileges."""
        from api_server import CreateUserRequest, create_user
        from security.middleware import CurrentUser

        # Create an admin user first
        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        request = CreateUserRequest(
            username="newuser",
            email="newuser@example.com",
            password="securepassword123",
            is_admin=False,
        )

        response = await create_user(request=request, current_user=admin_user)
        assert response.username == "newuser"
        assert response.email == "newuser@example.com"
        assert response.is_active is True
        assert response.is_admin is False

    @pytest.mark.asyncio
    async def test_create_admin_user(self):
        """Test creating an admin user."""
        from api_server import CreateUserRequest, create_user
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        request = CreateUserRequest(
            username="newadmin",
            email="newadmin@example.com",
            password="securepassword123",
            is_admin=True,
        )

        response = await create_user(request=request, current_user=admin_user)
        assert response.username == "newadmin"
        assert response.is_admin is True

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self):
        """Test creating a user with duplicate username fails."""
        from fastapi import HTTPException

        from api_server import CreateUserRequest, create_user
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        # Create first user
        request1 = CreateUserRequest(
            username="duplicateuser",
            email="first@example.com",
            password="securepassword123",
        )
        await create_user(request=request1, current_user=admin_user)

        # Try to create second user with same username
        request2 = CreateUserRequest(
            username="duplicateuser",
            email="second@example.com",
            password="securepassword123",
        )
        with pytest.raises(HTTPException) as exc_info:
            await create_user(request=request2, current_user=admin_user)
        assert exc_info.value.status_code == 400
        assert "already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self):
        """Test creating a user with duplicate email fails."""
        from fastapi import HTTPException

        from api_server import CreateUserRequest, create_user
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        # Create first user
        request1 = CreateUserRequest(
            username="firstuser",
            email="duplicate@example.com",
            password="securepassword123",
        )
        await create_user(request=request1, current_user=admin_user)

        # Try to create second user with same email
        request2 = CreateUserRequest(
            username="seconduser",
            email="duplicate@example.com",
            password="securepassword123",
        )
        with pytest.raises(HTTPException) as exc_info:
            await create_user(request=request2, current_user=admin_user)
        assert exc_info.value.status_code == 400
        assert "already exists" in str(exc_info.value.detail)


class TestCreateAPIKeyEndpoint:
    """Tests for POST /admin/keys endpoint."""

    @pytest.mark.asyncio
    async def test_create_api_key_success(self):
        """Test creating an API key with admin privileges."""
        from api_server import CreateAPIKeyRequest, create_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        request = CreateAPIKeyRequest(
            name="My API Key",
            user_id="user-123",
        )

        response = await create_api_key(request=request, current_user=admin_user)
        assert response.name == "My API Key"
        assert response.user_id == "user-123"
        assert response.raw_key is not None
        assert len(response.raw_key) > 20  # Should be a long secure key
        assert response.key_id is not None

    @pytest.mark.asyncio
    async def test_create_api_key_with_expiration(self):
        """Test creating an API key with expiration."""
        from api_server import CreateAPIKeyRequest, create_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        request = CreateAPIKeyRequest(
            name="Expiring Key",
            expires_in_days=30,
        )

        response = await create_api_key(request=request, current_user=admin_user)
        assert response.expires_at is not None

    @pytest.mark.asyncio
    async def test_create_api_key_with_scopes(self):
        """Test creating an API key with scopes."""
        from api_server import CreateAPIKeyRequest, create_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        request = CreateAPIKeyRequest(
            name="Scoped Key",
            scopes=["read", "write"],
        )

        response = await create_api_key(request=request, current_user=admin_user)
        assert response.key_id is not None

        # Verify the key has scopes
        key_info = api_key_manager.get_key_info(response.key_id)
        assert "read" in key_info.scopes
        assert "write" in key_info.scopes

    @pytest.mark.asyncio
    async def test_created_api_key_is_valid(self):
        """Test that created API key can be verified."""
        from api_server import CreateAPIKeyRequest, create_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        request = CreateAPIKeyRequest(
            name="Verifiable Key",
            user_id="test-user",
        )

        response = await create_api_key(request=request, current_user=admin_user)

        # Verify the key works
        key_info = api_key_manager.verify(response.raw_key)
        assert key_info is not None
        assert key_info.name == "Verifiable Key"
        assert key_info.user_id == "test-user"


class TestRevokeAPIKeyEndpoint:
    """Tests for DELETE /admin/keys/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_revoke_api_key_success(self):
        """Test revoking an API key with admin privileges."""
        from api_server import revoke_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        # Create a key first
        result = api_key_manager.generate(name="Key to Revoke")
        key_id = result.key_info.key_id

        # Revoke it
        response = await revoke_api_key(key_id=key_id, current_user=admin_user)
        assert response["status"] == "ok"
        assert "revoked" in response["message"]

        # Verify key is no longer valid
        key_info = api_key_manager.verify(result.raw_key)
        assert key_info is None

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key(self):
        """Test revoking a non-existent key returns 404."""
        from fastapi import HTTPException

        from api_server import revoke_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await revoke_api_key(key_id="nonexistent-key-id", current_user=admin_user)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_revoke_already_revoked_key(self):
        """Test revoking an already revoked key returns 400."""
        from fastapi import HTTPException

        from api_server import revoke_api_key
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        # Create and revoke a key
        result = api_key_manager.generate(name="Already Revoked Key")
        key_id = result.key_info.key_id
        api_key_manager.revoke(key_id)

        # Try to revoke again
        with pytest.raises(HTTPException) as exc_info:
            await revoke_api_key(key_id=key_id, current_user=admin_user)
        assert exc_info.value.status_code == 400
        assert "already revoked" in str(exc_info.value.detail)


class TestRequireAdminDependency:
    """Tests for the require_admin dependency."""

    @pytest.mark.asyncio
    async def test_admin_role_grants_access(self):
        """Test that admin role in JWT grants access."""
        from api_server import require_admin
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="user-1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
        )

        result = await require_admin(current_user=admin_user)
        assert result.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_admin_user_in_store_grants_access(self):
        """Test that admin flag in user store grants access."""
        from api_server import require_admin
        from security.middleware import CurrentUser

        # Create admin user in store
        user_store.create_user(
            username="adminuser",
            email="admin@test.com",
            password="password123",
            is_admin=True,
        )

        admin_user = CurrentUser(
            user_id="1",  # Matches the user ID in store
            is_authenticated=True,
            auth_method="jwt",
            roles=[],  # No admin role, but user is admin in store
        )

        result = await require_admin(current_user=admin_user)
        assert result.user_id == "1"

    @pytest.mark.asyncio
    async def test_non_admin_denied(self):
        """Test that non-admin users are denied."""
        from api_server import require_admin
        from security.middleware import AuthorizationError, CurrentUser

        regular_user = CurrentUser(
            user_id="user-1",
            is_authenticated=True,
            auth_method="jwt",
            roles=["user"],  # Not admin
        )

        with pytest.raises(AuthorizationError):
            await require_admin(current_user=regular_user)

    @pytest.mark.asyncio
    async def test_unauthenticated_denied(self):
        """Test that unauthenticated users are denied."""
        from api_server import require_admin
        from security.middleware import AuthorizationError, CurrentUser

        anon_user = CurrentUser(
            user_id="anonymous",
            is_authenticated=False,
            auth_method="anonymous",
        )

        with pytest.raises(AuthorizationError):
            await require_admin(current_user=anon_user)


class TestUserStore:
    """Tests for the UserStore class."""

    def test_create_user(self):
        """Test creating a user."""
        user = user_store.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["is_active"] is True
        assert user["is_admin"] is False
        assert "hashed_password" in user
        assert user["hashed_password"] != "password123"  # Should be hashed

    def test_get_user(self):
        """Test getting a user by ID."""
        user = user_store.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        retrieved = user_store.get_user(user["id"])
        assert retrieved is not None
        assert retrieved["username"] == "testuser"

    def test_get_user_by_username(self):
        """Test getting a user by username."""
        user_store.create_user(
            username="findme",
            email="findme@example.com",
            password="password123",
        )
        retrieved = user_store.get_user_by_username("findme")
        assert retrieved is not None
        assert retrieved["email"] == "findme@example.com"

    def test_list_users(self):
        """Test listing all users."""
        user_store.create_user(
            username="user1",
            email="user1@example.com",
            password="password123",
        )
        user_store.create_user(
            username="user2",
            email="user2@example.com",
            password="password123",
        )
        users = user_store.list_users()
        assert len(users) == 2

    def test_duplicate_username_raises(self):
        """Test that duplicate username raises error."""
        user_store.create_user(
            username="duplicate",
            email="first@example.com",
            password="password123",
        )
        with pytest.raises(ValueError) as exc_info:
            user_store.create_user(
                username="duplicate",
                email="second@example.com",
                password="password123",
            )
        assert "already exists" in str(exc_info.value)

    def test_duplicate_email_raises(self):
        """Test that duplicate email raises error."""
        user_store.create_user(
            username="first",
            email="duplicate@example.com",
            password="password123",
        )
        with pytest.raises(ValueError) as exc_info:
            user_store.create_user(
                username="second",
                email="duplicate@example.com",
                password="password123",
            )
        assert "already exists" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
