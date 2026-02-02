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

# Set auth disabled, high rate limit, and JWT secret before importing api_server
os.environ["ENABLE_API_KEY_AUTH"] = "false"
os.environ["ENABLE_JWT_AUTH"] = "false"
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ["JWT_SECRET"] = "test-secret-for-all-tests"

from api_server import app, limiter, user_store
from security.api_keys import api_key_manager


@pytest.fixture
def client():
    """Create a test client with auth disabled (anonymous access allowed)."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset user store, api key manager, and rate limiter before each test."""
    user_store._users.clear()
    user_store._next_id = 1
    api_key_manager._keys.clear()
    api_key_manager._revoked_hashes.clear()
    limiter.reset()
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
    def admin_client(self, client):
        """Create a client with admin JWT token."""
        from security import JWTAuth, SecurityConfig

        config = SecurityConfig()
        jwt_auth = JWTAuth(config=config)
        # Create admin user in store first
        user_store.create_user(
            username="admin",
            email="admin@example.com",
            password="adminpass123",
            is_admin=True,
        )
        token = jwt_auth.create_token(user_id="1", roles=["admin"])
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    def test_create_user_success(self, admin_client):
        """Test creating a user with admin privileges."""
        response = admin_client.post(
            "/admin/users",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "securepassword123",
                "is_admin": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert data["is_active"] is True
        assert data["is_admin"] is False

    def test_create_admin_user(self, admin_client):
        """Test creating an admin user."""
        response = admin_client.post(
            "/admin/users",
            json={
                "username": "newadmin",
                "email": "newadmin@example.com",
                "password": "securepassword123",
                "is_admin": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newadmin"
        assert data["is_admin"] is True

    def test_create_user_duplicate_username(self, admin_client):
        """Test creating a user with duplicate username fails."""
        # Create first user
        admin_client.post(
            "/admin/users",
            json={
                "username": "duplicateuser",
                "email": "first@example.com",
                "password": "securepassword123",
            },
        )

        # Try to create second user with same username
        response = admin_client.post(
            "/admin/users",
            json={
                "username": "duplicateuser",
                "email": "second@example.com",
                "password": "securepassword123",
            },
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_user_duplicate_email(self, admin_client):
        """Test creating a user with duplicate email fails."""
        # Create first user
        admin_client.post(
            "/admin/users",
            json={
                "username": "firstuser",
                "email": "duplicate@example.com",
                "password": "securepassword123",
            },
        )

        # Try to create second user with same email
        response = admin_client.post(
            "/admin/users",
            json={
                "username": "seconduser",
                "email": "duplicate@example.com",
                "password": "securepassword123",
            },
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]


class TestCreateAPIKeyEndpoint:
    """Tests for POST /admin/keys endpoint."""

    @pytest.fixture
    def admin_client(self, client):
        """Create a client with admin JWT token."""
        from security import JWTAuth, SecurityConfig

        config = SecurityConfig()
        jwt_auth = JWTAuth(config=config)
        # Create admin user in store first
        user_store.create_user(
            username="admin",
            email="admin@example.com",
            password="adminpass123",
            is_admin=True,
        )
        token = jwt_auth.create_token(user_id="1", roles=["admin"])
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    def test_create_api_key_success(self, admin_client):
        """Test creating an API key with admin privileges."""
        response = admin_client.post(
            "/admin/keys",
            json={
                "name": "My API Key",
                "user_id": "user-123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "My API Key"
        assert data["user_id"] == "user-123"
        assert data["raw_key"] is not None
        assert len(data["raw_key"]) > 20  # Should be a long secure key
        assert data["key_id"] is not None

    def test_create_api_key_with_expiration(self, admin_client):
        """Test creating an API key with expiration."""
        response = admin_client.post(
            "/admin/keys",
            json={
                "name": "Expiring Key",
                "expires_in_days": 30,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["expires_at"] is not None

    def test_create_api_key_with_scopes(self, admin_client):
        """Test creating an API key with scopes."""
        response = admin_client.post(
            "/admin/keys",
            json={
                "name": "Scoped Key",
                "scopes": ["read", "write"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["key_id"] is not None

        # Verify the key has scopes
        key_info = api_key_manager.get_key_info(data["key_id"])
        assert "read" in key_info.scopes
        assert "write" in key_info.scopes

    def test_created_api_key_is_valid(self, admin_client):
        """Test that created API key can be verified."""
        response = admin_client.post(
            "/admin/keys",
            json={
                "name": "Verifiable Key",
                "user_id": "test-user",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify the key works
        key_info = api_key_manager.verify(data["raw_key"])
        assert key_info is not None
        assert key_info.name == "Verifiable Key"
        assert key_info.user_id == "test-user"


class TestListUsersEndpoint:
    """Tests for GET /admin/users endpoint."""

    @pytest.fixture
    def admin_client(self, client):
        """Create a client with admin JWT token."""
        from security import JWTAuth, SecurityConfig

        config = SecurityConfig()
        jwt_auth = JWTAuth(config=config)
        # Create admin user in store first
        user_store.create_user(
            username="admin",
            email="admin@example.com",
            password="adminpass123",
            is_admin=True,
        )
        token = jwt_auth.create_token(user_id="1", roles=["admin"])
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    def test_list_users_success(self, admin_client):
        """Test listing users with admin privileges."""
        # Create some additional users
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

        response = admin_client.get("/admin/users")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3  # admin + 2 users
        usernames = [u["username"] for u in data]
        assert "admin" in usernames
        assert "user1" in usernames
        assert "user2" in usernames

    def test_list_users_empty(self, admin_client):
        """Test listing users when only admin exists."""
        response = admin_client.get("/admin/users")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # Just the admin
        assert data[0]["username"] == "admin"

    def test_list_users_requires_admin(self, client):
        """Test that list users requires admin privileges."""
        response = client.get("/admin/users")
        assert response.status_code == 403


class TestListAPIKeysEndpoint:
    """Tests for GET /admin/keys endpoint."""

    @pytest.fixture
    def admin_client(self, client):
        """Create a client with admin JWT token."""
        from security import JWTAuth, SecurityConfig

        config = SecurityConfig()
        jwt_auth = JWTAuth(config=config)
        # Create admin user in store first
        user_store.create_user(
            username="admin",
            email="admin@example.com",
            password="adminpass123",
            is_admin=True,
        )
        token = jwt_auth.create_token(user_id="1", roles=["admin"])
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    def test_list_keys_success(self, admin_client):
        """Test listing API keys with admin privileges."""
        # Create some keys
        api_key_manager.generate(name="Key 1", user_id="user-1")
        api_key_manager.generate(name="Key 2", user_id="user-2")

        response = admin_client.get("/admin/keys")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        names = [k["name"] for k in data]
        assert "Key 1" in names
        assert "Key 2" in names

    def test_list_keys_empty(self, admin_client):
        """Test listing keys when none exist."""
        response = admin_client.get("/admin/keys")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    def test_list_keys_filter_by_user(self, admin_client):
        """Test listing API keys filtered by user."""
        api_key_manager.generate(name="Key 1", user_id="user-1")
        api_key_manager.generate(name="Key 2", user_id="user-2")
        api_key_manager.generate(name="Key 3", user_id="user-1")

        response = admin_client.get("/admin/keys", params={"user_id": "user-1"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for key in data:
            assert key["user_id"] == "user-1"

    def test_list_keys_include_revoked(self, admin_client):
        """Test listing API keys including revoked ones."""
        api_key_manager.generate(name="Active Key")
        result2 = api_key_manager.generate(name="Revoked Key")
        api_key_manager.revoke(result2.key_info.key_id)

        # Without include_revoked
        response = admin_client.get("/admin/keys")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Active Key"

        # With include_revoked
        response = admin_client.get("/admin/keys", params={"include_revoked": "true"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_keys_requires_admin(self, client):
        """Test that list keys requires admin privileges."""
        response = client.get("/admin/keys")
        assert response.status_code == 403


class TestRevokeAPIKeyEndpoint:
    """Tests for DELETE /admin/keys/{id} endpoint."""

    @pytest.fixture
    def admin_client(self, client):
        """Create a client with admin JWT token."""
        from security import JWTAuth, SecurityConfig

        config = SecurityConfig()
        jwt_auth = JWTAuth(config=config)
        # Create admin user in store first
        user_store.create_user(
            username="admin",
            email="admin@example.com",
            password="adminpass123",
            is_admin=True,
        )
        token = jwt_auth.create_token(user_id="1", roles=["admin"])
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    def test_revoke_api_key_success(self, admin_client):
        """Test revoking an API key with admin privileges."""
        # Create a key first
        result = api_key_manager.generate(name="Key to Revoke")
        key_id = result.key_info.key_id

        # Revoke it
        response = admin_client.delete(f"/admin/keys/{key_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "revoked" in data["message"]

        # Verify key is no longer valid
        key_info = api_key_manager.verify(result.raw_key)
        assert key_info is None

    def test_revoke_nonexistent_key(self, admin_client):
        """Test revoking a non-existent key returns 404."""
        response = admin_client.delete("/admin/keys/nonexistent-key-id")
        assert response.status_code == 404

    def test_revoke_already_revoked_key(self, admin_client):
        """Test revoking an already revoked key returns 400."""
        # Create and revoke a key
        result = api_key_manager.generate(name="Already Revoked Key")
        key_id = result.key_info.key_id
        api_key_manager.revoke(key_id)

        # Try to revoke again
        response = admin_client.delete(f"/admin/keys/{key_id}")
        assert response.status_code == 400
        assert "already revoked" in response.json()["detail"]


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
