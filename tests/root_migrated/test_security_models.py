#!/usr/bin/env python3
"""
Tests for the security models module.

Run with: pytest tests/root_migrated/test_security_models.py -v
"""

from datetime import datetime, timedelta

import pytest

from security.models import APIKey, User

# ============================================================================
# User Model Tests
# ============================================================================


class TestUserModel:
    """Tests for the User SQLModel."""

    def test_user_creation(self):
        """Test basic user creation with required fields."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=User.hash_password("password123"),
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.is_admin is False

    def test_user_default_values(self):
        """Test that default values are set correctly."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )
        assert user.id is None
        assert user.is_active is True
        assert user.is_admin is False
        assert user.api_keys == []

    def test_user_admin_flag(self):
        """Test setting admin flag."""
        user = User(
            username="admin",
            email="admin@example.com",
            hashed_password="hashed",
            is_admin=True,
        )
        assert user.is_admin is True

    def test_user_inactive(self):
        """Test creating inactive user."""
        user = User(
            username="inactive",
            email="inactive@example.com",
            hashed_password="hashed",
            is_active=False,
        )
        assert user.is_active is False

    def test_hash_password(self):
        """Test password hashing."""
        password = "secure_password_123"
        hashed = User.hash_password(password)

        assert hashed != password
        assert len(hashed) == 64  # SHA-256 produces 64 hex characters
        assert User.hash_password(password) == hashed  # Consistent

    def test_hash_password_different_inputs(self):
        """Test that different passwords produce different hashes."""
        hash1 = User.hash_password("password1")
        hash2 = User.hash_password("password2")
        assert hash1 != hash2

    def test_verify_password_success(self):
        """Test successful password verification."""
        password = "my_secret_password"
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=User.hash_password(password),
        )
        assert user.verify_password(password) is True

    def test_verify_password_failure(self):
        """Test failed password verification."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=User.hash_password("correct_password"),
        )
        assert user.verify_password("wrong_password") is False

    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=User.hash_password("password"),
        )
        assert user.verify_password("") is False

    def test_user_timestamps(self):
        """Test that timestamps are set on creation."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )
        assert user.created_at is not None
        assert user.updated_at is not None
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)


# ============================================================================
# APIKey Model Tests
# ============================================================================


class TestAPIKeyModel:
    """Tests for the APIKey SQLModel."""

    def test_api_key_creation(self):
        """Test basic API key creation."""
        key = APIKey(
            key_hash=APIKey.hash_key("test_key"),
            name="Test Key",
            user_id=1,
        )
        assert key.name == "Test Key"
        assert key.user_id == 1
        assert key.is_active is True
        assert key.expires_at is None

    def test_api_key_default_values(self):
        """Test that default values are set correctly."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
        )
        assert key.id is None
        assert key.is_active is True
        assert key.expires_at is None
        assert key.last_used_at is None

    def test_generate_key(self):
        """Test API key generation."""
        key1 = APIKey.generate_key()
        key2 = APIKey.generate_key()

        assert key1 is not None
        assert len(key1) > 20  # Should be reasonably long
        assert key1 != key2  # Should be unique

    def test_hash_key(self):
        """Test API key hashing."""
        raw_key = "my_api_key_12345"
        hashed = APIKey.hash_key(raw_key)

        assert hashed != raw_key
        assert len(hashed) == 64  # SHA-256 produces 64 hex characters
        assert APIKey.hash_key(raw_key) == hashed  # Consistent

    def test_hash_key_different_inputs(self):
        """Test that different keys produce different hashes."""
        hash1 = APIKey.hash_key("key1")
        hash2 = APIKey.hash_key("key2")
        assert hash1 != hash2

    def test_is_valid_active(self):
        """Test is_valid for active key without expiration."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
            is_active=True,
        )
        assert key.is_valid() is True

    def test_is_valid_inactive(self):
        """Test is_valid for inactive key."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
            is_active=False,
        )
        assert key.is_valid() is False

    def test_is_valid_expired(self):
        """Test is_valid for expired key."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
            is_active=True,
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired 1 hour ago
        )
        assert key.is_valid() is False

    def test_is_valid_not_expired(self):
        """Test is_valid for key with future expiration."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
            is_active=True,
            expires_at=datetime.utcnow() + timedelta(days=30),  # Expires in 30 days
        )
        assert key.is_valid() is True

    def test_update_last_used(self):
        """Test update_last_used method."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
        )
        assert key.last_used_at is None

        key.update_last_used()
        assert key.last_used_at is not None
        assert isinstance(key.last_used_at, datetime)

    def test_api_key_timestamps(self):
        """Test that created_at timestamp is set on creation."""
        key = APIKey(
            key_hash="hashed",
            name="Test",
            user_id=1,
        )
        assert key.created_at is not None
        assert isinstance(key.created_at, datetime)


# ============================================================================
# Integration Tests
# ============================================================================


class TestUserAPIKeyRelationship:
    """Tests for the User-APIKey relationship."""

    def test_user_with_api_keys(self):
        """Test user with API keys relationship."""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )

        # Simulate API keys (in real DB, these would be linked via foreign key)
        key1 = APIKey(
            id=1,
            key_hash=APIKey.hash_key("key1"),
            name="Key 1",
            user_id=1,
        )
        key2 = APIKey(
            id=2,
            key_hash=APIKey.hash_key("key2"),
            name="Key 2",
            user_id=1,
        )

        assert key1.user_id == user.id
        assert key2.user_id == user.id

    def test_api_key_create_and_validate_flow(self):
        """Test the complete flow of creating and validating an API key."""
        # Generate a new key
        raw_key = APIKey.generate_key()

        # Create the API key record with hashed key
        api_key = APIKey(
            key_hash=APIKey.hash_key(raw_key),
            name="Production Key",
            user_id=1,
            expires_at=datetime.utcnow() + timedelta(days=365),
        )

        # Verify the key can be validated
        assert api_key.is_valid() is True
        assert APIKey.hash_key(raw_key) == api_key.key_hash

        # Update last used
        api_key.update_last_used()
        assert api_key.last_used_at is not None

    def test_user_password_flow(self):
        """Test the complete flow of creating a user with password."""
        password = "secure_password_123!"

        # Create user with hashed password
        user = User(
            username="newuser",
            email="new@example.com",
            hashed_password=User.hash_password(password),
        )

        # Verify password works
        assert user.verify_password(password) is True
        assert user.verify_password("wrong_password") is False

        # Verify user is active by default
        assert user.is_active is True
        assert user.is_admin is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
