#!/usr/bin/env python3
"""
Tests for the API Key Manager module.

Run with: pytest test_api_keys.py -v
"""

from datetime import datetime, timedelta

import pytest

from security.api_keys import APIKeyManager, GeneratedKey, api_key_manager


class TestAPIKeyGeneration:
    """Tests for API key generation."""

    def test_generate_basic_key(self):
        """Test generating a basic API key."""
        manager = APIKeyManager()
        result = manager.generate(name="Test Key")

        assert isinstance(result, GeneratedKey)
        assert result.raw_key is not None
        assert len(result.raw_key) > 20
        assert result.key_info.name == "Test Key"
        assert result.key_info.is_active is True
        assert result.key_info.user_id is None

    def test_generate_key_with_user_id(self):
        """Test generating a key with user association."""
        manager = APIKeyManager()
        result = manager.generate(name="User Key", user_id="user123")

        assert result.key_info.user_id == "user123"

    def test_generate_key_with_expiration(self):
        """Test generating a key with expiration."""
        manager = APIKeyManager()
        result = manager.generate(name="Expiring Key", expires_in_days=30)

        assert result.key_info.expires_at is not None
        expected_expiry = datetime.utcnow() + timedelta(days=30)
        # Allow 1 minute tolerance
        assert abs((result.key_info.expires_at - expected_expiry).total_seconds()) < 60

    def test_generate_key_with_scopes(self):
        """Test generating a key with scopes."""
        manager = APIKeyManager()
        scopes = ["read", "write", "admin"]
        result = manager.generate(name="Scoped Key", scopes=scopes)

        assert result.key_info.scopes == scopes

    def test_generate_unique_keys(self):
        """Test that generated keys are unique."""
        manager = APIKeyManager()
        keys = [manager.generate(name=f"Key {i}") for i in range(10)]
        raw_keys = [k.raw_key for k in keys]
        key_ids = [k.key_info.key_id for k in keys]

        assert len(set(raw_keys)) == 10
        assert len(set(key_ids)) == 10

    def test_key_prefix_matches(self):
        """Test that key prefix matches the raw key."""
        manager = APIKeyManager()
        result = manager.generate(name="Test Key")

        assert result.key_info.key_prefix == result.raw_key[:8]


class TestAPIKeyVerification:
    """Tests for API key verification."""

    def test_verify_valid_key(self):
        """Test verifying a valid key."""
        manager = APIKeyManager()
        result = manager.generate(name="Valid Key")

        key_info = manager.verify(result.raw_key)
        assert key_info is not None
        assert key_info.name == "Valid Key"

    def test_verify_invalid_key(self):
        """Test verifying an invalid key."""
        manager = APIKeyManager()
        key_info = manager.verify("invalid_key_12345")

        assert key_info is None

    def test_verify_empty_key(self):
        """Test verifying an empty key."""
        manager = APIKeyManager()
        assert manager.verify("") is None
        assert manager.verify(None) is None

    def test_verify_revoked_key(self):
        """Test that revoked keys fail verification."""
        manager = APIKeyManager()
        result = manager.generate(name="To Revoke")

        manager.revoke(result.key_info.key_id)
        key_info = manager.verify(result.raw_key)

        assert key_info is None

    def test_verify_expired_key(self):
        """Test that expired keys fail verification."""
        manager = APIKeyManager()
        result = manager.generate(name="Expired Key", expires_in_days=-1)  # Already expired

        key_info = manager.verify(result.raw_key)
        assert key_info is None

    def test_verify_updates_last_used(self):
        """Test that verification updates last_used_at."""
        manager = APIKeyManager()
        result = manager.generate(name="Test Key")

        assert result.key_info.last_used_at is None

        manager.verify(result.raw_key)
        key_info = manager.get_key_info(result.key_info.key_id)

        assert key_info.last_used_at is not None

    def test_verify_with_required_scope_success(self):
        """Test verification with required scope present."""
        manager = APIKeyManager()
        result = manager.generate(name="Scoped Key", scopes=["read", "write"])

        key_info = manager.verify(result.raw_key, required_scope="read")
        assert key_info is not None

    def test_verify_with_required_scope_failure(self):
        """Test verification fails when required scope is missing."""
        manager = APIKeyManager()
        result = manager.generate(name="Scoped Key", scopes=["read"])

        key_info = manager.verify(result.raw_key, required_scope="admin")
        assert key_info is None


class TestAPIKeyRevocation:
    """Tests for API key revocation."""

    def test_revoke_by_id(self):
        """Test revoking a key by ID."""
        manager = APIKeyManager()
        result = manager.generate(name="To Revoke")

        success = manager.revoke(result.key_info.key_id)
        assert success is True

        key_info = manager.get_key_info(result.key_info.key_id)
        assert key_info.is_active is False

    def test_revoke_nonexistent_key(self):
        """Test revoking a nonexistent key."""
        manager = APIKeyManager()
        success = manager.revoke("nonexistent_id")

        assert success is False

    def test_revoke_by_raw_key(self):
        """Test revoking a key using the raw key."""
        manager = APIKeyManager()
        result = manager.generate(name="To Revoke")

        success = manager.revoke_by_raw_key(result.raw_key)
        assert success is True

        # Verification should now fail
        assert manager.verify(result.raw_key) is None

    def test_revoke_by_raw_key_invalid(self):
        """Test revoking an invalid raw key."""
        manager = APIKeyManager()
        success = manager.revoke_by_raw_key("invalid_key")

        assert success is False

    def test_revoke_by_raw_key_empty(self):
        """Test revoking an empty raw key."""
        manager = APIKeyManager()
        assert manager.revoke_by_raw_key("") is False
        assert manager.revoke_by_raw_key(None) is False

    def test_revoke_all_for_user(self):
        """Test revoking all keys for a user."""
        manager = APIKeyManager()

        # Create keys for different users
        manager.generate(name="User1 Key1", user_id="user1")
        manager.generate(name="User1 Key2", user_id="user1")
        manager.generate(name="User2 Key", user_id="user2")

        count = manager.revoke_all_for_user("user1")
        assert count == 2

        # User1 keys should be inactive
        user1_keys = manager.list_keys(user_id="user1", include_revoked=True)
        assert all(not k.is_active for k in user1_keys)

        # User2 key should still be active
        user2_keys = manager.list_keys(user_id="user2")
        assert len(user2_keys) == 1
        assert user2_keys[0].is_active


class TestAPIKeyListing:
    """Tests for listing API keys."""

    def test_list_all_keys(self):
        """Test listing all keys."""
        manager = APIKeyManager()
        manager.generate(name="Key 1")
        manager.generate(name="Key 2")
        manager.generate(name="Key 3")

        keys = manager.list_keys()
        assert len(keys) == 3

    def test_list_keys_by_user(self):
        """Test listing keys filtered by user."""
        manager = APIKeyManager()
        manager.generate(name="User1 Key", user_id="user1")
        manager.generate(name="User2 Key", user_id="user2")
        manager.generate(name="No User Key")

        user1_keys = manager.list_keys(user_id="user1")
        assert len(user1_keys) == 1
        assert user1_keys[0].name == "User1 Key"

    def test_list_keys_excludes_revoked(self):
        """Test that revoked keys are excluded by default."""
        manager = APIKeyManager()
        manager.generate(name="Active Key")
        result2 = manager.generate(name="Revoked Key")
        manager.revoke(result2.key_info.key_id)

        keys = manager.list_keys()
        assert len(keys) == 1
        assert keys[0].name == "Active Key"

    def test_list_keys_includes_revoked(self):
        """Test listing with revoked keys included."""
        manager = APIKeyManager()
        manager.generate(name="Active Key")
        result2 = manager.generate(name="Revoked Key")
        manager.revoke(result2.key_info.key_id)

        keys = manager.list_keys(include_revoked=True)
        assert len(keys) == 2


class TestAPIKeyInfo:
    """Tests for getting key info."""

    def test_get_key_info(self):
        """Test getting key info by ID."""
        manager = APIKeyManager()
        result = manager.generate(name="Test Key")

        key_info = manager.get_key_info(result.key_info.key_id)
        assert key_info is not None
        assert key_info.name == "Test Key"
        assert key_info.key_id == result.key_info.key_id

    def test_get_key_info_nonexistent(self):
        """Test getting info for nonexistent key."""
        manager = APIKeyManager()
        key_info = manager.get_key_info("nonexistent_id")

        assert key_info is None


class TestAPIKeyCleanup:
    """Tests for key cleanup."""

    def test_cleanup_expired_keys(self):
        """Test cleaning up expired keys."""
        manager = APIKeyManager()
        manager.generate(name="Valid Key", expires_in_days=30)
        manager.generate(name="Expired Key 1", expires_in_days=-1)
        manager.generate(name="Expired Key 2", expires_in_days=-7)

        count = manager.cleanup_expired()
        assert count == 2

        keys = manager.list_keys()
        assert len(keys) == 1
        assert keys[0].name == "Valid Key"


class TestAPIKeyStatePersistence:
    """Tests for state export/import."""

    def test_export_state(self):
        """Test exporting manager state."""
        manager = APIKeyManager()
        result = manager.generate(name="Test Key", user_id="user1")
        manager.revoke(result.key_info.key_id)

        state = manager.export_state()

        assert "keys" in state
        assert "revoked_hashes" in state
        assert len(state["keys"]) == 1
        assert len(state["revoked_hashes"]) == 1

    def test_import_state(self):
        """Test importing previously exported state."""
        manager1 = APIKeyManager()
        result = manager1.generate(name="Test Key", user_id="user1", scopes=["read"])

        state = manager1.export_state()

        manager2 = APIKeyManager()
        manager2.import_state(state)

        # Verify the key was imported
        key_info = manager2.verify(result.raw_key)
        assert key_info is not None
        assert key_info.name == "Test Key"
        assert key_info.user_id == "user1"
        assert key_info.scopes == ["read"]

    def test_import_state_with_revoked(self):
        """Test importing state preserves revoked keys."""
        manager1 = APIKeyManager()
        result = manager1.generate(name="Revoked Key")
        manager1.revoke(result.key_info.key_id)

        state = manager1.export_state()

        manager2 = APIKeyManager()
        manager2.import_state(state)

        # Verify the key remains revoked
        assert manager2.verify(result.raw_key) is None


class TestGlobalInstance:
    """Tests for the global api_key_manager instance."""

    def test_global_instance_exists(self):
        """Test that global instance is available."""
        assert api_key_manager is not None
        assert isinstance(api_key_manager, APIKeyManager)

    def test_global_instance_works(self):
        """Test that global instance functions correctly."""
        result = api_key_manager.generate(name="Global Test")
        key_info = api_key_manager.verify(result.raw_key)

        assert key_info is not None
        assert key_info.name == "Global Test"

        # Clean up
        api_key_manager.revoke(result.key_info.key_id)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_double_revoke(self):
        """Test revoking the same key twice."""
        manager = APIKeyManager()
        result = manager.generate(name="Test Key")

        assert manager.revoke(result.key_info.key_id) is True
        assert manager.revoke(result.key_info.key_id) is False  # Already revoked

    def test_verify_after_cleanup(self):
        """Test that cleaned up keys can't be verified."""
        manager = APIKeyManager()
        result = manager.generate(name="Expired Key", expires_in_days=-1)
        raw_key = result.raw_key

        manager.cleanup_expired()

        assert manager.verify(raw_key) is None

    def test_special_characters_in_name(self):
        """Test keys with special characters in name."""
        manager = APIKeyManager()
        result = manager.generate(name="Key with !@#$%^&*() special chars")

        assert result.key_info.name == "Key with !@#$%^&*() special chars"
        assert manager.verify(result.raw_key) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
