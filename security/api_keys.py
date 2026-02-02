"""
API Key Manager for MLOps Agent.

Provides a high-level interface for managing API keys with support for
both in-memory storage (testing) and database persistence (production).
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class APIKeyInfo(BaseModel):
    """Information about an API key (without the raw key)."""

    key_id: str = Field(..., description="Unique identifier for the key")
    name: str = Field(..., description="Human-readable name for the key")
    key_prefix: str = Field(..., description="First 8 characters of the key for identification")
    user_id: str | None = Field(default=None, description="Associated user ID")
    is_active: bool = Field(default=True, description="Whether the key is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(default=None, description="Expiration timestamp")
    last_used_at: datetime | None = Field(default=None, description="Last usage timestamp")
    scopes: list[str] = Field(default_factory=list, description="Allowed scopes/permissions")


class GeneratedKey(BaseModel):
    """Result of generating a new API key."""

    raw_key: str = Field(..., description="The raw API key (only shown once)")
    key_info: APIKeyInfo = Field(..., description="Key metadata")


class APIKeyManager:
    """
    Manages API keys for authentication.

    Supports generating, verifying, and revoking API keys with metadata tracking.
    Uses in-memory storage by default; can be extended for database persistence.

    Usage:
        manager = APIKeyManager()
        result = manager.generate(name="My API Key", user_id="user123")
        print(f"Save this key: {result.raw_key}")

        # Later, verify the key
        key_info = manager.verify(raw_key)
        if key_info:
            print(f"Valid key: {key_info.name}")

        # Revoke when no longer needed
        manager.revoke(result.key_info.key_id)
    """

    def __init__(self):
        """Initialize the API key manager with in-memory storage."""
        # Map key_hash -> APIKeyInfo
        self._keys: dict[str, APIKeyInfo] = {}
        # Track revoked key hashes for fast lookup
        self._revoked_hashes: set[str] = set()

    @staticmethod
    def _hash_key(key: str) -> str:
        """
        Hash an API key using SHA-256.

        Args:
            key: The raw API key string

        Returns:
            The hashed key as a hex string
        """
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _generate_key_id() -> str:
        """Generate a unique key ID."""
        return secrets.token_hex(16)

    @staticmethod
    def _generate_raw_key() -> str:
        """
        Generate a secure random API key.

        Returns:
            A URL-safe random string (43 characters)
        """
        return secrets.token_urlsafe(32)

    def generate(
        self,
        name: str,
        user_id: str | None = None,
        expires_in_days: int | None = None,
        scopes: list[str] | None = None,
    ) -> GeneratedKey:
        """
        Generate a new API key.

        Args:
            name: Human-readable name for the key
            user_id: Optional user ID to associate with the key
            expires_in_days: Optional number of days until expiration
            scopes: Optional list of allowed scopes/permissions

        Returns:
            GeneratedKey with raw key and metadata

        Example:
            result = manager.generate(name="Production API", user_id="user123")
            # Save result.raw_key securely - it won't be shown again!
        """
        raw_key = self._generate_raw_key()
        key_hash = self._hash_key(raw_key)
        key_id = self._generate_key_id()

        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        key_info = APIKeyInfo(
            key_id=key_id,
            name=name,
            key_prefix=raw_key[:8],
            user_id=user_id,
            is_active=True,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            scopes=scopes or [],
        )

        self._keys[key_hash] = key_info

        return GeneratedKey(raw_key=raw_key, key_info=key_info)

    def verify(self, raw_key: str, required_scope: str | None = None) -> APIKeyInfo | None:
        """
        Verify an API key and return its info if valid.

        Args:
            raw_key: The raw API key to verify
            required_scope: Optional scope that must be present

        Returns:
            APIKeyInfo if valid, None if invalid/expired/revoked

        Example:
            key_info = manager.verify(api_key)
            if key_info:
                print(f"Authenticated as: {key_info.user_id}")
        """
        if not raw_key:
            return None

        key_hash = self._hash_key(raw_key)

        # Check if revoked
        if key_hash in self._revoked_hashes:
            return None

        # Look up the key
        key_info = self._keys.get(key_hash)
        if key_info is None:
            return None

        # Check if active
        if not key_info.is_active:
            return None

        # Check expiration
        if key_info.expires_at and datetime.utcnow() > key_info.expires_at:
            return None

        # Check scope if required
        if required_scope and required_scope not in key_info.scopes:
            return None

        # Update last used timestamp
        key_info.last_used_at = datetime.utcnow()

        return key_info

    def revoke(self, key_id: str) -> bool:
        """
        Revoke an API key by its ID.

        Args:
            key_id: The key ID to revoke

        Returns:
            True if key was found and revoked, False if not found

        Example:
            if manager.revoke(key_id):
                print("Key revoked successfully")
        """
        # Find the key by ID
        for key_hash, key_info in self._keys.items():
            if key_info.key_id == key_id:
                if not key_info.is_active:
                    return False  # Already revoked
                key_info.is_active = False
                self._revoked_hashes.add(key_hash)
                return True
        return False

    def revoke_by_raw_key(self, raw_key: str) -> bool:
        """
        Revoke an API key using the raw key string.

        Args:
            raw_key: The raw API key to revoke

        Returns:
            True if key was found and revoked, False if not found
        """
        if not raw_key:
            return False

        key_hash = self._hash_key(raw_key)
        key_info = self._keys.get(key_hash)

        if key_info is None:
            return False

        key_info.is_active = False
        self._revoked_hashes.add(key_hash)
        return True

    def get_key_info(self, key_id: str) -> APIKeyInfo | None:
        """
        Get information about a key by its ID.

        Args:
            key_id: The key ID to look up

        Returns:
            APIKeyInfo if found, None otherwise
        """
        for key_info in self._keys.values():
            if key_info.key_id == key_id:
                return key_info
        return None

    def list_keys(
        self, user_id: str | None = None, include_revoked: bool = False
    ) -> list[APIKeyInfo]:
        """
        List all API keys, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by
            include_revoked: Whether to include revoked keys

        Returns:
            List of APIKeyInfo objects
        """
        keys = []
        for key_info in self._keys.values():
            if user_id is not None and key_info.user_id != user_id:
                continue
            if not include_revoked and not key_info.is_active:
                continue
            keys.append(key_info)
        return keys

    def revoke_all_for_user(self, user_id: str) -> int:
        """
        Revoke all API keys for a user.

        Args:
            user_id: The user ID whose keys should be revoked

        Returns:
            Number of keys revoked
        """
        count = 0
        for key_hash, key_info in self._keys.items():
            if key_info.user_id == user_id and key_info.is_active:
                key_info.is_active = False
                self._revoked_hashes.add(key_hash)
                count += 1
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired keys from storage.

        Returns:
            Number of keys cleaned up
        """
        now = datetime.utcnow()
        expired_hashes = []

        for key_hash, key_info in self._keys.items():
            if key_info.expires_at and key_info.expires_at < now:
                expired_hashes.append(key_hash)

        for key_hash in expired_hashes:
            del self._keys[key_hash]
            self._revoked_hashes.discard(key_hash)

        return len(expired_hashes)

    def export_state(self) -> dict[str, Any]:
        """
        Export the manager state for persistence.

        Returns:
            Dictionary containing all key data (hashes, not raw keys)
        """
        return {
            "keys": {key_hash: key_info.model_dump() for key_hash, key_info in self._keys.items()},
            "revoked_hashes": list(self._revoked_hashes),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """
        Import previously exported state.

        Args:
            state: State dictionary from export_state()
        """
        self._keys.clear()
        self._revoked_hashes.clear()

        for key_hash, key_data in state.get("keys", {}).items():
            self._keys[key_hash] = APIKeyInfo(**key_data)

        self._revoked_hashes.update(state.get("revoked_hashes", []))


# Global instance for easy access
api_key_manager = APIKeyManager()


__all__ = [
    "APIKeyManager",
    "APIKeyInfo",
    "GeneratedKey",
    "api_key_manager",
]
