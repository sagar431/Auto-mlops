"""
Security Database Models

SQLModel classes for User and APIKey entities used in authentication and authorization.
"""

import hashlib
import secrets
from datetime import datetime

from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    """
    User model for authentication and authorization.

    Stores user credentials and metadata for the MLOps Agent API.
    """

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=50)
    email: str = Field(index=True, unique=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    is_active: bool = Field(default=True)
    is_admin: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to API keys
    api_keys: list["APIKey"] = Relationship(back_populates="user")

    password_salt: str | None = Field(default=None, max_length=64)

    @staticmethod
    def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
        """
        Hash a password using PBKDF2-SHA256 with a random salt.

        Args:
            password: Plain text password
            salt: Optional salt bytes (generated if not provided)

        Returns:
            Tuple of (hashed_password_hex, salt_hex)
        """
        import os

        if salt is None:
            salt = os.urandom(32)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=100_000)
        return hashed.hex(), salt.hex()

    def verify_password(self, password: str) -> bool:
        """
        Verify a password against the stored hash.

        Args:
            password: Plain text password to verify

        Returns:
            True if password matches, False otherwise
        """
        if self.password_salt:
            salt = bytes.fromhex(self.password_salt)
            hashed, _ = self.hash_password(password, salt)
            return self.hashed_password == hashed
        # Fallback for legacy unsalted hashes
        return self.hashed_password == hashlib.sha256(password.encode()).hexdigest()


class APIKey(SQLModel, table=True):
    """
    API Key model for programmatic access authentication.

    Stores API keys with metadata for tracking usage and managing access.
    """

    __tablename__ = "api_keys"

    id: int | None = Field(default=None, primary_key=True)
    key_hash: str = Field(index=True, unique=True, max_length=64)
    name: str = Field(max_length=100)
    user_id: int = Field(foreign_key="users.id", index=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(default=None)
    last_used_at: datetime | None = Field(default=None)

    # Relationship to user
    user: User | None = Relationship(back_populates="api_keys")

    @staticmethod
    def generate_key() -> str:
        """
        Generate a new random API key.

        Returns:
            A secure random API key string
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_key(key: str) -> str:
        """
        Hash an API key for secure storage.

        Args:
            key: Plain text API key

        Returns:
            Hashed key string
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def is_valid(self) -> bool:
        """
        Check if the API key is valid (active and not expired).

        Returns:
            True if the key is valid, False otherwise
        """
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def update_last_used(self) -> None:
        """Update the last_used_at timestamp to now."""
        self.last_used_at = datetime.utcnow()
