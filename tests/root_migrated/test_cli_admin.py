#!/usr/bin/env python3
"""
Tests for CLI admin commands.

Tests verify that the CLI admin commands work correctly by:
1. Testing argument parsing for all admin subcommands
2. Testing the admin command functions with mocked HTTP responses
3. Testing error handling for connection errors and API errors

Run with: pytest tests/root_migrated/test_cli_admin.py -v
"""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables before importing cli
os.environ["MLOPS_API_URL"] = "http://test-api:8000"
os.environ.setdefault("ENABLE_API_KEY_AUTH", "false")
os.environ.setdefault("ENABLE_JWT_AUTH", "false")

from cli import (
    DEFAULT_API_URL,
    admin_create_key,
    admin_create_user,
    admin_list_keys,
    admin_list_users,
    admin_revoke_key,
    setup_admin_parser,
)


class TestAdminParserSetup:
    """Tests for argument parsing of admin commands."""

    @pytest.fixture
    def parser(self):
        """Create a parser with admin subcommand."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_admin_parser(subparsers)
        return parser

    def test_create_user_args(self, parser):
        """Test create-user argument parsing."""
        args = parser.parse_args(
            [
                "admin",
                "create-user",
                "--username",
                "testuser",
                "--email",
                "test@example.com",
                "--password",
                "secret123",
                "--admin",
            ]
        )
        assert args.command == "admin"
        assert args.admin_command == "create-user"
        assert args.username == "testuser"
        assert args.email == "test@example.com"
        assert args.password == "secret123"
        assert args.admin is True

    def test_create_user_short_args(self, parser):
        """Test create-user with short argument forms."""
        args = parser.parse_args(
            [
                "admin",
                "create-user",
                "-u",
                "testuser",
                "-e",
                "test@example.com",
                "-p",
                "secret123",
                "-a",
            ]
        )
        assert args.username == "testuser"
        assert args.email == "test@example.com"
        assert args.password == "secret123"
        assert args.admin is True

    def test_create_key_args(self, parser):
        """Test create-key argument parsing."""
        args = parser.parse_args(
            [
                "admin",
                "create-key",
                "--name",
                "My Key",
                "--user-id",
                "user123",
                "--expires-in-days",
                "30",
                "--scopes",
                "read,write",
            ]
        )
        assert args.admin_command == "create-key"
        assert args.name == "My Key"
        assert args.user_id == "user123"
        assert args.expires_in_days == 30
        assert args.scopes == "read,write"

    def test_create_key_short_args(self, parser):
        """Test create-key with short argument forms."""
        args = parser.parse_args(
            ["admin", "create-key", "-n", "My Key", "-u", "user123", "-e", "30", "-s", "read,write"]
        )
        assert args.name == "My Key"
        assert args.user_id == "user123"
        assert args.expires_in_days == 30
        assert args.scopes == "read,write"

    def test_list_users_args(self, parser):
        """Test list-users argument parsing."""
        args = parser.parse_args(["admin", "list-users"])
        assert args.admin_command == "list-users"

    def test_list_keys_args(self, parser):
        """Test list-keys argument parsing."""
        args = parser.parse_args(
            ["admin", "list-keys", "--user-id", "user123", "--include-revoked"]
        )
        assert args.admin_command == "list-keys"
        assert args.user_id == "user123"
        assert args.include_revoked is True

    def test_list_keys_short_args(self, parser):
        """Test list-keys with short argument forms."""
        args = parser.parse_args(["admin", "list-keys", "-u", "user123", "-r"])
        assert args.user_id == "user123"
        assert args.include_revoked is True

    def test_revoke_key_args(self, parser):
        """Test revoke-key argument parsing."""
        args = parser.parse_args(["admin", "revoke-key", "--key-id", "abc123"])
        assert args.admin_command == "revoke-key"
        assert args.key_id == "abc123"

    def test_revoke_key_short_args(self, parser):
        """Test revoke-key with short argument forms."""
        args = parser.parse_args(["admin", "revoke-key", "-k", "abc123"])
        assert args.key_id == "abc123"

    def test_api_url_default(self, parser):
        """Test that API URL has a default value."""
        args = parser.parse_args(["admin", "list-users"])
        # The default comes from the environment or the constant
        assert args.api_url is not None

    def test_api_key_from_args(self, parser):
        """Test API key argument."""
        args = parser.parse_args(["admin", "--api-key", "my-secret-key", "list-users"])
        assert args.api_key == "my-secret-key"


class TestAdminCreateUser:
    """Tests for admin_create_user function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for create_user."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            username="newuser",
            email="newuser@example.com",
            password="securepassword123",
            admin=False,
        )
        return args

    def test_create_user_success(self, mock_args):
        """Test successful user creation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "1",
            "username": "newuser",
            "email": "newuser@example.com",
            "is_active": True,
            "is_admin": False,
            "created_at": "2024-01-01T00:00:00",
        }

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = admin_create_user(mock_args)

        assert result == 0

    def test_create_user_api_error(self, mock_args):
        """Test user creation with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Username already exists"}

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = admin_create_user(mock_args)

        assert result == 1

    def test_create_user_connection_error(self, mock_args):
        """Test user creation with connection error."""
        import httpx

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            result = admin_create_user(mock_args)

        assert result == 1

    def test_create_user_prompts_for_password(self):
        """Test that password is prompted if not provided."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            username="newuser",
            email="newuser@example.com",
            password=None,  # Not provided
            admin=False,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "1",
            "username": "newuser",
            "email": "newuser@example.com",
            "is_active": True,
            "is_admin": False,
            "created_at": "2024-01-01T00:00:00",
        }

        with (
            patch("cli.getpass.getpass", return_value="prompted_password"),
            patch("cli.httpx.Client") as mock_client,
        ):
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = admin_create_user(args)

        assert result == 0


class TestAdminCreateKey:
    """Tests for admin_create_key function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for create_key."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            name="My API Key",
            user_id=None,
            expires_in_days=None,
            scopes=None,
        )
        return args

    def test_create_key_success(self, mock_args):
        """Test successful API key creation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "raw_key": "secret-api-key-1234567890abcdef",
            "key_id": "key-123",
            "name": "My API Key",
            "user_id": None,
            "created_at": "2024-01-01T00:00:00",
            "expires_at": None,
        }

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = admin_create_key(mock_args)

        assert result == 0

    def test_create_key_with_options(self):
        """Test API key creation with all options."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            name="My API Key",
            user_id="user-123",
            expires_in_days=30,
            scopes="read,write",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "raw_key": "secret-api-key-1234567890abcdef",
            "key_id": "key-123",
            "name": "My API Key",
            "user_id": "user-123",
            "created_at": "2024-01-01T00:00:00",
            "expires_at": "2024-01-31T00:00:00",
        }

        with patch("cli.httpx.Client") as mock_client:
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.post.return_value = mock_response
            result = admin_create_key(args)

            # Verify the request was made with correct data
            call_args = mock_instance.post.call_args
            json_data = call_args.kwargs["json"]
            assert json_data["name"] == "My API Key"
            assert json_data["user_id"] == "user-123"
            assert json_data["expires_in_days"] == 30
            assert json_data["scopes"] == ["read", "write"]

        assert result == 0

    def test_create_key_api_error(self, mock_args):
        """Test API key creation with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"detail": "Admin privileges required"}

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = admin_create_key(mock_args)

        assert result == 1


class TestAdminListUsers:
    """Tests for admin_list_users function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for list_users."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
        )
        return args

    def test_list_users_success(self, mock_args):
        """Test successful user listing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "1",
                "username": "admin",
                "email": "admin@example.com",
                "is_active": True,
                "is_admin": True,
                "created_at": "2024-01-01T00:00:00",
            },
            {
                "id": "2",
                "username": "user",
                "email": "user@example.com",
                "is_active": True,
                "is_admin": False,
                "created_at": "2024-01-02T00:00:00",
            },
        ]

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = admin_list_users(mock_args)

        assert result == 0

    def test_list_users_empty(self, mock_args):
        """Test listing with no users."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = admin_list_users(mock_args)

        assert result == 0

    def test_list_users_api_error(self, mock_args):
        """Test user listing with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"detail": "Admin privileges required"}

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = admin_list_users(mock_args)

        assert result == 1


class TestAdminListKeys:
    """Tests for admin_list_keys function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for list_keys."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            user_id=None,
            include_revoked=False,
        )
        return args

    def test_list_keys_success(self, mock_args):
        """Test successful API key listing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key_id": "key-123-abcdef-1234567890123456",
                "name": "Production Key",
                "key_prefix": "secret12",
                "user_id": "user-1",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00",
                "expires_at": None,
                "last_used_at": "2024-01-15T10:30:00",
            },
        ]

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = admin_list_keys(mock_args)

        assert result == 0

    def test_list_keys_with_filters(self):
        """Test API key listing with filters."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            user_id="user-123",
            include_revoked=True,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("cli.httpx.Client") as mock_client:
            mock_instance = mock_client.return_value.__enter__.return_value
            mock_instance.get.return_value = mock_response
            result = admin_list_keys(args)

            # Verify the request was made with correct params
            call_args = mock_instance.get.call_args
            params = call_args.kwargs["params"]
            assert params["user_id"] == "user-123"
            assert params["include_revoked"] == "true"

        assert result == 0

    def test_list_keys_empty(self, mock_args):
        """Test listing with no keys."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            result = admin_list_keys(mock_args)

        assert result == 0


class TestAdminRevokeKey:
    """Tests for admin_revoke_key function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for revoke_key."""
        args = argparse.Namespace(
            api_url="http://localhost:8000",
            api_key="test-api-key",
            key_id="key-123",
        )
        return args

    def test_revoke_key_success(self, mock_args):
        """Test successful API key revocation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "message": "API key key-123 revoked successfully",
        }

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.delete.return_value = mock_response
            result = admin_revoke_key(mock_args)

        assert result == 0

    def test_revoke_key_not_found(self, mock_args):
        """Test revoking non-existent key."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "API key not found: key-123"}

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.delete.return_value = mock_response
            result = admin_revoke_key(mock_args)

        assert result == 1

    def test_revoke_key_already_revoked(self, mock_args):
        """Test revoking already revoked key."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "API key is already revoked"}

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.delete.return_value = mock_response
            result = admin_revoke_key(mock_args)

        assert result == 1

    def test_revoke_key_connection_error(self, mock_args):
        """Test key revocation with connection error."""
        import httpx

        with patch("cli.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.delete.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            result = admin_revoke_key(mock_args)

        assert result == 1


class TestDefaultApiUrl:
    """Test that DEFAULT_API_URL is set correctly."""

    def test_default_api_url_from_env(self):
        """Test that DEFAULT_API_URL can be set from environment."""
        # The environment was set at module import time
        assert DEFAULT_API_URL == "http://test-api:8000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
