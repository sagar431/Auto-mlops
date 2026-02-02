"""
FastAPI Middleware and Dependencies for MLOps Agent API.

Provides the `get_current_user` dependency for authenticating requests
via API keys or JWT tokens, along with related user context utilities.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from security import (
    APIKeyAuth,
    JWTAuth,
    api_key_auth,
    jwt_auth,
)
from security.api_keys import api_key_manager


class CurrentUser(BaseModel):
    """
    Represents the authenticated user context.

    This model is returned by the `get_current_user` dependency and contains
    information about the authenticated user from either API key or JWT auth.
    """

    user_id: str = Field(..., description="User identifier")
    is_authenticated: bool = Field(default=True, description="Whether the user is authenticated")
    auth_method: str = Field(
        ..., description="Authentication method used: 'api_key', 'jwt', or 'anonymous'"
    )
    roles: list[str] = Field(default_factory=list, description="User roles (from JWT)")
    scopes: list[str] = Field(default_factory=list, description="API key scopes")
    session_ids: list[str] = Field(
        default_factory=list, description="Accessible session IDs (from JWT)"
    )
    api_key_id: str | None = Field(
        default=None, description="API key ID if authenticated via API key"
    )
    api_key_name: str | None = Field(
        default=None, description="API key name if authenticated via API key"
    )


class AuthenticationError(HTTPException):
    """Exception raised when authentication fails."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )


class AuthorizationError(HTTPException):
    """Exception raised when authorization fails."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


async def get_current_user(
    request: Request,
    api_key_authenticator: Annotated[APIKeyAuth, Depends(lambda: api_key_auth)],
    jwt_authenticator: Annotated[JWTAuth, Depends(lambda: jwt_auth)],
) -> CurrentUser:
    """
    FastAPI dependency to get the current authenticated user.

    This dependency attempts authentication in the following order:
    1. API Key authentication (X-API-Key header)
    2. JWT Bearer token authentication (Authorization header)
    3. Anonymous access (if auth is disabled)

    Args:
        request: The incoming FastAPI request
        api_key_authenticator: API key auth instance
        jwt_authenticator: JWT auth instance

    Returns:
        CurrentUser: The authenticated user context

    Raises:
        AuthenticationError: If authentication is required but fails

    Usage:
        ```python
        from fastapi import Depends
        from security.middleware import get_current_user, CurrentUser

        @app.get("/protected")
        async def protected_endpoint(user: CurrentUser = Depends(get_current_user)):
            return {"user_id": user.user_id, "roles": user.roles}
        ```
    """
    # Get config from authenticators (they each have their own config)
    api_key_config = api_key_authenticator.config
    jwt_config = jwt_authenticator.config

    # Check if any authentication is enabled
    auth_enabled = api_key_config.enable_api_key_auth or jwt_config.enable_jwt_auth

    # Try API Key authentication first
    api_key = request.headers.get(api_key_config.api_key_header_name)
    if api_key:
        # First try the APIKeyManager (newer system with full key info)
        key_info = api_key_manager.verify(api_key)
        if key_info:
            return CurrentUser(
                user_id=key_info.user_id or f"apikey:{key_info.key_id}",
                is_authenticated=True,
                auth_method="api_key",
                scopes=key_info.scopes,
                api_key_id=key_info.key_id,
                api_key_name=key_info.name,
            )

        # Fall back to APIKeyAuth (simpler system with hashed keys)
        if api_key_authenticator.validate_key(api_key):
            return CurrentUser(
                user_id=f"apikey:{api_key[:8]}",
                is_authenticated=True,
                auth_method="api_key",
            )

        # Invalid API key
        if api_key_config.enable_api_key_auth:
            raise AuthenticationError("Invalid API key")

    # Try JWT authentication
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = jwt_authenticator.verify_token(token)

        if payload:
            return CurrentUser(
                user_id=payload.sub,
                is_authenticated=True,
                auth_method="jwt",
                roles=payload.roles,
                session_ids=payload.session_ids,
            )

        # Invalid JWT token
        if jwt_config.enable_jwt_auth:
            raise AuthenticationError("Invalid or expired token")

    # No credentials provided
    if auth_enabled:
        raise AuthenticationError("Authentication required")

    # Auth is disabled, return anonymous user
    return CurrentUser(
        user_id="anonymous",
        is_authenticated=False,
        auth_method="anonymous",
    )


async def get_current_user_optional(
    request: Request,
    api_key_authenticator: Annotated[APIKeyAuth, Depends(lambda: api_key_auth)],
    jwt_authenticator: Annotated[JWTAuth, Depends(lambda: jwt_auth)],
) -> CurrentUser | None:
    """
    FastAPI dependency to optionally get the current user.

    Similar to `get_current_user`, but returns None instead of raising
    an exception if no valid credentials are provided.

    Args:
        request: The incoming FastAPI request
        api_key_authenticator: API key auth instance
        jwt_authenticator: JWT auth instance

    Returns:
        CurrentUser if authenticated, None otherwise

    Usage:
        ```python
        @app.get("/public")
        async def public_endpoint(user: CurrentUser | None = Depends(get_current_user_optional)):
            if user and user.is_authenticated:
                return {"message": f"Hello, {user.user_id}"}
            return {"message": "Hello, anonymous user"}
        ```
    """
    try:
        return await get_current_user(request, api_key_authenticator, jwt_authenticator)
    except HTTPException:
        return None


def require_roles(*required_roles: str):
    """
    Factory function to create a dependency that requires specific roles.

    Args:
        *required_roles: Role names that are required (any one must match)

    Returns:
        A FastAPI dependency function

    Usage:
        ```python
        @app.get("/admin")
        async def admin_endpoint(user: CurrentUser = Depends(require_roles("admin"))):
            return {"message": "Admin access granted"}

        @app.get("/staff")
        async def staff_endpoint(user: CurrentUser = Depends(require_roles("admin", "staff"))):
            return {"message": "Staff access granted"}
        ```
    """

    async def role_checker(
        user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if not user.is_authenticated:
            raise AuthenticationError("Authentication required")

        if not required_roles:
            return user

        # Check if user has any of the required roles
        if not any(role in user.roles for role in required_roles):
            raise AuthorizationError(f"Required roles: {', '.join(required_roles)}")

        return user

    return role_checker


def require_scopes(*required_scopes: str):
    """
    Factory function to create a dependency that requires specific API key scopes.

    Args:
        *required_scopes: Scope names that are required (all must be present)

    Returns:
        A FastAPI dependency function

    Usage:
        ```python
        @app.post("/deploy")
        async def deploy_endpoint(user: CurrentUser = Depends(require_scopes("deploy:write"))):
            return {"message": "Deploy access granted"}
        ```
    """

    async def scope_checker(
        user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if not user.is_authenticated:
            raise AuthenticationError("Authentication required")

        if not required_scopes:
            return user

        # Check if user has all required scopes
        missing_scopes = set(required_scopes) - set(user.scopes)
        if missing_scopes:
            raise AuthorizationError(f"Missing required scopes: {', '.join(missing_scopes)}")

        return user

    return scope_checker


# Type alias for common dependency injection pattern
CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
OptionalUserDep = Annotated[CurrentUser | None, Depends(get_current_user_optional)]


__all__ = [
    "CurrentUser",
    "CurrentUserDep",
    "OptionalUserDep",
    "AuthenticationError",
    "AuthorizationError",
    "get_current_user",
    "get_current_user_optional",
    "require_roles",
    "require_scopes",
]
