"""
Security module for MLOps Agent API.

Provides authentication, authorization, rate limiting, and input validation
for securing the FastAPI endpoints.

Components:
- APIKeyAuth: API key-based authentication
- JWTAuth: JWT token authentication and management
- RateLimiter: Request rate limiting per client
- InputValidator: Input sanitization and validation
- SecurityMiddleware: Security headers middleware
- SessionOwnership: Session access control
- get_current_user: FastAPI dependency for user authentication
"""

import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================


class SecurityConfig:
    """Security configuration loaded from environment variables."""

    def __init__(self):
        self.api_key_header_name = os.getenv("API_KEY_HEADER", "X-API-Key")
        self.jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        self.enable_api_key_auth = os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true"
        self.enable_jwt_auth = os.getenv("ENABLE_JWT_AUTH", "false").lower() == "true"
        self.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"

        # Load valid API keys from environment (comma-separated)
        api_keys_str = os.getenv("VALID_API_KEYS", "")
        self.valid_api_keys: set[str] = set(
            key.strip() for key in api_keys_str.split(",") if key.strip()
        )


# Global config instance
security_config = SecurityConfig()


# ============================================================================
# API Key Authentication
# ============================================================================


class APIKeyAuth:
    """
    API key-based authentication.

    Validates API keys passed in request headers.
    Keys are stored as hashed values for security.
    """

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or security_config
        self.header_scheme = APIKeyHeader(
            name=self.config.api_key_header_name,
            auto_error=False,
        )
        # Store hashed versions of valid keys
        self._hashed_keys: set[str] = {self._hash_key(k) for k in self.config.valid_api_keys}

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage/comparison."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_api_key() -> str:
        """Generate a new random API key."""
        return secrets.token_urlsafe(32)

    def validate_key(self, api_key: str) -> bool:
        """Validate an API key against stored keys."""
        if not api_key:
            return False
        hashed = self._hash_key(api_key)
        return hashed in self._hashed_keys

    def add_key(self, api_key: str) -> None:
        """Add a new valid API key."""
        self._hashed_keys.add(self._hash_key(api_key))

    def remove_key(self, api_key: str) -> None:
        """Remove an API key."""
        self._hashed_keys.discard(self._hash_key(api_key))

    async def __call__(self, request: Request) -> str | None:
        """
        FastAPI dependency for API key authentication.

        Returns the API key if valid, raises HTTPException otherwise.
        """
        if not self.config.enable_api_key_auth:
            return None

        api_key = request.headers.get(self.config.api_key_header_name)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if not self.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        return api_key


# ============================================================================
# JWT Authentication
# ============================================================================


class JWTPayload(BaseModel):
    """JWT token payload structure."""

    sub: str = Field(..., description="Subject (user identifier)")
    exp: float = Field(..., description="Expiration timestamp")
    iat: float = Field(..., description="Issued at timestamp")
    roles: list[str] = Field(default_factory=list, description="User roles")
    session_ids: list[str] = Field(default_factory=list, description="Accessible session IDs")


class JWTAuth:
    """
    JWT token authentication.

    Handles token generation, validation, and parsing.
    Uses HMAC-SHA256 for signing.
    """

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or security_config
        self.bearer_scheme = HTTPBearer(auto_error=False)
        self._revoked_tokens: set[str] = set()

    def _base64url_encode(self, data: bytes) -> str:
        """Base64url encode without padding."""
        import base64

        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _base64url_decode(self, data: str) -> bytes:
        """Base64url decode with padding restoration."""
        import base64

        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def create_token(
        self,
        user_id: str,
        roles: list[str] | None = None,
        session_ids: list[str] | None = None,
        expires_in_hours: int | None = None,
    ) -> str:
        """
        Create a new JWT token.

        Args:
            user_id: User identifier (subject)
            roles: List of user roles
            session_ids: List of session IDs the user can access
            expires_in_hours: Token expiration time in hours

        Returns:
            Signed JWT token string
        """
        import json

        now = time.time()
        exp_hours = (
            expires_in_hours if expires_in_hours is not None else self.config.jwt_expiration_hours
        )

        payload = {
            "sub": user_id,
            "exp": now + (exp_hours * 3600),
            "iat": now,
            "roles": roles or [],
            "session_ids": session_ids or [],
        }

        # Create header
        header = {"alg": self.config.jwt_algorithm, "typ": "JWT"}

        # Encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header).encode())
        payload_b64 = self._base64url_encode(json.dumps(payload).encode())

        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.config.jwt_secret.encode(), message.encode(), hashlib.sha256
        ).digest()
        signature_b64 = self._base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> JWTPayload | None:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            JWTPayload if valid, None otherwise
        """
        import json

        if not token or token in self._revoked_tokens:
            return None

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(
                self.config.jwt_secret.encode(), message.encode(), hashlib.sha256
            ).digest()
            actual_sig = self._base64url_decode(signature_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            # Decode payload
            payload_json = self._base64url_decode(payload_b64).decode()
            payload_dict = json.loads(payload_json)

            # Check expiration
            if payload_dict.get("exp", 0) < time.time():
                return None

            return JWTPayload(**payload_dict)

        except Exception:
            return None

    def revoke_token(self, token: str) -> None:
        """Revoke a token to prevent further use."""
        self._revoked_tokens.add(token)

    async def __call__(self, request: Request) -> JWTPayload | None:
        """
        FastAPI dependency for JWT authentication.

        Returns JWTPayload if valid, raises HTTPException otherwise.
        """
        if not self.config.enable_jwt_auth:
            return None

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]  # Remove "Bearer " prefix
        payload = self.verify_token(token)

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload


# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


class RateLimiter:
    """
    Token bucket rate limiter.

    Tracks requests per client IP and enforces rate limits.
    """

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or security_config
        # Track request timestamps per client: {client_id: [timestamps]}
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup every 60 seconds

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise use client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self) -> None:
        """Remove expired request timestamps."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        cutoff = now - self.config.rate_limit_window_seconds
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [ts for ts in self._requests[client_id] if ts > cutoff]
            if not self._requests[client_id]:
                del self._requests[client_id]

        self._last_cleanup = now

    def check_rate_limit(self, request: Request) -> bool:
        """
        Check if request is within rate limit.

        Returns True if allowed, raises RateLimitExceeded otherwise.
        """
        if not self.config.enable_rate_limiting:
            return True

        self._cleanup_old_requests()

        client_id = self._get_client_id(request)
        now = time.time()
        cutoff = now - self.config.rate_limit_window_seconds

        # Filter to recent requests
        recent = [ts for ts in self._requests[client_id] if ts > cutoff]
        self._requests[client_id] = recent

        if len(recent) >= self.config.rate_limit_requests:
            # Calculate retry-after time
            oldest = min(recent)
            retry_after = int(oldest + self.config.rate_limit_window_seconds - now) + 1
            raise RateLimitExceeded(retry_after=max(1, retry_after))

        # Record this request
        self._requests[client_id].append(now)
        return True

    def get_remaining(self, request: Request) -> int:
        """Get remaining requests for client."""
        client_id = self._get_client_id(request)
        cutoff = time.time() - self.config.rate_limit_window_seconds
        recent = [ts for ts in self._requests[client_id] if ts > cutoff]
        return max(0, self.config.rate_limit_requests - len(recent))

    async def __call__(self, request: Request) -> bool:
        """FastAPI dependency for rate limiting."""
        return self.check_rate_limit(request)


# ============================================================================
# Input Validation
# ============================================================================


class PathTraversalError(HTTPException):
    """Exception raised when path traversal is detected."""

    def __init__(self, path: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid path: potential path traversal detected",
        )


class InputValidator:
    """
    Input validation and sanitization utilities.

    Provides methods for validating and sanitizing user inputs
    to prevent injection attacks and path traversal.
    """

    # Patterns for dangerous content
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.",  # Parent directory traversal
    ]

    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--|#|\/\*)",  # SQL comments
        r"(\bOR\b.*=)",  # OR-based injection
    ]

    XSS_PATTERNS = [
        r"<script",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe",
        r"<object",
    ]

    def __init__(self, max_string_length: int = 10000):
        self.max_string_length = max_string_length

    def validate_path(self, path: str, base_path: str | None = None) -> str:
        """
        Validate and normalize a file system path.

        Args:
            path: Path to validate
            base_path: Optional base path to restrict access to

        Returns:
            Normalized, validated path

        Raises:
            PathTraversalError: If path traversal is detected
        """
        if not path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Path cannot be empty"
            )

        # Check for path traversal patterns
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path):
                raise PathTraversalError(path)

        # Normalize the path
        import os

        normalized = os.path.normpath(path)

        # If base_path is provided, ensure path is within it
        if base_path:
            base_normalized = os.path.normpath(os.path.abspath(base_path))
            path_normalized = os.path.normpath(os.path.abspath(normalized))

            if not path_normalized.startswith(base_normalized):
                raise PathTraversalError(path)

        return normalized

    def sanitize_string(self, value: str, allow_html: bool = False) -> str:
        """
        Sanitize a string input.

        Args:
            value: String to sanitize
            allow_html: Whether to allow HTML content

        Returns:
            Sanitized string
        """
        if not value:
            return value

        # Truncate if too long
        if len(value) > self.max_string_length:
            value = value[: self.max_string_length]

        # Remove null bytes
        value = value.replace("\x00", "")

        # Check for XSS if HTML not allowed
        if not allow_html:
            for pattern in self.XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    # Remove the dangerous content
                    value = re.sub(pattern, "", value, flags=re.IGNORECASE)

        return value

    def validate_query(self, query: str) -> str:
        """
        Validate and sanitize a query string.

        Args:
            query: Query string to validate

        Returns:
            Validated query string
        """
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty"
            )

        return self.sanitize_string(query.strip())

    def validate_session_id(self, session_id: str) -> str:
        """
        Validate a session ID format.

        Args:
            session_id: Session ID to validate

        Returns:
            Validated session ID

        Raises:
            HTTPException: If session ID format is invalid
        """
        # UUID v4 format
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

        if not session_id or not re.match(uuid_pattern, session_id.lower()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session ID format"
            )

        return session_id.lower()


# ============================================================================
# Session Ownership
# ============================================================================


class SessionOwnership:
    """
    Session access control.

    Tracks which users own which sessions and enforces access control.
    """

    def __init__(self):
        # Map session_id -> user_id
        self._session_owners: dict[str, str] = {}
        # Map user_id -> set of session_ids
        self._user_sessions: dict[str, set[str]] = defaultdict(set)

    def register_session(self, session_id: str, user_id: str) -> None:
        """Register a session as owned by a user."""
        self._session_owners[session_id] = user_id
        self._user_sessions[user_id].add(session_id)

    def unregister_session(self, session_id: str) -> None:
        """Remove session ownership tracking."""
        if session_id in self._session_owners:
            user_id = self._session_owners.pop(session_id)
            self._user_sessions[user_id].discard(session_id)

    def get_owner(self, session_id: str) -> str | None:
        """Get the owner of a session."""
        return self._session_owners.get(session_id)

    def get_user_sessions(self, user_id: str) -> set[str]:
        """Get all sessions owned by a user."""
        return self._user_sessions.get(user_id, set()).copy()

    def check_access(
        self,
        session_id: str,
        user_id: str,
        allow_admin: bool = True,
        admin_roles: list[str] | None = None,
    ) -> bool:
        """
        Check if a user can access a session.

        Args:
            session_id: Session to access
            user_id: User requesting access
            allow_admin: Whether admin users bypass ownership
            admin_roles: List of roles that grant admin access

        Returns:
            True if access allowed, False otherwise
        """
        # Session not tracked means public access
        if session_id not in self._session_owners:
            return True

        # Owner always has access
        if self._session_owners[session_id] == user_id:
            return True

        return False

    def verify_access(
        self,
        session_id: str,
        user_id: str,
        allow_admin: bool = True,
        admin_roles: list[str] | None = None,
    ) -> None:
        """
        Verify access and raise exception if denied.

        Raises:
            HTTPException: If access is denied
        """
        if not self.check_access(session_id, user_id, allow_admin, admin_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: you do not own this session",
            )


# ============================================================================
# Security Headers Middleware
# ============================================================================


class SecurityHeaders:
    """
    Security headers configuration.

    Provides recommended security headers for HTTP responses.
    """

    DEFAULT_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }

    def __init__(self, custom_headers: dict[str, str] | None = None):
        self.headers = {**self.DEFAULT_HEADERS}
        if custom_headers:
            self.headers.update(custom_headers)

    def get_headers(self) -> dict[str, str]:
        """Get all security headers."""
        return self.headers.copy()


async def security_headers_middleware(request: Request, call_next: Callable) -> Any:
    """
    Middleware to add security headers to all responses.

    Usage:
        app.middleware("http")(security_headers_middleware)
    """
    response = await call_next(request)

    # Add security headers
    headers = SecurityHeaders()
    for name, value in headers.get_headers().items():
        response.headers[name] = value

    return response


# ============================================================================
# Decorator Utilities
# ============================================================================


def require_auth(
    api_key_auth: APIKeyAuth | None = None, jwt_auth: JWTAuth | None = None
) -> Callable:
    """
    Decorator to require authentication on an endpoint.

    Usage:
        @app.get("/protected")
        @require_auth(api_key_auth=api_key_auth)
        async def protected_endpoint(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is not None:
                if api_key_auth:
                    await api_key_auth(request)
                if jwt_auth:
                    await jwt_auth(request)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_rate_limit(rate_limiter: RateLimiter | None = None) -> Callable:
    """
    Decorator to enforce rate limiting on an endpoint.

    Usage:
        @app.get("/api")
        @require_rate_limit(rate_limiter=rate_limiter)
        async def api_endpoint(request: Request):
            ...
    """
    limiter = rate_limiter or RateLimiter()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is not None:
                await limiter(request)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Global Instances
# ============================================================================

# Create global instances for easy import
api_key_auth = APIKeyAuth()
jwt_auth = JWTAuth()
rate_limiter = RateLimiter()
input_validator = InputValidator()
session_ownership = SessionOwnership()
security_headers = SecurityHeaders()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Configuration
    "SecurityConfig",
    "security_config",
    # Authentication
    "APIKeyAuth",
    "api_key_auth",
    "JWTAuth",
    "JWTPayload",
    "jwt_auth",
    # Rate Limiting
    "RateLimiter",
    "RateLimitExceeded",
    "rate_limiter",
    # Input Validation
    "InputValidator",
    "PathTraversalError",
    "input_validator",
    # Session Ownership
    "SessionOwnership",
    "session_ownership",
    # Security Headers
    "SecurityHeaders",
    "security_headers",
    "security_headers_middleware",
    # Decorators
    "require_auth",
    "require_rate_limit",
]


# Import middleware components for convenience
from security.middleware import (
    CurrentUser,
    CurrentUserDep,
    OptionalUserDep,
    AuthenticationError,
    AuthorizationError,
    get_current_user,
    get_current_user_optional,
    require_roles,
    require_scopes,
)

__all__ += [
    # Middleware
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
