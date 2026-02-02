"""
Circuit Breaker Pattern Implementation.

Provides protection against cascading failures by tracking error rates
and temporarily stopping calls to failing services.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failures exceeded threshold, requests fail fast
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from observability import get_logger

logger = get_logger("resilience.circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    """Number of failures before opening the circuit."""

    success_threshold: int = 2
    """Number of successes in half-open state before closing."""

    timeout_seconds: float = 30.0
    """Time to wait before transitioning from open to half-open."""

    half_open_max_calls: int = 3
    """Maximum concurrent calls allowed in half-open state."""

    excluded_exceptions: tuple[type[Exception], ...] = ()
    """Exceptions that should not count as failures."""


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitState, retry_after: float | None = None):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        message = f"Circuit breaker '{name}' is {state.value}"
        if retry_after:
            message += f", retry after {retry_after:.1f}s"
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker("external_api")

        @breaker
        async def call_external_api():
            ...

        # Or manually:
        async with breaker:
            await call_external_api()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def _should_transition_to_half_open(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._state != CircuitState.OPEN:
            return False
        elapsed = time.time() - self._last_state_change
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        self._stats.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(
            "Circuit state changed",
            circuit=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
            consecutive_failures=self._stats.consecutive_failures,
            consecutive_successes=self._stats.consecutive_successes,
        )

    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if isinstance(error, self.config.excluded_exceptions):
            return

        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0

        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)

    def _record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self._stats.total_calls += 1
        self._stats.rejected_calls += 1

    async def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        async with self._lock:
            # Check for state transition
            if self._should_transition_to_half_open():
                self._transition_to(CircuitState.HALF_OPEN)

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            # OPEN state
            return False

    def get_retry_after(self) -> float | None:
        """Get seconds until circuit might close."""
        if self._state != CircuitState.OPEN:
            return None
        elapsed = time.time() - self._last_state_change
        remaining = self.config.timeout_seconds - elapsed
        return max(0, remaining)

    async def __aenter__(self):
        """Async context manager entry."""
        if not await self._can_execute():
            self._record_rejection()
            raise CircuitBreakerError(
                self.name,
                self._state,
                self.get_retry_after(),
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_val is None:
            self._record_success()
        else:
            self._record_failure(exc_val)
        return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for protecting async functions."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with self:
                return await func(*args, **kwargs)

        return wrapper

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._half_open_calls = 0
        logger.info("Circuit breaker reset", circuit=self.name)


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()
        breaker = registry.get_or_create("api_calls")
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name,
                    config or self._default_config,
                )
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: {
                "state": breaker.state.value,
                "total_calls": breaker.stats.total_calls,
                "successful_calls": breaker.stats.successful_calls,
                "failed_calls": breaker.stats.failed_calls,
                "rejected_calls": breaker.stats.rejected_calls,
                "consecutive_failures": breaker.stats.consecutive_failures,
                "state_changes": breaker.stats.state_changes,
            }
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
