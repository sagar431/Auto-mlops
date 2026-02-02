"""
Timeout Management for Async Operations.

Provides configurable timeouts for async operations with proper
cancellation handling and logging.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from observability import get_logger

logger = get_logger("resilience.timeout")

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""

    def __init__(
        self,
        operation_name: str,
        timeout_seconds: float,
        message: str | None = None,
    ):
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        msg = message or f"Operation '{operation_name}' timed out after {timeout_seconds}s"
        super().__init__(msg)


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""

    timeout_seconds: float = 30.0
    """Maximum time allowed for the operation."""

    cancel_on_timeout: bool = True
    """Whether to cancel the operation on timeout."""

    log_timeout: bool = True
    """Whether to log timeout events."""


@dataclass
class TimeoutStats:
    """Statistics for timeout operations."""

    total_calls: int = 0
    successful_calls: int = 0
    timed_out_calls: int = 0
    total_duration_seconds: float = 0.0


class TimeoutManager:
    """
    Manager for tracking and enforcing timeouts.

    Usage:
        manager = TimeoutManager("api_calls", TimeoutConfig(timeout_seconds=10))

        @manager
        async def call_api():
            ...

        # Or manually:
        async with manager.timeout() as ctx:
            await call_api()
    """

    def __init__(
        self,
        name: str,
        config: TimeoutConfig | None = None,
    ):
        self.name = name
        self.config = config or TimeoutConfig()
        self._stats = TimeoutStats()

    @property
    def stats(self) -> TimeoutStats:
        """Get timeout statistics."""
        return self._stats

    def timeout(self, timeout_seconds: float | None = None) -> "TimeoutContext":
        """Create a timeout context."""
        return TimeoutContext(
            self,
            timeout_seconds or self.config.timeout_seconds,
        )

    def _record_success(self, duration: float) -> None:
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.total_duration_seconds += duration

    def _record_timeout(self, timeout_seconds: float) -> None:
        """Record a timed out call."""
        self._stats.total_calls += 1
        self._stats.timed_out_calls += 1
        self._stats.total_duration_seconds += timeout_seconds

        if self.config.log_timeout:
            logger.warning(
                "Operation timed out",
                operation=self.name,
                timeout_seconds=timeout_seconds,
                total_timeouts=self._stats.timed_out_calls,
            )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for adding timeout to async functions."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            import time

            start = time.time()
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
                duration = time.time() - start
                self._record_success(duration)
                return result
            except asyncio.TimeoutError:
                self._record_timeout(self.config.timeout_seconds)
                raise TimeoutError(self.name, self.config.timeout_seconds)

        return wrapper


class TimeoutContext:
    """
    Context manager for timeout operations.

    Usage:
        async with TimeoutContext(manager, 30.0) as ctx:
            await some_operation()
    """

    def __init__(self, manager: TimeoutManager, timeout_seconds: float):
        self.manager = manager
        self.timeout_seconds = timeout_seconds
        self._start_time: float | None = None
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        """Enter the timeout context."""
        import time

        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the timeout context."""
        import time

        if self._start_time:
            duration = time.time() - self._start_time
            if exc_type is asyncio.TimeoutError:
                self.manager._record_timeout(self.timeout_seconds)
                raise TimeoutError(
                    self.manager.name,
                    self.timeout_seconds,
                ) from exc_val
            else:
                self.manager._record_success(duration)
        return False


def timeout(
    timeout_seconds: float,
    operation_name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding timeout to async functions.

    Usage:
        @timeout(30.0)
        async def fetch_data():
            ...

        @timeout(10.0, operation_name="api_call")
        async def call_api():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Operation timed out",
                    operation=name,
                    timeout_seconds=timeout_seconds,
                )
                raise TimeoutError(name, timeout_seconds)

        return wrapper

    return decorator


async def with_timeout(
    operation: Callable[[], T],
    timeout_seconds: float,
    operation_name: str = "operation",
) -> T:
    """
    Execute an async operation with a timeout.

    Usage:
        result = await with_timeout(
            lambda: fetch_data(url),
            timeout_seconds=30.0,
            operation_name="fetch_data",
        )
    """
    try:
        return await asyncio.wait_for(
            operation(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Operation timed out",
            operation=operation_name,
            timeout_seconds=timeout_seconds,
        )
        raise TimeoutError(operation_name, timeout_seconds)


class AdaptiveTimeout:
    """
    Adaptive timeout that adjusts based on historical performance.

    Tracks operation durations and adjusts timeout based on percentiles.
    """

    def __init__(
        self,
        name: str,
        initial_timeout: float = 30.0,
        min_timeout: float = 5.0,
        max_timeout: float = 120.0,
        percentile: float = 95.0,
        window_size: int = 100,
    ):
        self.name = name
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.percentile = percentile
        self.window_size = window_size
        self._durations: list[float] = []
        self._current_timeout = initial_timeout

    @property
    def current_timeout(self) -> float:
        """Get the current adaptive timeout value."""
        return self._current_timeout

    def record_duration(self, duration: float) -> None:
        """Record an operation duration."""
        self._durations.append(duration)

        # Keep only the window size
        if len(self._durations) > self.window_size:
            self._durations = self._durations[-self.window_size :]

        # Recalculate timeout
        self._update_timeout()

    def _update_timeout(self) -> None:
        """Update the timeout based on recorded durations."""
        if len(self._durations) < 10:
            # Not enough data, use initial
            return

        # Sort and get percentile
        sorted_durations = sorted(self._durations)
        index = int(len(sorted_durations) * self.percentile / 100)
        index = min(index, len(sorted_durations) - 1)

        # Add buffer (1.5x the percentile value)
        new_timeout = sorted_durations[index] * 1.5

        # Clamp to min/max
        self._current_timeout = max(self.min_timeout, min(self.max_timeout, new_timeout))

        logger.debug(
            "Adaptive timeout updated",
            operation=self.name,
            new_timeout=round(self._current_timeout, 2),
            sample_count=len(self._durations),
        )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for adding adaptive timeout to async functions."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            import time

            start = time.time()
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self._current_timeout,
                )
                duration = time.time() - start
                self.record_duration(duration)
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "Operation timed out (adaptive)",
                    operation=self.name,
                    timeout_seconds=self._current_timeout,
                )
                raise TimeoutError(self.name, self._current_timeout)

        return wrapper
