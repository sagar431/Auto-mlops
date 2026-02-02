"""
Retry Pattern with Exponential Backoff.

Provides configurable retry logic with exponential backoff and jitter
for handling transient failures in async operations.
"""

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

from observability import get_logger

logger = get_logger("resilience.retry")

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    """Maximum number of attempts (including the first one)."""

    base_delay_seconds: float = 1.0
    """Initial delay between retries."""

    max_delay_seconds: float = 60.0
    """Maximum delay between retries."""

    exponential_base: float = 2.0
    """Base for exponential backoff calculation."""

    jitter: bool = True
    """Whether to add random jitter to delays."""

    jitter_factor: float = 0.5
    """Maximum jitter as a fraction of the delay (0.5 = up to 50% variation)."""

    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    """Exceptions that should trigger a retry."""

    non_retryable_exceptions: tuple[type[Exception], ...] = ()
    """Exceptions that should NOT trigger a retry (takes precedence)."""


@dataclass
class RetryStats:
    """Statistics for retry operations."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0
    last_error: Exception | None = field(default=None, repr=False)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
        total_delay: float = 0.0,
    ):
        self.attempts = attempts
        self.last_error = last_error
        self.total_delay = total_delay
        super().__init__(message)


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay for the given attempt number.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * exponential_base^(attempt-1)
    delay = config.base_delay_seconds * (config.exponential_base ** (attempt - 1))

    # Cap at max delay
    delay = min(delay, config.max_delay_seconds)

    # Add jitter if enabled
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay = delay + random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # Ensure non-negative

    return delay


def should_retry(
    error: Exception,
    config: RetryConfig,
) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: The exception that occurred
        config: Retry configuration

    Returns:
        True if the error should be retried
    """
    # Non-retryable exceptions take precedence
    if isinstance(error, config.non_retryable_exceptions):
        return False

    # Check if it's a retryable exception
    return isinstance(error, config.retryable_exceptions)


class RetryContext:
    """
    Context manager for retry operations.

    Usage:
        async with RetryContext(config) as ctx:
            while ctx.should_continue():
                try:
                    result = await some_operation()
                    ctx.record_success()
                    break
                except Exception as e:
                    await ctx.record_failure(e)
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        operation_name: str = "operation",
    ):
        self.config = config or RetryConfig()
        self.operation_name = operation_name
        self._attempt = 0
        self._stats = RetryStats()
        self._success = False
        self._last_error: Exception | None = None

    @property
    def attempt(self) -> int:
        """Current attempt number (1-indexed)."""
        return self._attempt

    @property
    def stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def should_continue(self) -> bool:
        """Check if more attempts should be made."""
        if self._success:
            return False
        return self._attempt < self.config.max_attempts

    def record_success(self) -> None:
        """Record a successful attempt."""
        self._success = True
        self._stats.successful_attempts += 1
        if self._attempt > 1:
            logger.info(
                "Operation succeeded after retries",
                operation=self.operation_name,
                attempt=self._attempt,
                total_retries=self._attempt - 1,
            )

    async def record_failure(self, error: Exception) -> None:
        """
        Record a failed attempt and wait before next retry if applicable.

        Args:
            error: The exception that occurred

        Raises:
            RetryExhaustedError: If no more retries are available
            Exception: If the error is not retryable
        """
        self._last_error = error
        self._stats.last_error = error
        self._stats.failed_attempts += 1

        # Check if we should retry this error
        if not should_retry(error, self.config):
            logger.warning(
                "Non-retryable error encountered",
                operation=self.operation_name,
                attempt=self._attempt,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise error

        # Check if we have more attempts
        if self._attempt >= self.config.max_attempts:
            logger.error(
                "Retry attempts exhausted",
                operation=self.operation_name,
                attempts=self._attempt,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise RetryExhaustedError(
                f"Operation '{self.operation_name}' failed after {self._attempt} attempts",
                attempts=self._attempt,
                last_error=error,
                total_delay=self._stats.total_delay_seconds,
            )

        # Calculate and apply delay
        delay = calculate_delay(self._attempt, self.config)
        self._stats.total_retries += 1
        self._stats.total_delay_seconds += delay

        logger.info(
            "Retrying operation",
            operation=self.operation_name,
            attempt=self._attempt,
            max_attempts=self.config.max_attempts,
            delay_seconds=round(delay, 2),
            error_type=type(error).__name__,
            error=str(error),
        )

        await asyncio.sleep(delay)

    async def __aenter__(self):
        """Enter the retry context."""
        self._attempt = 0
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the retry context."""
        self._stats.total_attempts = self._attempt
        return False

    def next_attempt(self) -> int:
        """Move to the next attempt and return the attempt number."""
        self._attempt += 1
        return self._attempt


def retry(
    config: RetryConfig | None = None,
    operation_name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding retry logic to async functions.

    Usage:
        @retry(RetryConfig(max_attempts=5))
        async def fetch_data():
            ...

        # Or with defaults:
        @retry()
        async def fetch_data():
            ...
    """
    _config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with RetryContext(_config, name) as ctx:
                while ctx.should_continue():
                    ctx.next_attempt()
                    try:
                        result = await func(*args, **kwargs)
                        ctx.record_success()
                        return result
                    except Exception as e:
                        await ctx.record_failure(e)

            # This should not be reached, but satisfy type checker
            raise RetryExhaustedError(
                f"Operation '{name}' failed",
                attempts=ctx.attempt,
                last_error=ctx._last_error,
            )

        return wrapper

    return decorator


async def retry_async(
    operation: Callable[[], T],
    config: RetryConfig | None = None,
    operation_name: str = "operation",
) -> T:
    """
    Execute an async operation with retry logic.

    Usage:
        result = await retry_async(
            lambda: fetch_data(url),
            config=RetryConfig(max_attempts=5),
            operation_name="fetch_data",
        )
    """
    _config = config or RetryConfig()

    async with RetryContext(_config, operation_name) as ctx:
        while ctx.should_continue():
            ctx.next_attempt()
            try:
                result = await operation()
                ctx.record_success()
                return result
            except Exception as e:
                await ctx.record_failure(e)

    # This should not be reached
    raise RetryExhaustedError(
        f"Operation '{operation_name}' failed",
        attempts=ctx.attempt,
        last_error=ctx._last_error,
    )
