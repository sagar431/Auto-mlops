"""
Bulkhead Pattern Implementation.

Provides resource isolation to prevent cascading failures by limiting
concurrent access to specific resources or operations.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from observability import get_logger

logger = get_logger("resilience.bulkhead")

T = TypeVar("T")


class BulkheadFullError(Exception):
    """Raised when bulkhead has no available capacity."""

    def __init__(
        self,
        name: str,
        max_concurrent: int,
        queue_size: int | None = None,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        msg = f"Bulkhead '{name}' is full (max_concurrent={max_concurrent}"
        if queue_size is not None:
            msg += f", queue_size={queue_size}"
        msg += ")"
        super().__init__(msg)


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead behavior."""

    max_concurrent: int = 10
    """Maximum number of concurrent executions."""

    max_queue_size: int | None = None
    """Maximum number of waiting requests. None means no queue (fail fast)."""

    queue_timeout_seconds: float | None = None
    """Timeout for waiting in queue. None means wait indefinitely."""


@dataclass
class BulkheadStats:
    """Statistics for bulkhead operations."""

    total_calls: int = 0
    successful_calls: int = 0
    rejected_calls: int = 0
    timed_out_calls: int = 0
    current_concurrent: int = 0
    current_queued: int = 0
    max_concurrent_reached: int = 0
    max_queued_reached: int = 0


class Bulkhead:
    """
    Bulkhead for limiting concurrent operations.

    Usage:
        bulkhead = Bulkhead("database", BulkheadConfig(max_concurrent=5))

        @bulkhead
        async def query_database():
            ...

        # Or manually:
        async with bulkhead:
            await query_database()
    """

    def __init__(
        self,
        name: str,
        config: BulkheadConfig | None = None,
    ):
        self.name = name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._stats = BulkheadStats()
        self._queue_size = 0
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> BulkheadStats:
        """Get bulkhead statistics."""
        return self._stats

    @property
    def available_permits(self) -> int:
        """Get number of available permits."""
        # Semaphore internal value represents available permits
        return self._semaphore._value

    @property
    def current_concurrent(self) -> int:
        """Get number of currently executing operations."""
        return self.config.max_concurrent - self._semaphore._value

    async def _try_acquire(self) -> bool:
        """Try to acquire a permit without blocking."""
        return self._semaphore.locked() is False and self._semaphore._value > 0

    async def acquire(self) -> None:
        """
        Acquire a permit from the bulkhead.

        Raises:
            BulkheadFullError: If bulkhead is full and no queue or queue is full
            asyncio.TimeoutError: If queue timeout is exceeded
        """
        async with self._lock:
            # Check if we can execute immediately
            if self._semaphore._value > 0:
                await self._semaphore.acquire()
                self._stats.current_concurrent = self.current_concurrent
                self._stats.max_concurrent_reached = max(
                    self._stats.max_concurrent_reached,
                    self._stats.current_concurrent,
                )
                return

            # Check if we should queue
            if self.config.max_queue_size is None:
                # No queue, fail fast
                self._stats.rejected_calls += 1
                raise BulkheadFullError(
                    self.name,
                    self.config.max_concurrent,
                )

            if self._queue_size >= self.config.max_queue_size:
                # Queue is full
                self._stats.rejected_calls += 1
                raise BulkheadFullError(
                    self.name,
                    self.config.max_concurrent,
                    self.config.max_queue_size,
                )

            # Add to queue
            self._queue_size += 1
            self._stats.current_queued = self._queue_size
            self._stats.max_queued_reached = max(
                self._stats.max_queued_reached,
                self._queue_size,
            )

        # Wait for permit (outside lock)
        try:
            if self.config.queue_timeout_seconds:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.queue_timeout_seconds,
                )
            else:
                await self._semaphore.acquire()
        except asyncio.TimeoutError:
            async with self._lock:
                self._queue_size -= 1
                self._stats.current_queued = self._queue_size
                self._stats.timed_out_calls += 1
            raise
        finally:
            async with self._lock:
                self._queue_size -= 1
                self._stats.current_queued = self._queue_size

        async with self._lock:
            self._stats.current_concurrent = self.current_concurrent
            self._stats.max_concurrent_reached = max(
                self._stats.max_concurrent_reached,
                self._stats.current_concurrent,
            )

    def release(self) -> None:
        """Release a permit back to the bulkhead."""
        self._semaphore.release()
        self._stats.current_concurrent = self.current_concurrent

    async def __aenter__(self):
        """Async context manager entry."""
        self._stats.total_calls += 1
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
        if exc_val is None:
            self._stats.successful_calls += 1
        return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for protecting async functions with bulkhead."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with self:
                return await func(*args, **kwargs)

        return wrapper


class BulkheadRegistry:
    """
    Registry for managing multiple bulkheads.

    Usage:
        registry = BulkheadRegistry()
        bulkhead = await registry.get_or_create("database", BulkheadConfig(max_concurrent=5))
    """

    def __init__(self, default_config: BulkheadConfig | None = None):
        self._bulkheads: dict[str, Bulkhead] = {}
        self._default_config = default_config or BulkheadConfig()
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: BulkheadConfig | None = None,
    ) -> Bulkhead:
        """Get existing or create new bulkhead."""
        async with self._lock:
            if name not in self._bulkheads:
                self._bulkheads[name] = Bulkhead(
                    name,
                    config or self._default_config,
                )
                logger.info(
                    "Bulkhead created",
                    name=name,
                    max_concurrent=self._bulkheads[name].config.max_concurrent,
                    max_queue_size=self._bulkheads[name].config.max_queue_size,
                )
            return self._bulkheads[name]

    def get(self, name: str) -> Bulkhead | None:
        """Get bulkhead by name."""
        return self._bulkheads.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all bulkheads."""
        return {
            name: {
                "total_calls": bulkhead.stats.total_calls,
                "successful_calls": bulkhead.stats.successful_calls,
                "rejected_calls": bulkhead.stats.rejected_calls,
                "timed_out_calls": bulkhead.stats.timed_out_calls,
                "current_concurrent": bulkhead.stats.current_concurrent,
                "current_queued": bulkhead.stats.current_queued,
                "max_concurrent_reached": bulkhead.stats.max_concurrent_reached,
                "max_queued_reached": bulkhead.stats.max_queued_reached,
                "available_permits": bulkhead.available_permits,
            }
            for name, bulkhead in self._bulkheads.items()
        }


# Global registry instance
bulkhead_registry = BulkheadRegistry()


class ThreadPoolBulkhead:
    """
    Bulkhead for running blocking operations in a thread pool.

    Isolates blocking operations from the async event loop while
    limiting concurrent thread usage.

    Usage:
        bulkhead = ThreadPoolBulkhead("file_io", max_workers=4)

        @bulkhead
        def read_file(path):
            with open(path) as f:
                return f.read()

        # Called as async:
        content = await read_file("/path/to/file")
    """

    def __init__(
        self,
        name: str,
        max_workers: int = 4,
    ):
        self.name = name
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
        self._stats = BulkheadStats()

    @property
    def stats(self) -> BulkheadStats:
        """Get bulkhead statistics."""
        return self._stats

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for running blocking functions in thread pool."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            self._stats.total_calls += 1

            async with self._semaphore:
                self._stats.current_concurrent = self.max_workers - self._semaphore._value
                self._stats.max_concurrent_reached = max(
                    self._stats.max_concurrent_reached,
                    self._stats.current_concurrent,
                )

                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: func(*args, **kwargs),
                    )
                    self._stats.successful_calls += 1
                    return result
                finally:
                    self._stats.current_concurrent = self.max_workers - self._semaphore._value

        return wrapper
