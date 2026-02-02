"""
Resilience Module for MLOps Agent.

Provides fault tolerance patterns for handling transient failures and
protecting against cascading failures in async operations.

Components:
- circuit_breaker: Circuit breaker pattern for failing fast
- retry: Retry with exponential backoff for transient failures
- timeout: Timeout management for async operations
- bulkhead: Resource isolation to limit concurrent access

Usage:
    # Circuit Breaker
    from resilience import CircuitBreaker, CircuitBreakerConfig

    breaker = CircuitBreaker("external_api", CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=30.0,
    ))

    @breaker
    async def call_external_api():
        ...

    # Retry with Exponential Backoff
    from resilience import retry, RetryConfig

    @retry(RetryConfig(max_attempts=5, base_delay_seconds=1.0))
    async def fetch_data():
        ...

    # Timeout
    from resilience import timeout, with_timeout

    @timeout(30.0)
    async def slow_operation():
        ...

    # Or inline:
    result = await with_timeout(
        lambda: fetch_data(url),
        timeout_seconds=30.0,
    )

    # Bulkhead
    from resilience import Bulkhead, BulkheadConfig

    bulkhead = Bulkhead("database", BulkheadConfig(max_concurrent=5))

    @bulkhead
    async def query_database():
        ...

    # Combined patterns
    from resilience import (
        CircuitBreaker,
        retry,
        timeout,
        Bulkhead,
    )

    # Apply multiple resilience patterns
    circuit = CircuitBreaker("api")
    rate_limiter = Bulkhead("api", BulkheadConfig(max_concurrent=10))

    @circuit
    @rate_limiter
    @retry(RetryConfig(max_attempts=3))
    @timeout(30.0)
    async def resilient_api_call():
        ...
"""

from .bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadFullError,
    BulkheadRegistry,
    BulkheadStats,
    ThreadPoolBulkhead,
    bulkhead_registry,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitStats,
    circuit_breaker_registry,
)
from .retry import (
    RetryConfig,
    RetryContext,
    RetryExhaustedError,
    RetryStats,
    calculate_delay,
    retry,
    retry_async,
    should_retry,
)
from .timeout import (
    AdaptiveTimeout,
    TimeoutConfig,
    TimeoutError,
    TimeoutManager,
    TimeoutStats,
    timeout,
    with_timeout,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    "CircuitStats",
    "circuit_breaker_registry",
    # Retry
    "RetryConfig",
    "RetryContext",
    "RetryExhaustedError",
    "RetryStats",
    "retry",
    "retry_async",
    "calculate_delay",
    "should_retry",
    # Timeout
    "TimeoutConfig",
    "TimeoutError",
    "TimeoutManager",
    "TimeoutStats",
    "AdaptiveTimeout",
    "timeout",
    "with_timeout",
    # Bulkhead
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadFullError",
    "BulkheadRegistry",
    "BulkheadStats",
    "ThreadPoolBulkhead",
    "bulkhead_registry",
]
