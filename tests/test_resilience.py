#!/usr/bin/env python3
"""
Tests for resilience module - circuit breaker, retry, timeout, and bulkhead patterns.

Run with: pytest tests/test_resilience.py -v
"""

import asyncio

import pytest

from resilience import (
    AdaptiveTimeout,
    Bulkhead,
    BulkheadConfig,
    BulkheadFullError,
    BulkheadRegistry,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    RetryConfig,
    RetryContext,
    RetryExhaustedError,
    ThreadPoolBulkhead,
    TimeoutConfig,
    TimeoutManager,
    calculate_delay,
    retry,
    retry_async,
    should_retry,
    timeout,
    with_timeout,
)
from resilience.timeout import TimeoutError

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def circuit_config():
    """Create a circuit breaker config for testing."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1.0,
        half_open_max_calls=2,
    )


@pytest.fixture
def retry_config():
    """Create a retry config for testing."""
    return RetryConfig(
        max_attempts=3,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.fixture
def bulkhead_config():
    """Create a bulkhead config for testing."""
    return BulkheadConfig(
        max_concurrent=2,
        max_queue_size=2,
        queue_timeout_seconds=0.5,
    )


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.excluded_exceptions == ()

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=60.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_create_circuit_breaker(self, circuit_config):
        """Test creating a circuit breaker."""
        breaker = CircuitBreaker("test_service", circuit_config)
        assert breaker.name == "test_service"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0

    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_config):
        """Test successful calls through circuit breaker."""
        breaker = CircuitBreaker("test", circuit_config)

        async with breaker:
            pass  # Simulate successful call

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.failed_calls == 0

    @pytest.mark.asyncio
    async def test_failed_calls_open_circuit(self, circuit_config):
        """Test that failures open the circuit."""
        breaker = CircuitBreaker("test", circuit_config)

        # Cause failures up to threshold
        for i in range(circuit_config.failure_threshold):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.consecutive_failures == circuit_config.failure_threshold

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, circuit_config):
        """Test that open circuit rejects calls."""
        breaker = CircuitBreaker("test", circuit_config)

        # Force circuit open
        for _ in range(circuit_config.failure_threshold):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError) as exc_info:
            async with breaker:
                pass

        assert exc_info.value.state == CircuitState.OPEN
        assert breaker.stats.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, circuit_config):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
        )
        breaker = CircuitBreaker("test", config)

        # Force circuit open
        for _ in range(2):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open
        async with breaker:
            pass

        # After success in half-open, should need more successes to close
        assert breaker.state == CircuitState.HALF_OPEN or breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_usage(self, circuit_config):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker("test", circuit_config)

        @breaker
        async def protected_function():
            return "success"

        result = await protected_function()
        assert result == "success"
        assert breaker.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test that excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)

        # ValueError should not count as failure
        for _ in range(5):
            try:
                async with breaker:
                    raise ValueError("Excluded")
            except ValueError:
                pass

        # Circuit should still be closed
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0

    def test_reset(self, circuit_config):
        """Test resetting circuit breaker."""
        breaker = CircuitBreaker("test", circuit_config)
        breaker._stats.total_calls = 100
        breaker._stats.failed_calls = 50
        breaker._state = CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0
        assert breaker.stats.failed_calls == 0


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()

        breaker1 = await registry.get_or_create("service_a")
        breaker2 = await registry.get_or_create("service_a")

        assert breaker1 is breaker2

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting stats for all circuit breakers."""
        registry = CircuitBreakerRegistry()

        await registry.get_or_create("service_a")
        await registry.get_or_create("service_b")

        stats = registry.get_all_stats()
        assert "service_a" in stats
        assert "service_b" in stats

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        # Create breakers synchronously for this test
        registry._breakers["service_a"] = CircuitBreaker("service_a")
        registry._breakers["service_b"] = CircuitBreaker("service_b")

        registry.reset_all()
        # Should not raise


# ============================================================================
# Retry Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_seconds=0.5,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.base_delay_seconds == 0.5
        assert config.jitter is False


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=100.0,
            jitter=False,
        )

        assert calculate_delay(1, config) == 1.0  # 1 * 2^0
        assert calculate_delay(2, config) == 2.0  # 1 * 2^1
        assert calculate_delay(3, config) == 4.0  # 1 * 2^2
        assert calculate_delay(4, config) == 8.0  # 1 * 2^3

    def test_max_delay_cap(self):
        """Test that delay is capped at max."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=5.0,
            jitter=False,
        )

        # Attempt 10 would be 1 * 2^9 = 512, but capped at 5
        assert calculate_delay(10, config) == 5.0

    def test_jitter_variation(self):
        """Test that jitter adds variation."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            jitter=True,
            jitter_factor=0.5,
        )

        delays = [calculate_delay(1, config) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1


class TestShouldRetry:
    """Tests for should_retry function."""

    def test_retryable_exception(self):
        """Test retryable exceptions return True."""
        config = RetryConfig(retryable_exceptions=(ValueError, TypeError))

        assert should_retry(ValueError("test"), config) is True
        assert should_retry(TypeError("test"), config) is True

    def test_non_retryable_exception(self):
        """Test non-retryable exceptions return False."""
        config = RetryConfig(
            retryable_exceptions=(ValueError,),
            non_retryable_exceptions=(KeyError,),
        )

        assert should_retry(KeyError("test"), config) is False

    def test_non_retryable_takes_precedence(self):
        """Test non-retryable exceptions take precedence."""
        config = RetryConfig(
            retryable_exceptions=(Exception,),
            non_retryable_exceptions=(ValueError,),
        )

        # ValueError is both retryable (via Exception) and non-retryable
        # Non-retryable should take precedence
        assert should_retry(ValueError("test"), config) is False


class TestRetryContext:
    """Tests for RetryContext class."""

    @pytest.mark.asyncio
    async def test_successful_operation(self, retry_config):
        """Test successful operation doesn't retry."""
        async with RetryContext(retry_config, "test_op") as ctx:
            ctx.next_attempt()
            ctx.record_success()

        assert ctx.attempt == 1
        assert ctx.stats.successful_attempts == 1
        assert ctx.stats.total_retries == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, retry_config):
        """Test retry on failure."""
        async with RetryContext(retry_config, "test_op") as ctx:
            while ctx.should_continue():
                ctx.next_attempt()
                if ctx.attempt < 2:
                    try:
                        raise ValueError("Test error")
                    except ValueError as e:
                        await ctx.record_failure(e)
                else:
                    ctx.record_success()
                    break

        assert ctx.attempt == 2
        assert ctx.stats.total_retries == 1

    @pytest.mark.asyncio
    async def test_exhaust_retries(self, retry_config):
        """Test exhausting all retries."""
        with pytest.raises(RetryExhaustedError) as exc_info:
            async with RetryContext(retry_config, "test_op") as ctx:
                while ctx.should_continue():
                    ctx.next_attempt()
                    try:
                        raise ValueError("Test error")
                    except ValueError as e:
                        await ctx.record_failure(e)

        assert exc_info.value.attempts == retry_config.max_attempts


class TestRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_function(self, retry_config):
        """Test successful function doesn't retry."""
        call_count = 0

        @retry(retry_config)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_until_success(self, retry_config):
        """Test function retries until success."""
        call_count = 0

        @retry(retry_config)
        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not yet")
            return "success"

        result = await eventually_succeeds()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, retry_config):
        """Test exhausting retries raises error."""

        @retry(retry_config)
        async def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError):
            await always_fails()


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_retry_async_success(self, retry_config):
        """Test retry_async with successful operation."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await retry_async(operation, retry_config, "test_op")
        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_with_retries(self, retry_config):
        """Test retry_async with retries."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry me")
            return "result"

        result = await retry_async(operation, retry_config, "test_op")
        assert result == "result"
        assert call_count == 3


# ============================================================================
# Timeout Tests
# ============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimeoutConfig()
        assert config.timeout_seconds == 30.0
        assert config.cancel_on_timeout is True
        assert config.log_timeout is True


class TestTimeoutManager:
    """Tests for TimeoutManager class."""

    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Test operation within timeout."""
        manager = TimeoutManager("test", TimeoutConfig(timeout_seconds=1.0))

        @manager
        async def fast_operation():
            return "result"

        result = await fast_operation()
        assert result == "result"
        assert manager.stats.successful_calls == 1
        assert manager.stats.timed_out_calls == 0

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test operation exceeding timeout."""
        manager = TimeoutManager("test", TimeoutConfig(timeout_seconds=0.1))

        @manager
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "result"

        with pytest.raises(TimeoutError) as exc_info:
            await slow_operation()

        assert exc_info.value.operation_name == "test"
        assert exc_info.value.timeout_seconds == 0.1


class TestTimeoutDecorator:
    """Tests for timeout decorator."""

    @pytest.mark.asyncio
    async def test_timeout_decorator_success(self):
        """Test timeout decorator with successful operation."""

        @timeout(1.0)
        async def fast_operation():
            return "result"

        result = await fast_operation()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_timeout_decorator_exceeded(self):
        """Test timeout decorator with exceeded timeout."""

        @timeout(0.1)
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "result"

        with pytest.raises(TimeoutError):
            await slow_operation()


class TestWithTimeout:
    """Tests for with_timeout function."""

    @pytest.mark.asyncio
    async def test_with_timeout_success(self):
        """Test with_timeout with successful operation."""

        async def fast_operation():
            return "result"

        result = await with_timeout(fast_operation, 1.0, "test")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_with_timeout_exceeded(self):
        """Test with_timeout with exceeded timeout."""

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "result"

        with pytest.raises(TimeoutError):
            await with_timeout(slow_operation, 0.1, "test")


class TestAdaptiveTimeout:
    """Tests for AdaptiveTimeout class."""

    def test_initial_timeout(self):
        """Test initial timeout value."""
        adaptive = AdaptiveTimeout("test", initial_timeout=10.0)
        assert adaptive.current_timeout == 10.0

    def test_timeout_adjusts_with_data(self):
        """Test timeout adjusts based on recorded durations."""
        adaptive = AdaptiveTimeout(
            "test",
            initial_timeout=30.0,
            min_timeout=1.0,
            max_timeout=60.0,
            window_size=20,
        )

        # Record fast operations
        for _ in range(15):
            adaptive.record_duration(0.5)

        # Timeout should decrease (95th percentile of ~0.5 * 1.5)
        assert adaptive.current_timeout < 30.0
        assert adaptive.current_timeout >= 1.0

    def test_timeout_respects_min_max(self):
        """Test timeout respects min/max bounds."""
        adaptive = AdaptiveTimeout(
            "test",
            initial_timeout=10.0,
            min_timeout=5.0,
            max_timeout=20.0,
        )

        # Record very fast operations
        for _ in range(20):
            adaptive.record_duration(0.01)

        assert adaptive.current_timeout >= 5.0

        # Record very slow operations
        for _ in range(20):
            adaptive.record_duration(100.0)

        assert adaptive.current_timeout <= 20.0


# ============================================================================
# Bulkhead Tests
# ============================================================================


class TestBulkheadConfig:
    """Tests for BulkheadConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BulkheadConfig()
        assert config.max_concurrent == 10
        assert config.max_queue_size is None
        assert config.queue_timeout_seconds is None


class TestBulkhead:
    """Tests for Bulkhead class."""

    @pytest.mark.asyncio
    async def test_single_execution(self, bulkhead_config):
        """Test single execution through bulkhead."""
        bulkhead = Bulkhead("test", bulkhead_config)

        async with bulkhead:
            pass

        assert bulkhead.stats.total_calls == 1
        assert bulkhead.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """Test concurrent limit is enforced."""
        config = BulkheadConfig(max_concurrent=2, max_queue_size=None)
        bulkhead = Bulkhead("test", config)

        executing = 0
        max_executing = 0

        async def task():
            nonlocal executing, max_executing
            async with bulkhead:
                executing += 1
                max_executing = max(max_executing, executing)
                await asyncio.sleep(0.1)
                executing -= 1

        # Try to run more tasks than allowed
        with pytest.raises(BulkheadFullError):
            # Start two tasks (which is the limit)
            task1 = asyncio.create_task(task())
            task2 = asyncio.create_task(task())
            await asyncio.sleep(0.01)  # Let them start

            # Third task should fail immediately (no queue)
            await task()

        # Clean up
        await asyncio.gather(task1, task2, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_queue_behavior(self):
        """Test queuing behavior."""
        config = BulkheadConfig(
            max_concurrent=1,
            max_queue_size=2,
            queue_timeout_seconds=1.0,
        )
        bulkhead = Bulkhead("test", config)

        results = []

        async def task(id):
            async with bulkhead:
                await asyncio.sleep(0.05)
                results.append(id)

        # Run tasks that should queue
        await asyncio.gather(task(1), task(2), task(3))

        assert len(results) == 3
        assert bulkhead.stats.successful_calls == 3

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self):
        """Test rejection when queue is full."""
        config = BulkheadConfig(
            max_concurrent=1,
            max_queue_size=1,
        )
        bulkhead = Bulkhead("test", config)

        async def slow_task():
            async with bulkhead:
                await asyncio.sleep(0.5)

        # Start first task (executes)
        task1 = asyncio.create_task(slow_task())
        await asyncio.sleep(0.01)

        # Start second task (queues)
        task2 = asyncio.create_task(slow_task())
        await asyncio.sleep(0.01)

        # Third task should be rejected (queue full)
        with pytest.raises(BulkheadFullError):
            await slow_task()

        # Clean up
        task1.cancel()
        task2.cancel()
        await asyncio.gather(task1, task2, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_decorator_usage(self, bulkhead_config):
        """Test bulkhead as decorator."""
        bulkhead = Bulkhead("test", bulkhead_config)

        @bulkhead
        async def protected_function():
            return "success"

        result = await protected_function()
        assert result == "success"
        assert bulkhead.stats.successful_calls == 1


class TestBulkheadRegistry:
    """Tests for BulkheadRegistry."""

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """Test getting or creating bulkheads."""
        registry = BulkheadRegistry()

        bulkhead1 = await registry.get_or_create("db")
        bulkhead2 = await registry.get_or_create("db")

        assert bulkhead1 is bulkhead2

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting stats for all bulkheads."""
        registry = BulkheadRegistry()

        await registry.get_or_create("db")
        await registry.get_or_create("api")

        stats = registry.get_all_stats()
        assert "db" in stats
        assert "api" in stats


class TestThreadPoolBulkhead:
    """Tests for ThreadPoolBulkhead."""

    @pytest.mark.asyncio
    async def test_blocking_operation(self):
        """Test running blocking operation in thread pool."""
        bulkhead = ThreadPoolBulkhead("io", max_workers=2)

        @bulkhead
        def blocking_operation():
            import time

            time.sleep(0.01)
            return "result"

        result = await blocking_operation()
        assert result == "result"
        assert bulkhead.stats.successful_calls == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestCombinedPatterns:
    """Tests for combining multiple resilience patterns."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test combining retry with circuit breaker."""
        circuit = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=0.1),
        )
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,
            jitter=False,
        )

        call_count = 0

        @circuit
        @retry(retry_config)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry me")
            return "success"

        result = await flaky_operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_with_retry(self):
        """Test combining timeout with retry."""
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.01,
            jitter=False,
        )

        call_count = 0

        @retry(retry_config)
        @timeout(0.5)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                await asyncio.sleep(1.0)  # Will timeout
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_bulkhead_with_circuit_breaker(self):
        """Test combining bulkhead with circuit breaker."""
        bulkhead = Bulkhead("test", BulkheadConfig(max_concurrent=5))
        circuit = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))

        @circuit
        @bulkhead
        async def protected_operation():
            return "success"

        results = await asyncio.gather(*[protected_operation() for _ in range(3)])
        assert all(r == "success" for r in results)
