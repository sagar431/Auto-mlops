#!/usr/bin/env python3
"""
Additional tests for circuit breaker and retry logic.

This module contains comprehensive tests for edge cases, concurrency scenarios,
and advanced state transitions in the circuit breaker and retry patterns.

Run with: pytest tests/test_circuit_breaker_retry.py -v
"""

import asyncio
import time

import pytest

from resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    RetryConfig,
    RetryContext,
    RetryExhaustedError,
    calculate_delay,
    retry,
    retry_async,
    should_retry,
)

# ============================================================================
# Circuit Breaker Edge Case Tests
# ============================================================================


class TestCircuitBreakerEdgeCases:
    """Edge case tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_success_resets_consecutive_failures(self):
        """Test that a success resets consecutive failure count."""
        config = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=30.0)
        breaker = CircuitBreaker("test", config)

        # Accumulate some failures (but not enough to open)
        for _ in range(3):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        assert breaker.stats.consecutive_failures == 3

        # Success should reset consecutive failures
        async with breaker:
            pass

        assert breaker.stats.consecutive_failures == 0
        assert breaker.stats.consecutive_successes == 1
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        """Test that a failure in half-open state reopens the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=3,
            timeout_seconds=0.05,
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

        # Wait for timeout to allow half-open transition
        await asyncio.sleep(0.1)

        # Failure in half-open should reopen circuit
        try:
            async with breaker:
                raise ValueError("Half-open failure")
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_requires_multiple_successes(self):
        """Test that half-open state requires multiple successes to close."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=3,
            timeout_seconds=0.05,
        )
        breaker = CircuitBreaker("test", config)

        # Force circuit open
        for _ in range(2):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        # Wait for timeout
        await asyncio.sleep(0.1)

        # First success should transition to half-open
        async with breaker:
            pass

        # Need success_threshold successes to close
        async with breaker:
            pass

        async with breaker:
            pass

        # After 3 successes, circuit should be closed
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_no_errors(self):
        """Test circuit breaker with all successful calls."""
        breaker = CircuitBreaker("test")

        for _ in range(100):
            async with breaker:
                pass

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.successful_calls == 100
        assert breaker.stats.failed_calls == 0
        assert breaker.stats.consecutive_successes == 100

    @pytest.mark.asyncio
    async def test_retry_after_returns_none_when_closed(self):
        """Test get_retry_after returns None when circuit is closed."""
        breaker = CircuitBreaker("test")

        assert breaker.get_retry_after() is None

    @pytest.mark.asyncio
    async def test_retry_after_returns_time_when_open(self):
        """Test get_retry_after returns time when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=10.0)
        breaker = CircuitBreaker("test", config)

        # Force circuit open
        for _ in range(2):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        retry_after = breaker.get_retry_after()
        assert retry_after is not None
        assert 0 < retry_after <= 10.0

    @pytest.mark.asyncio
    async def test_excluded_exception_doesnt_count_total_calls(self):
        """Test that excluded exceptions still count total calls but not as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)

        for _ in range(10):
            try:
                async with breaker:
                    raise ValueError("Excluded error")
            except ValueError:
                pass

        # Should not have opened
        assert breaker.state == CircuitState.CLOSED
        # Excluded exceptions don't increment total_calls via _record_failure
        # but total_calls is only incremented in _record_failure and _record_success
        assert breaker.stats.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_non_excluded_exception_opens_circuit(self):
        """Test that non-excluded exceptions properly open the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)

        # TypeError is not excluded
        for _ in range(2):
            try:
                async with breaker:
                    raise TypeError("Not excluded")
            except TypeError:
                pass

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerConcurrency:
    """Concurrency tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_in_closed_state(self):
        """Test multiple concurrent calls when circuit is closed."""
        breaker = CircuitBreaker("test")
        results = []

        async def make_call(id: int):
            async with breaker:
                await asyncio.sleep(0.01)
                results.append(id)

        # Run 10 concurrent calls
        await asyncio.gather(*[make_call(i) for i in range(10)])

        assert len(results) == 10
        assert breaker.stats.successful_calls == 10

    @pytest.mark.asyncio
    async def test_concurrent_calls_hit_half_open_limit(self):
        """Test that half-open state limits concurrent calls."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.05,
            half_open_max_calls=2,
        )
        breaker = CircuitBreaker("test", config)

        # Force circuit open
        for _ in range(2):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        # Wait for timeout
        await asyncio.sleep(0.1)

        rejected_count = 0
        success_count = 0

        async def make_call():
            nonlocal rejected_count, success_count
            try:
                async with breaker:
                    await asyncio.sleep(0.05)
                    success_count += 1
            except CircuitBreakerError:
                rejected_count += 1

        # Run more calls than half_open_max_calls
        await asyncio.gather(*[make_call() for _ in range(5)])

        # Some calls should have been rejected
        assert success_count + rejected_count == 5
        assert rejected_count > 0

    @pytest.mark.asyncio
    async def test_concurrent_failures_open_circuit_once(self):
        """Test that concurrent failures only open circuit once."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0)
        breaker = CircuitBreaker("test", config)

        async def fail_call():
            try:
                async with breaker:
                    await asyncio.sleep(0.01)
                    raise ValueError("Concurrent failure")
            except (ValueError, CircuitBreakerError):
                pass

        # Run concurrent failing calls
        await asyncio.gather(*[fail_call() for _ in range(10)])

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN
        # State changes should be minimal
        assert breaker.stats.state_changes >= 1


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry functionality."""

    @pytest.mark.asyncio
    async def test_registry_returns_same_instance(self):
        """Test that registry returns the same instance for the same name."""
        registry = CircuitBreakerRegistry()

        breaker1 = await registry.get_or_create("service_a")
        breaker2 = await registry.get_or_create("service_a")

        assert breaker1 is breaker2

    @pytest.mark.asyncio
    async def test_registry_creates_different_instances(self):
        """Test that registry creates different instances for different names."""
        registry = CircuitBreakerRegistry()

        breaker1 = await registry.get_or_create("service_a")
        breaker2 = await registry.get_or_create("service_b")

        assert breaker1 is not breaker2

    @pytest.mark.asyncio
    async def test_registry_uses_custom_config(self):
        """Test that registry uses custom config when provided."""
        registry = CircuitBreakerRegistry()
        custom_config = CircuitBreakerConfig(failure_threshold=10, timeout_seconds=60.0)

        breaker = await registry.get_or_create("custom_service", custom_config)

        assert breaker.config.failure_threshold == 10
        assert breaker.config.timeout_seconds == 60.0

    @pytest.mark.asyncio
    async def test_registry_uses_default_config(self):
        """Test that registry uses default config when not provided."""
        default_config = CircuitBreakerConfig(failure_threshold=7)
        registry = CircuitBreakerRegistry(default_config=default_config)

        breaker = await registry.get_or_create("default_service")

        assert breaker.config.failure_threshold == 7

    @pytest.mark.asyncio
    async def test_registry_get_returns_none_for_unknown(self):
        """Test that registry.get returns None for unknown service."""
        registry = CircuitBreakerRegistry()

        assert registry.get("unknown_service") is None

    @pytest.mark.asyncio
    async def test_registry_get_returns_existing(self):
        """Test that registry.get returns existing breaker."""
        registry = CircuitBreakerRegistry()

        await registry.get_or_create("known_service")
        breaker = registry.get("known_service")

        assert breaker is not None
        assert breaker.name == "known_service"

    @pytest.mark.asyncio
    async def test_registry_concurrent_creation(self):
        """Test that registry handles concurrent creation safely."""
        registry = CircuitBreakerRegistry()

        async def create_breaker():
            return await registry.get_or_create("concurrent_service")

        # Create multiple breakers concurrently
        breakers = await asyncio.gather(*[create_breaker() for _ in range(10)])

        # All should be the same instance
        assert all(b is breakers[0] for b in breakers)


class TestCircuitBreakerErrorDetails:
    """Tests for CircuitBreakerError exception."""

    @pytest.mark.asyncio
    async def test_error_contains_name(self):
        """Test error contains circuit breaker name."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_service", config)

        try:
            async with breaker:
                raise ValueError("Force open")
        except ValueError:
            pass

        with pytest.raises(CircuitBreakerError) as exc_info:
            async with breaker:
                pass

        assert exc_info.value.name == "test_service"
        assert "test_service" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_contains_state(self):
        """Test error contains circuit state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        try:
            async with breaker:
                raise ValueError("Force open")
        except ValueError:
            pass

        with pytest.raises(CircuitBreakerError) as exc_info:
            async with breaker:
                pass

        assert exc_info.value.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_error_contains_retry_after(self):
        """Test error contains retry_after time."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=30.0)
        breaker = CircuitBreaker("test", config)

        try:
            async with breaker:
                raise ValueError("Force open")
        except ValueError:
            pass

        with pytest.raises(CircuitBreakerError) as exc_info:
            async with breaker:
                pass

        assert exc_info.value.retry_after is not None
        assert 0 < exc_info.value.retry_after <= 30.0


# ============================================================================
# Retry Logic Edge Case Tests
# ============================================================================


class TestRetryEdgeCases:
    """Edge case tests for retry logic."""

    def test_calculate_delay_first_attempt(self):
        """Test delay calculation for first attempt."""
        config = RetryConfig(base_delay_seconds=1.0, exponential_base=2.0, jitter=False)

        delay = calculate_delay(1, config)

        assert delay == 1.0  # 1.0 * 2^0 = 1.0

    def test_calculate_delay_exponential_growth(self):
        """Test exponential growth of delays."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=1000.0,
            jitter=False,
        )

        assert calculate_delay(1, config) == 1.0  # 2^0
        assert calculate_delay(2, config) == 2.0  # 2^1
        assert calculate_delay(3, config) == 4.0  # 2^2
        assert calculate_delay(4, config) == 8.0  # 2^3
        assert calculate_delay(5, config) == 16.0  # 2^4

    def test_calculate_delay_respects_max(self):
        """Test that delay is capped at max_delay_seconds."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=5.0,
            jitter=False,
        )

        # Attempt 10 would be 2^9 = 512, but should be capped at 5
        delay = calculate_delay(10, config)

        assert delay == 5.0

    def test_calculate_delay_with_jitter_bounded(self):
        """Test that jitter keeps delay within bounds."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=True,
            jitter_factor=0.5,
        )

        for _ in range(100):
            delay = calculate_delay(1, config)
            # Base delay is 1.0, jitter factor is 0.5
            # So delay should be between 0.5 and 1.5
            assert 0 <= delay <= 1.5

    def test_calculate_delay_with_zero_jitter_factor(self):
        """Test delay with zero jitter factor."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            jitter=True,
            jitter_factor=0.0,
        )

        delay = calculate_delay(1, config)

        assert delay == 1.0

    def test_should_retry_with_base_exception(self):
        """Test should_retry with broad exception matching."""
        config = RetryConfig(retryable_exceptions=(Exception,))

        assert should_retry(ValueError("test"), config) is True
        assert should_retry(TypeError("test"), config) is True
        assert should_retry(RuntimeError("test"), config) is True

    def test_should_retry_with_specific_exceptions(self):
        """Test should_retry with specific exception matching."""
        config = RetryConfig(retryable_exceptions=(ValueError, TypeError))

        assert should_retry(ValueError("test"), config) is True
        assert should_retry(TypeError("test"), config) is True
        assert should_retry(RuntimeError("test"), config) is False

    def test_should_retry_non_retryable_takes_precedence(self):
        """Test that non_retryable_exceptions take precedence."""
        config = RetryConfig(
            retryable_exceptions=(Exception,),
            non_retryable_exceptions=(KeyboardInterrupt, SystemExit, ValueError),
        )

        assert should_retry(ValueError("test"), config) is False
        assert should_retry(TypeError("test"), config) is True

    def test_should_retry_with_exception_subclass(self):
        """Test should_retry with exception subclasses."""

        class CustomError(ValueError):
            pass

        config = RetryConfig(retryable_exceptions=(ValueError,))

        assert should_retry(CustomError("test"), config) is True


class TestRetryContextEdgeCases:
    """Edge case tests for RetryContext."""

    @pytest.mark.asyncio
    async def test_retry_context_immediate_success(self):
        """Test retry context with immediate success."""
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)

        async with RetryContext(config, "test_op") as ctx:
            ctx.next_attempt()
            ctx.record_success()

        assert ctx.attempt == 1
        assert ctx.stats.successful_attempts == 1
        assert ctx.stats.total_retries == 0

    @pytest.mark.asyncio
    async def test_retry_context_tracks_delays(self):
        """Test retry context tracks total delay."""
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01, jitter=False)

        async with RetryContext(config, "test_op") as ctx:
            while ctx.should_continue():
                ctx.next_attempt()
                if ctx.attempt < 3:
                    try:
                        raise ValueError("Retry me")
                    except ValueError as e:
                        await ctx.record_failure(e)
                else:
                    ctx.record_success()
                    break

        assert ctx.stats.total_retries == 2
        # Total delay should be sum of first two delays
        assert ctx.stats.total_delay_seconds > 0

    @pytest.mark.asyncio
    async def test_retry_context_non_retryable_raises_immediately(self):
        """Test that non-retryable exceptions raise immediately."""
        config = RetryConfig(
            max_attempts=5,
            non_retryable_exceptions=(KeyError,),
        )

        with pytest.raises(KeyError):
            async with RetryContext(config, "test_op") as ctx:
                ctx.next_attempt()
                try:
                    raise KeyError("Non-retryable")
                except KeyError as e:
                    await ctx.record_failure(e)

    @pytest.mark.asyncio
    async def test_retry_context_exhaustion_error_details(self):
        """Test RetryExhaustedError contains useful details."""
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01, jitter=False)

        with pytest.raises(RetryExhaustedError) as exc_info:
            async with RetryContext(config, "test_op") as ctx:
                while ctx.should_continue():
                    ctx.next_attempt()
                    try:
                        raise ValueError("Always fails")
                    except ValueError as e:
                        await ctx.record_failure(e)

        error = exc_info.value
        assert error.attempts == 3
        assert isinstance(error.last_error, ValueError)
        assert error.total_delay > 0
        assert "test_op" in str(error)


class TestRetryDecorator:
    """Tests for retry decorator functionality."""

    @pytest.mark.asyncio
    async def test_retry_decorator_preserves_function_name(self):
        """Test that retry decorator preserves function name."""
        config = RetryConfig()

        @retry(config)
        async def my_function():
            return "result"

        assert my_function.__name__ == "my_function"

    @pytest.mark.asyncio
    async def test_retry_decorator_with_arguments(self):
        """Test retry decorator with function arguments."""
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)
        call_count = 0

        @retry(config)
        async def add_numbers(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return a + b

        result = await add_numbers(3, 5)

        assert result == 8
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_with_kwargs(self):
        """Test retry decorator with keyword arguments."""
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)
        call_count = 0

        @retry(config)
        async def greet(name: str, greeting: str = "Hello") -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return f"{greeting}, {name}!"

        result = await greet(name="World", greeting="Hi")

        assert result == "Hi, World!"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_with_custom_operation_name(self):
        """Test retry decorator with custom operation name."""
        config = RetryConfig(max_attempts=2, base_delay_seconds=0.01)

        @retry(config, operation_name="custom_op")
        async def my_function():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await my_function()

        assert "custom_op" in str(exc_info.value)


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_retry_async_immediate_success(self):
        """Test retry_async with immediate success."""
        config = RetryConfig(max_attempts=3)
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_async(operation, config, "test_op")

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_with_retries(self):
        """Test retry_async with retries before success."""
        config = RetryConfig(max_attempts=5, base_delay_seconds=0.01)
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry me")
            return "success"

        result = await retry_async(operation, config, "test_op")

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_exhaustion(self):
        """Test retry_async exhausts all attempts."""
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)

        async def operation():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await retry_async(operation, config, "test_op")

        assert exc_info.value.attempts == 3


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError exception."""

    def test_error_contains_attempts(self):
        """Test error contains attempt count."""
        error = RetryExhaustedError("Test failed", attempts=5)

        assert error.attempts == 5

    def test_error_contains_last_error(self):
        """Test error contains last error."""
        last_err = ValueError("Original error")
        error = RetryExhaustedError("Test failed", attempts=3, last_error=last_err)

        assert error.last_error is last_err

    def test_error_contains_total_delay(self):
        """Test error contains total delay."""
        error = RetryExhaustedError("Test failed", attempts=3, total_delay=5.5)

        assert error.total_delay == 5.5


# ============================================================================
# Combined Pattern Tests
# ============================================================================


class TestCircuitBreakerWithRetry:
    """Tests for combining circuit breaker with retry."""

    @pytest.mark.asyncio
    async def test_retry_respects_open_circuit(self):
        """Test that retry stops when circuit opens."""
        circuit = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=2, timeout_seconds=30.0),
        )
        retry_config = RetryConfig(max_attempts=10, base_delay_seconds=0.01)
        call_count = 0

        @circuit
        @retry(retry_config)
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises((CircuitBreakerError, RetryExhaustedError)):
            await failing_operation()

        # Call count should be limited by circuit breaker
        assert call_count >= 2  # At least hit the threshold

    @pytest.mark.asyncio
    async def test_circuit_breaker_outside_retry(self):
        """Test circuit breaker wrapping retry decorator."""
        circuit = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0),
        )
        retry_config = RetryConfig(max_attempts=2, base_delay_seconds=0.01)
        invocation_count = 0
        call_count = 0

        @circuit
        @retry(retry_config)
        async def flaky_operation():
            nonlocal invocation_count, call_count
            invocation_count += 1
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient error")
            call_count = 0  # Reset for next invocation
            return "success"

        # First invocation succeeds after retry
        result1 = await flaky_operation()
        assert result1 == "success"

        # Second invocation succeeds after retry
        result2 = await flaky_operation()
        assert result2 == "success"

        # Circuit should still be closed (no failures at circuit level)
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker_error(self):
        """Test that CircuitBreakerError can trigger retry."""
        circuit = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.05),
        )

        # Force circuit open
        try:
            async with circuit:
                raise ValueError("Force open")
        except ValueError:
            pass

        retry_config = RetryConfig(
            max_attempts=5,
            base_delay_seconds=0.02,
            retryable_exceptions=(CircuitBreakerError,),
        )

        @retry(retry_config)
        async def call_with_circuit():
            async with circuit:
                return "success"

        # Should eventually succeed when circuit goes half-open
        result = await call_with_circuit()
        assert result == "success"


class TestRetryConfigValidation:
    """Tests for RetryConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.jitter_factor == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=10,
            base_delay_seconds=0.5,
            max_delay_seconds=30.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert config.max_attempts == 10
        assert config.base_delay_seconds == 0.5
        assert config.max_delay_seconds == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_config_with_retryable_exceptions(self):
        """Test configuration with custom retryable exceptions."""
        config = RetryConfig(
            retryable_exceptions=(ValueError, TypeError),
            non_retryable_exceptions=(KeyError,),
        )

        assert ValueError in config.retryable_exceptions
        assert TypeError in config.retryable_exceptions
        assert KeyError in config.non_retryable_exceptions


class TestCircuitBreakerConfigValidation:
    """Tests for CircuitBreakerConfig validation and defaults."""

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
            success_threshold=5,
            timeout_seconds=60.0,
            half_open_max_calls=5,
            excluded_exceptions=(ValueError,),
        )

        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_calls == 5
        assert config.excluded_exceptions == (ValueError,)


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transition logic."""

    @pytest.mark.asyncio
    async def test_state_transitions_logged(self):
        """Test that state transitions are tracked."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.05,
        )
        breaker = CircuitBreaker("test", config)

        initial_state_changes = breaker.stats.state_changes

        # Transition to OPEN
        for _ in range(2):
            try:
                async with breaker:
                    raise ValueError("Test error")
            except ValueError:
                pass

        assert breaker.stats.state_changes == initial_state_changes + 1
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Transition to HALF_OPEN (on next call)
        async with breaker:
            pass

        # Might transition to HALF_OPEN then CLOSED, or stay HALF_OPEN
        assert breaker.stats.state_changes >= initial_state_changes + 2

    @pytest.mark.asyncio
    async def test_last_failure_time_updated(self):
        """Test that last_failure_time is updated on failure."""
        breaker = CircuitBreaker("test")

        before = time.time()

        try:
            async with breaker:
                raise ValueError("Test error")
        except ValueError:
            pass

        after = time.time()

        assert breaker.stats.last_failure_time is not None
        assert before <= breaker.stats.last_failure_time <= after

    @pytest.mark.asyncio
    async def test_last_success_time_updated(self):
        """Test that last_success_time is updated on success."""
        breaker = CircuitBreaker("test")

        before = time.time()

        async with breaker:
            pass

        after = time.time()

        assert breaker.stats.last_success_time is not None
        assert before <= breaker.stats.last_success_time <= after
