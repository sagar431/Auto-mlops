#!/usr/bin/env python3
"""
Tests for StepExecutionTracker with circuit breaker integration.

Run with: pytest tests/test_step_execution_tracker.py -v
"""

import asyncio

import pytest

from agent.agent_loop import StepExecutionError, StepExecutionTracker
from resilience import CircuitBreakerConfig, CircuitState


class TestStepExecutionTracker:
    """Tests for StepExecutionTracker class."""

    def test_create_tracker_with_defaults(self):
        """Test creating tracker with default values."""
        tracker = StepExecutionTracker()
        assert tracker.max_steps == 15
        assert tracker.max_retries == 3
        assert tracker.tries == 0
        assert tracker.attempts == {}

    def test_create_tracker_with_custom_values(self):
        """Test creating tracker with custom values."""
        tracker = StepExecutionTracker(max_steps=10, max_retries=5)
        assert tracker.max_steps == 10
        assert tracker.max_retries == 5

    def test_create_tracker_with_custom_circuit_breaker_config(self):
        """Test creating tracker with custom circuit breaker config."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_seconds=60.0,
        )
        tracker = StepExecutionTracker(circuit_breaker_config=config)
        assert tracker.circuit_breaker.config.failure_threshold == 5
        assert tracker.circuit_breaker.config.success_threshold == 3
        assert tracker.circuit_breaker.config.timeout_seconds == 60.0

    def test_increment(self):
        """Test incrementing tries counter."""
        tracker = StepExecutionTracker()
        assert tracker.tries == 0
        tracker.increment()
        assert tracker.tries == 1
        tracker.increment()
        assert tracker.tries == 2

    def test_record_failure(self):
        """Test recording step failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 1
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 2
        tracker.record_failure("step_2")
        assert tracker.attempts["step_2"] == 1

    def test_retry_step_id(self):
        """Test retry step ID generation."""
        tracker = StepExecutionTracker()
        assert tracker.retry_step_id("step_1") == "step_1"
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F1"
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F2"

    def test_should_continue(self):
        """Test should_continue check."""
        tracker = StepExecutionTracker(max_steps=3)
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is False

    def test_has_exceeded_retries(self):
        """Test has_exceeded_retries check."""
        tracker = StepExecutionTracker(max_retries=2)
        assert tracker.has_exceeded_retries("step_1") is False
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is False
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is True


class TestStepExecutionTrackerCircuitBreaker:
    """Tests for StepExecutionTracker circuit breaker integration."""

    def test_circuit_breaker_property(self):
        """Test circuit breaker property access."""
        tracker = StepExecutionTracker()
        assert tracker.circuit_breaker is not None
        assert tracker.circuit_breaker.name == "step_execution"

    def test_is_circuit_open_initially_false(self):
        """Test circuit is initially closed."""
        tracker = StepExecutionTracker()
        assert tracker.is_circuit_open() is False
        assert tracker.circuit_breaker.state == CircuitState.CLOSED

    def test_get_circuit_stats(self):
        """Test getting circuit stats."""
        tracker = StepExecutionTracker()
        stats = tracker.get_circuit_stats()
        assert "state" in stats
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "failed_calls" in stats
        assert "rejected_calls" in stats
        assert "consecutive_failures" in stats
        assert "consecutive_successes" in stats
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0

    def test_reset_circuit(self):
        """Test resetting circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        tracker = StepExecutionTracker(circuit_breaker_config=config)

        # Manually update stats
        tracker.circuit_breaker._stats.total_calls = 100
        tracker.circuit_breaker._stats.failed_calls = 50

        tracker.reset_circuit()

        stats = tracker.get_circuit_stats()
        assert stats["total_calls"] == 0
        assert stats["failed_calls"] == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after consecutive failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
        )
        tracker = StepExecutionTracker(circuit_breaker_config=config)

        # Simulate failures
        for _ in range(3):
            try:
                async with tracker.circuit_breaker:
                    raise ValueError("Test failure")
            except ValueError:
                pass

        assert tracker.is_circuit_open() is True
        assert tracker.circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_stays_closed_with_successes(self):
        """Test circuit stays closed with successful calls."""
        tracker = StepExecutionTracker()

        for _ in range(5):
            async with tracker.circuit_breaker:
                pass  # Successful call

        assert tracker.is_circuit_open() is False
        stats = tracker.get_circuit_stats()
        assert stats["successful_calls"] == 5
        assert stats["consecutive_successes"] == 5

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
        )
        tracker = StepExecutionTracker(circuit_breaker_config=config)

        # Force circuit open
        for _ in range(2):
            try:
                async with tracker.circuit_breaker:
                    raise ValueError("Test failure")
            except ValueError:
                pass

        assert tracker.is_circuit_open() is True

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open
        async with tracker.circuit_breaker:
            pass

        # After success, should be half-open or closed
        state = tracker.circuit_breaker.state
        assert state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)


class TestStepExecutionError:
    """Tests for StepExecutionError exception."""

    def test_create_step_execution_error(self):
        """Test creating StepExecutionError."""
        error = StepExecutionError("step_1", "Something went wrong")
        assert error.step_id == "step_1"
        assert error.error_message == "Something went wrong"
        assert "step_1" in str(error)
        assert "Something went wrong" in str(error)

    def test_step_execution_error_inheritance(self):
        """Test StepExecutionError is an Exception."""
        error = StepExecutionError("step_1", "Error")
        assert isinstance(error, Exception)


class TestStepExecutionTrackerIntegration:
    """Integration tests for StepExecutionTracker."""

    @pytest.mark.asyncio
    async def test_tracker_with_mixed_results(self):
        """Test tracker handles mixed success/failure results."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=30.0,
        )
        tracker = StepExecutionTracker(max_steps=10, max_retries=3, circuit_breaker_config=config)

        # Simulate some successes
        for _ in range(3):
            async with tracker.circuit_breaker:
                pass

        # Simulate some failures
        for _ in range(2):
            try:
                async with tracker.circuit_breaker:
                    raise ValueError("Test failure")
            except ValueError:
                pass

        assert tracker.is_circuit_open() is False
        stats = tracker.get_circuit_stats()
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 2

    @pytest.mark.asyncio
    async def test_tracker_rejects_calls_when_open(self):
        """Test tracker rejects calls when circuit is open."""
        from resilience import CircuitBreakerError

        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=60.0,
        )
        tracker = StepExecutionTracker(circuit_breaker_config=config)

        # Force circuit open
        for _ in range(2):
            try:
                async with tracker.circuit_breaker:
                    raise ValueError("Test failure")
            except ValueError:
                pass

        assert tracker.is_circuit_open() is True

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError):
            async with tracker.circuit_breaker:
                pass

        stats = tracker.get_circuit_stats()
        assert stats["rejected_calls"] == 1
