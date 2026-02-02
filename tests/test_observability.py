#!/usr/bin/env python3
"""
Tests for observability module - structured logging and Prometheus metrics.

Run with: pytest tests/test_observability.py -v
"""

import json

import pytest

from observability import (
    Counter,
    Gauge,
    Histogram,
    LoggerFactory,
    LogLevel,
    MLOpsMetrics,
    PrometheusRegistry,
    StructuredLogger,
    Timer,
    configure_logging,
    get_logger,
    get_metrics_endpoint,
    get_registry,
    mlops_metrics,
)
from observability.logging import (
    clear_log_context,
    get_log_context,
    get_structlog_logger,
    is_configured,
    set_log_context,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_logger_factory():
    """Reset the logger factory between tests."""
    LoggerFactory.clear()
    clear_log_context()
    configure_logging(level="info", json_output=True)
    yield
    LoggerFactory.clear()
    clear_log_context()


@pytest.fixture
def clean_registry():
    """Create a clean Prometheus registry for testing."""
    return PrometheusRegistry()


# ============================================================================
# Structured Logging Tests
# ============================================================================


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_create_logger(self):
        """Test creating a structured logger."""
        logger = StructuredLogger(name="test.module", level=LogLevel.DEBUG, json_output=True)

        assert logger.name == "test.module"
        assert logger._level == LogLevel.DEBUG
        assert logger._json_output is True

    def test_logger_bind_context(self):
        """Test binding context to logger."""
        logger = StructuredLogger(name="test", json_output=False)
        logger.bind(session_id="sess-123", step_id="step-1")

        assert logger._bound_values.get("session_id") == "sess-123"
        assert logger._bound_values.get("step_id") == "step-1"

    def test_logger_bind_chaining(self):
        """Test that bind returns self for chaining."""
        logger = StructuredLogger(name="test", json_output=False)
        result = logger.bind(session_id="sess-123").bind(tool_name="create_hydra_config")

        assert result is logger
        assert logger._bound_values.get("session_id") == "sess-123"
        assert logger._bound_values.get("tool_name") == "create_hydra_config"

    def test_logger_unbind(self):
        """Test unbinding context from logger."""
        logger = StructuredLogger(name="test", json_output=False)
        logger.bind(session_id="sess-123", step_id="step-1")
        logger.unbind("session_id")

        assert logger._bound_values.get("session_id") is None
        assert logger._bound_values.get("step_id") == "step-1"

    def test_logger_clear_context(self):
        """Test clearing all context."""
        logger = StructuredLogger(name="test", json_output=False)
        logger.bind(session_id="sess-123", step_id="step-1", phase="perception")
        logger.clear_context()

        assert logger._bound_values.get("session_id") is None
        assert logger._bound_values.get("step_id") is None
        assert logger._bound_values.get("phase") is None

    def test_logger_json_output(self, capsys):
        """Test that logger outputs valid JSON."""
        configure_logging(level="info", json_output=True)
        LoggerFactory.clear()
        logger = StructuredLogger(name="test.json", level=LogLevel.INFO, json_output=True)
        logger.bind(session_id="sess-456")
        logger.info("Test message")

        captured = capsys.readouterr()
        log_entry = json.loads(captured.out.strip())

        assert log_entry["level"] == "info"
        assert log_entry["message"] == "Test message"
        assert log_entry["session_id"] == "sess-456"
        assert "timestamp" in log_entry

    def test_logger_json_with_extra(self, capsys):
        """Test JSON output with extra fields."""
        configure_logging(level="info", json_output=True)
        LoggerFactory.clear()
        logger = StructuredLogger(name="test.extra", level=LogLevel.INFO, json_output=True)
        logger.info("Tool completed", duration_ms=150.5, tool="create_hydra")

        captured = capsys.readouterr()
        log_entry = json.loads(captured.out.strip())

        assert log_entry["duration_ms"] == 150.5
        assert log_entry["tool"] == "create_hydra"

    def test_logger_error_with_error_field(self, capsys):
        """Test error logging includes error field."""
        configure_logging(level="info", json_output=True)
        LoggerFactory.clear()
        logger = StructuredLogger(name="test.error", level=LogLevel.INFO, json_output=True)
        logger.error("Tool failed", error="Connection timeout")

        captured = capsys.readouterr()
        log_entry = json.loads(captured.out.strip())

        assert log_entry["level"] == "error"
        assert log_entry["error"] == "Connection timeout"

    def test_logger_error_with_exception(self, capsys):
        """Test error logging with exception object."""
        configure_logging(level="info", json_output=True)
        LoggerFactory.clear()
        logger = StructuredLogger(name="test.exc", level=LogLevel.INFO, json_output=True)
        try:
            raise ValueError("Something went wrong")
        except ValueError as e:
            logger.error("Operation failed", error=e)

        captured = capsys.readouterr()
        log_entry = json.loads(captured.out.strip())

        assert log_entry["level"] == "error"
        assert log_entry["error"] == "Something went wrong"
        assert log_entry["error_type"] == "ValueError"

    def test_logger_new_creates_copy(self):
        """Test that new() creates a copy with additional bindings."""
        logger = StructuredLogger(name="test", json_output=True)
        logger.bind(session_id="sess-123")

        new_logger = logger.new(step_id="step-1")

        assert new_logger is not logger
        assert new_logger._bound_values.get("session_id") == "sess-123"
        assert new_logger._bound_values.get("step_id") == "step-1"
        # Original logger should not have step_id
        assert logger._bound_values.get("step_id") is None


class TestLoggerFactory:
    """Tests for LoggerFactory class."""

    def test_get_logger_creates_new(self):
        """Test getting a new logger creates it."""
        logger = LoggerFactory.get_logger("new.module")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "new.module"

    def test_get_logger_returns_same_instance(self):
        """Test getting the same logger returns same instance."""
        logger1 = LoggerFactory.get_logger("same.module")
        logger2 = LoggerFactory.get_logger("same.module")
        assert logger1 is logger2

    def test_configure_changes_defaults(self):
        """Test that configure changes default settings."""
        LoggerFactory.configure(level=LogLevel.DEBUG, json_output=False)

        logger = LoggerFactory.get_logger("configured.module")
        assert logger._level == LogLevel.DEBUG
        assert logger._json_output is False


class TestGetLogger:
    """Tests for get_logger convenience function."""

    def test_get_logger_function(self):
        """Test the get_logger convenience function."""
        logger = get_logger("my.module")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "my.module"


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_level(self):
        """Test configuring log level."""
        LoggerFactory.clear()
        configure_logging(level="debug")
        logger = get_logger("debug.test")
        assert logger._level == LogLevel.DEBUG

    def test_configure_logging_json_output(self):
        """Test configuring JSON output."""
        LoggerFactory.clear()
        configure_logging(json_output=False)
        logger = get_logger("non_json.test")
        assert logger._json_output is False

    def test_is_configured(self):
        """Test is_configured returns True after configuration."""
        configure_logging(level="info", json_output=True)
        assert is_configured() is True


class TestLogContext:
    """Tests for log context functions."""

    def test_set_and_get_context(self):
        """Test setting and getting log context."""
        clear_log_context()
        set_log_context(session_id="sess-123", user="test_user")

        ctx = get_log_context()
        assert ctx["session_id"] == "sess-123"
        assert ctx["user"] == "test_user"

    def test_clear_context(self):
        """Test clearing log context."""
        set_log_context(session_id="sess-123")
        clear_log_context()

        ctx = get_log_context()
        assert ctx == {}

    def test_context_is_isolated(self):
        """Test that get_log_context returns a copy."""
        set_log_context(session_id="sess-123")
        ctx = get_log_context()
        ctx["new_key"] = "new_value"

        # Original context should not be modified
        assert get_log_context().get("new_key") is None


class TestStructlogLogger:
    """Tests for get_structlog_logger function."""

    def test_get_structlog_logger(self):
        """Test getting a raw structlog logger."""
        logger = get_structlog_logger("raw.structlog")
        assert logger is not None
        # Verify it has expected methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "bind")


# ============================================================================
# Prometheus Metrics Tests
# ============================================================================


class TestCounter:
    """Tests for Prometheus Counter metric."""

    def test_counter_creation(self):
        """Test creating a counter."""
        counter = Counter(
            name="test_counter",
            help_text="A test counter",
            label_names=["status"],
        )
        assert counter.name == "test_counter"
        assert counter.help_text == "A test counter"
        assert counter.label_names == ["status"]

    def test_counter_increment(self):
        """Test incrementing counter."""
        counter = Counter(name="test_counter", help_text="Test")
        counter.inc()
        counter.inc(amount=5)

        metric = counter.collect()
        assert len(metric.values) == 1
        assert metric.values[0].value == 6.0

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter(name="test_counter", help_text="Test", label_names=["status"])
        counter.inc(status="success")
        counter.inc(status="success")
        counter.inc(status="failure")

        metric = counter.collect()
        assert len(metric.values) == 2

        values_by_status = {tuple(v.labels.items()): v.value for v in metric.values}
        assert values_by_status[(("status", "success"),)] == 2.0
        assert values_by_status[(("status", "failure"),)] == 1.0


class TestGauge:
    """Tests for Prometheus Gauge metric."""

    def test_gauge_creation(self):
        """Test creating a gauge."""
        gauge = Gauge(name="test_gauge", help_text="A test gauge")
        assert gauge.name == "test_gauge"

    def test_gauge_set(self):
        """Test setting gauge value."""
        gauge = Gauge(name="test_gauge", help_text="Test")
        gauge.set(42.5)

        metric = gauge.collect()
        assert metric.values[0].value == 42.5

    def test_gauge_inc_dec(self):
        """Test incrementing and decrementing gauge."""
        gauge = Gauge(name="test_gauge", help_text="Test")
        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)

        metric = gauge.collect()
        assert metric.values[0].value == 12.0

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge(name="test_gauge", help_text="Test", label_names=["session_id"])
        gauge.set(0.85, session_id="sess-1")
        gauge.set(0.92, session_id="sess-2")

        metric = gauge.collect()
        assert len(metric.values) == 2


class TestHistogram:
    """Tests for Prometheus Histogram metric."""

    def test_histogram_creation(self):
        """Test creating a histogram."""
        histogram = Histogram(
            name="test_histogram",
            help_text="A test histogram",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )
        assert histogram.name == "test_histogram"
        assert histogram.buckets == (0.1, 0.5, 1.0, float("inf"))

    def test_histogram_observe(self):
        """Test observing values in histogram."""
        histogram = Histogram(
            name="test_histogram",
            help_text="Test",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.7)
        histogram.observe(5.0)

        metrics = histogram.collect()
        assert len(metrics) == 3  # bucket, sum, count

        # Find the count metric
        count_metric = next(m for m in metrics if m.name.endswith("_count"))
        assert count_metric.values[0].value == 4

        # Find the sum metric
        sum_metric = next(m for m in metrics if m.name.endswith("_sum"))
        assert sum_metric.values[0].value == pytest.approx(6.05)

    def test_histogram_buckets_cumulative(self):
        """Test that histogram buckets are cumulative."""
        histogram = Histogram(
            name="test_histogram",
            help_text="Test",
            buckets=(1.0, 5.0, 10.0, float("inf")),
        )
        histogram.observe(0.5)  # In bucket 1.0
        histogram.observe(3.0)  # In bucket 5.0
        histogram.observe(7.0)  # In bucket 10.0

        metrics = histogram.collect()
        bucket_metric = next(m for m in metrics if m.name.endswith("_bucket"))

        # Check cumulative counts
        bucket_values = {v.labels.get("le"): v.value for v in bucket_metric.values}
        assert bucket_values["1.0"] == 1  # 1 value <= 1.0
        assert bucket_values["5.0"] == 2  # 2 values <= 5.0
        assert bucket_values["10.0"] == 3  # 3 values <= 10.0
        assert bucket_values["+Inf"] == 3  # 3 values total


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        histogram = Histogram(name="test_timer", help_text="Test", buckets=(0.1, 1.0, float("inf")))

        with Timer(histogram):
            pass  # Minimal execution time

        metrics = histogram.collect()
        count_metric = next(m for m in metrics if m.name.endswith("_count"))
        assert count_metric.values[0].value == 1

    def test_timer_with_labels(self):
        """Test timer with labels."""
        histogram = Histogram(
            name="test_timer",
            help_text="Test",
            label_names=["operation"],
            buckets=(0.1, 1.0, float("inf")),
        )

        with Timer(histogram, operation="test_op"):
            pass

        metrics = histogram.collect()
        count_metric = next(m for m in metrics if m.name.endswith("_count"))
        assert count_metric.values[0].labels.get("operation") == "test_op"


class TestPrometheusRegistry:
    """Tests for PrometheusRegistry class."""

    def test_registry_register(self, clean_registry):
        """Test registering metrics."""
        counter = Counter(name="test_counter", help_text="Test")
        clean_registry.register(counter)

        metrics = clean_registry.collect_all()
        assert len(metrics) == 1
        assert metrics[0].name == "test_counter"

    def test_registry_unregister(self, clean_registry):
        """Test unregistering metrics."""
        counter = Counter(name="test_counter", help_text="Test")
        clean_registry.register(counter)
        clean_registry.unregister("test_counter")

        metrics = clean_registry.collect_all()
        assert len(metrics) == 0

    def test_registry_export_text(self, clean_registry):
        """Test Prometheus text format export."""
        counter = Counter(name="test_requests_total", help_text="Total requests")
        counter.inc(amount=42)
        clean_registry.register(counter)

        output = clean_registry.export_text()

        assert "# HELP test_requests_total Total requests" in output
        assert "# TYPE test_requests_total counter" in output
        assert "test_requests_total 42" in output

    def test_registry_export_with_labels(self, clean_registry):
        """Test Prometheus text export with labels."""
        counter = Counter(name="test_counter", help_text="Test", label_names=["method", "status"])
        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="500")
        clean_registry.register(counter)

        output = clean_registry.export_text()

        # Labels should be in alphabetical order
        assert 'test_counter{method="GET",status="200"} 1' in output
        assert 'test_counter{method="POST",status="500"} 1' in output


class TestMLOpsMetrics:
    """Tests for pre-defined MLOps metrics."""

    def test_mlops_metrics_creation(self, clean_registry):
        """Test creating MLOps metrics."""
        metrics = MLOpsMetrics(registry=clean_registry)

        assert metrics.sessions_total is not None
        assert metrics.sessions_active is not None
        assert metrics.tool_invocations is not None
        assert metrics.tool_duration is not None

    def test_mlops_metrics_sessions(self, clean_registry):
        """Test session metrics."""
        metrics = MLOpsMetrics(registry=clean_registry)

        metrics.sessions_total.inc(status="started")
        metrics.sessions_total.inc(status="completed")
        metrics.sessions_active.set(5)

        all_metrics = clean_registry.collect_all()
        metric_names = [m.name for m in all_metrics]

        assert "mlops_sessions_total" in metric_names
        assert "mlops_sessions_active" in metric_names

    def test_mlops_metrics_tools(self, clean_registry):
        """Test tool metrics."""
        metrics = MLOpsMetrics(registry=clean_registry)

        metrics.tool_invocations.inc(tool="create_hydra_config", status="success")
        metrics.tool_invocations.inc(tool="init_mlflow", status="success")
        metrics.tool_invocations.inc(tool="init_mlflow", status="failure")

        collected = metrics.tool_invocations.collect()
        assert len(collected.values) == 3


class TestGetMetricsEndpoint:
    """Tests for get_metrics_endpoint function."""

    def test_get_metrics_endpoint_returns_string(self):
        """Test that get_metrics_endpoint returns a string."""
        output = get_metrics_endpoint()
        assert isinstance(output, str)
        assert output.endswith("\n")

    def test_get_metrics_endpoint_includes_mlops_metrics(self):
        """Test that endpoint includes MLOps metrics."""
        # Use the global metrics instance
        mlops_metrics.sessions_total.inc(status="test")

        output = get_metrics_endpoint()
        assert "mlops_sessions_total" in output


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_get_registry_returns_registry(self):
        """Test that get_registry returns the global registry."""
        registry = get_registry()
        assert isinstance(registry, PrometheusRegistry)

    def test_get_registry_is_singleton(self):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
