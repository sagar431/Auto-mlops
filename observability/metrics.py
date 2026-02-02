"""
Prometheus Metrics Module

Provides Prometheus-compatible metrics for monitoring the MLOps agent.
Exports metrics in the Prometheus text format for scraping.
"""

import time
from collections import defaultdict
from enum import Enum
from threading import Lock

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Prometheus metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricLabel(BaseModel):
    """Labels for a metric."""

    name: str
    value: str


class MetricValue(BaseModel):
    """A metric value with optional labels."""

    value: float
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: float | None = None


class Metric(BaseModel):
    """A Prometheus metric."""

    name: str
    metric_type: MetricType
    help_text: str
    values: list[MetricValue] = Field(default_factory=list)


class Counter:
    """
    Prometheus Counter metric.

    Monotonically increasing counter that resets on restart.
    """

    def __init__(self, name: str, help_text: str, label_names: list[str] | None = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = Lock()

    def inc(self, amount: float = 1.0, **labels):
        """Increment the counter."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] += amount

    def _make_label_key(self, labels: dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> Metric:
        """Collect current metric values."""
        values = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key)
                values.append(MetricValue(value=value, labels=labels))
        return Metric(
            name=self.name,
            metric_type=MetricType.COUNTER,
            help_text=self.help_text,
            values=values,
        )


class Gauge:
    """
    Prometheus Gauge metric.

    Can go up and down, represents current state.
    """

    def __init__(self, name: str, help_text: str, label_names: list[str] | None = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = Lock()

    def set(self, value: float, **labels):
        """Set the gauge value."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, **labels):
        """Increment the gauge."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1.0, **labels):
        """Decrement the gauge."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] -= amount

    def _make_label_key(self, labels: dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> Metric:
        """Collect current metric values."""
        values = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key)
                values.append(MetricValue(value=value, labels=labels))
        return Metric(
            name=self.name,
            metric_type=MetricType.GAUGE,
            help_text=self.help_text,
            values=values,
        )


class Histogram:
    """
    Prometheus Histogram metric.

    Tracks distribution of values with configurable buckets.
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))

    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple, dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: dict[tuple, float] = defaultdict(float)
        self._totals: dict[tuple, int] = defaultdict(int)
        self._lock = Lock()

    def observe(self, value: float, **labels):
        """Observe a value."""
        label_key = self._make_label_key(labels)
        with self._lock:
            # Only increment the first bucket that the value fits into
            for bucket in sorted(self.buckets):
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
                    break
            self._sums[label_key] += value
            self._totals[label_key] += 1

    def _make_label_key(self, labels: dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> list[Metric]:
        """Collect histogram metrics (bucket, sum, count)."""
        metrics = []

        with self._lock:
            # Bucket metric
            bucket_values = []
            for label_key, buckets in self._counts.items():
                labels = dict(label_key)
                cumulative = 0
                for bucket, count in sorted(buckets.items()):
                    cumulative += count
                    bucket_labels = {
                        **labels,
                        "le": str(bucket) if bucket != float("inf") else "+Inf",
                    }
                    bucket_values.append(MetricValue(value=cumulative, labels=bucket_labels))

            metrics.append(
                Metric(
                    name=f"{self.name}_bucket",
                    metric_type=MetricType.HISTOGRAM,
                    help_text=self.help_text,
                    values=bucket_values,
                )
            )

            # Sum metric
            sum_values = []
            for label_key, total in self._sums.items():
                labels = dict(label_key)
                sum_values.append(MetricValue(value=total, labels=labels))
            metrics.append(
                Metric(
                    name=f"{self.name}_sum",
                    metric_type=MetricType.HISTOGRAM,
                    help_text=f"Total sum of {self.name}",
                    values=sum_values,
                )
            )

            # Count metric
            count_values = []
            for label_key, count in self._totals.items():
                labels = dict(label_key)
                count_values.append(MetricValue(value=count, labels=labels))
            metrics.append(
                Metric(
                    name=f"{self.name}_count",
                    metric_type=MetricType.HISTOGRAM,
                    help_text=f"Total count of {self.name}",
                    values=count_values,
                )
            )

        return metrics


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, **labels):
        self.histogram = histogram
        self.labels = labels
        self.start_time: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.histogram.observe(duration, **self.labels)


class PrometheusRegistry:
    """
    Registry for Prometheus metrics.

    Collects all registered metrics and exports them in Prometheus text format.
    """

    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = Lock()

    def register(self, metric: Counter | Gauge | Histogram):
        """Register a metric."""
        with self._lock:
            self._metrics[metric.name] = metric

    def unregister(self, name: str):
        """Unregister a metric."""
        with self._lock:
            self._metrics.pop(name, None)

    def collect_all(self) -> list[Metric]:
        """Collect all metrics."""
        all_metrics = []
        with self._lock:
            for metric in self._metrics.values():
                if isinstance(metric, Histogram):
                    all_metrics.extend(metric.collect())
                else:
                    all_metrics.append(metric.collect())
        return all_metrics

    def export_text(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        metrics = self.collect_all()

        for metric in metrics:
            # HELP line
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            # TYPE line
            lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")

            # Values
            for value in metric.values:
                if value.labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(value.labels.items()))
                    lines.append(f"{metric.name}{{{label_str}}} {value.value}")
                else:
                    lines.append(f"{metric.name} {value.value}")

        return "\n".join(lines) + "\n"


# Global registry
_registry = PrometheusRegistry()


def get_registry() -> PrometheusRegistry:
    """Get the global Prometheus registry."""
    return _registry


# Pre-defined metrics for MLOps agent
class MLOpsMetrics:
    """
    Pre-defined Prometheus metrics for the MLOps agent.

    Usage:
        from observability.metrics import mlops_metrics

        # Record session start
        mlops_metrics.sessions_total.inc(status="started")

        # Record tool invocation
        with mlops_metrics.tool_duration.time(tool="create_hydra_config"):
            # ... execute tool ...
            pass

        # Or manually:
        mlops_metrics.tool_invocations.inc(tool="init_mlflow", status="success")
    """

    def __init__(self, registry: PrometheusRegistry | None = None):
        self.registry = registry or _registry

        # Session metrics
        self.sessions_total = Counter(
            name="mlops_sessions_total",
            help_text="Total number of agent sessions",
            label_names=["status"],
        )
        self.sessions_active = Gauge(
            name="mlops_sessions_active",
            help_text="Number of currently active sessions",
        )
        self.session_duration = Histogram(
            name="mlops_session_duration_seconds",
            help_text="Duration of agent sessions in seconds",
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, float("inf")),
        )

        # Tool metrics
        self.tool_invocations = Counter(
            name="mlops_tool_invocations_total",
            help_text="Total tool invocations",
            label_names=["tool", "status"],
        )
        self.tool_duration = Histogram(
            name="mlops_tool_duration_seconds",
            help_text="Duration of tool executions in seconds",
            label_names=["tool"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, float("inf")),
        )

        # Pipeline metrics
        self.pipelines_total = Counter(
            name="mlops_pipelines_total",
            help_text="Total pipelines executed",
            label_names=["status"],
        )
        self.pipeline_steps = Histogram(
            name="mlops_pipeline_steps",
            help_text="Number of steps per pipeline",
            buckets=(1, 2, 5, 10, 15, 20, 30, 50, float("inf")),
        )

        # LLM metrics
        self.llm_requests = Counter(
            name="mlops_llm_requests_total",
            help_text="Total LLM API requests",
            label_names=["provider", "status"],
        )
        self.llm_latency = Histogram(
            name="mlops_llm_latency_seconds",
            help_text="LLM API request latency in seconds",
            label_names=["provider"],
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, float("inf")),
        )
        self.llm_tokens = Counter(
            name="mlops_llm_tokens_total",
            help_text="Total LLM tokens used",
            label_names=["provider", "type"],
        )

        # Training metrics
        self.training_runs = Counter(
            name="mlops_training_runs_total",
            help_text="Total training runs",
            label_names=["status"],
        )
        self.accuracy_current = Gauge(
            name="mlops_accuracy_current",
            help_text="Current model accuracy",
            label_names=["session_id"],
        )
        self.improvement_attempts = Counter(
            name="mlops_improvement_attempts_total",
            help_text="Total improvement loop attempts",
            label_names=["result"],
        )

        # Error metrics
        self.errors_total = Counter(
            name="mlops_errors_total",
            help_text="Total errors encountered",
            label_names=["component", "error_type"],
        )

        # Register all metrics
        self._register_all()

    def _register_all(self):
        """Register all metrics with the registry."""
        for name, metric in vars(self).items():
            if isinstance(metric, (Counter, Gauge, Histogram)):
                self.registry.register(metric)


# Global instance
mlops_metrics = MLOpsMetrics()


def get_metrics_endpoint() -> str:
    """
    Get the Prometheus metrics endpoint output.

    Returns:
        Prometheus text format metrics

    Example:
        from fastapi import Response
        from observability.metrics import get_metrics_endpoint

        @app.get("/metrics")
        def metrics():
            return Response(
                content=get_metrics_endpoint(),
                media_type="text/plain; charset=utf-8"
            )
    """
    return _registry.export_text()
