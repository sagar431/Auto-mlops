"""
Observability Module for MLOps Agent

Provides structured logging, Prometheus metrics, and monitoring capabilities.

Components:
- logging: Structured JSON logging with context binding
- metrics: Prometheus-compatible metrics for monitoring

Usage:
    # Logging
    from observability import get_logger, configure_logging

    # Configure for development
    configure_logging(level="debug", json_output=False)

    # Configure for production
    configure_logging(level="info", json_output=True)

    # Get a logger
    logger = get_logger("agent.agent_loop")
    logger.bind(session_id="abc123")
    logger.info("Session started", query="Train model")

    # Metrics
    from observability import mlops_metrics, get_metrics_endpoint

    # Record events
    mlops_metrics.sessions_total.inc(status="started")
    mlops_metrics.tool_invocations.inc(tool="create_hydra_config", status="success")

    # Get Prometheus endpoint output
    metrics_text = get_metrics_endpoint()
"""

from .logging import (
    LogContext,
    LoggerFactory,
    LogLevel,
    StructuredLogEntry,
    StructuredLogger,
    configure_logging,
    get_logger,
)
from .metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricType,
    MetricValue,
    MLOpsMetrics,
    PrometheusRegistry,
    Timer,
    get_metrics_endpoint,
    get_registry,
    mlops_metrics,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "LogLevel",
    "LogContext",
    "LoggerFactory",
    "StructuredLogger",
    "StructuredLogEntry",
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "Metric",
    "MetricType",
    "MetricValue",
    "PrometheusRegistry",
    "MLOpsMetrics",
    "get_registry",
    "get_metrics_endpoint",
    "mlops_metrics",
]
