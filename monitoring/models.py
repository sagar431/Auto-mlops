"""
Pydantic models for the monitoring module.

Defines data models for drift detection, model performance monitoring,
and alerting functionality.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DriftType(str, Enum):
    """Types of data drift."""

    DATA = "data"
    CONCEPT = "concept"
    PREDICTION = "prediction"
    FEATURE = "feature"


class DriftSeverity(str, Enum):
    """Severity level of detected drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Supported alert notification channels."""

    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    LOG = "log"


class FeatureDriftResult(BaseModel):
    """Drift detection result for a single feature."""

    feature_name: str = Field(..., description="Name of the feature")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(
        ..., ge=0.0, le=1.0, description="Drift score (0-1, higher = more drift)"
    )
    stattest_name: str = Field(..., description="Name of statistical test used")
    stattest_threshold: float = Field(..., description="Threshold used for detection")
    p_value: float | None = Field(None, description="P-value from statistical test")
    reference_distribution: dict[str, Any] | None = Field(
        None, description="Summary of reference distribution"
    )
    current_distribution: dict[str, Any] | None = Field(
        None, description="Summary of current distribution"
    )


class DriftReport(BaseModel):
    """Complete drift detection report."""

    report_id: str = Field(..., description="Unique identifier for this report")
    dataset_name: str = Field(..., description="Name of the dataset analyzed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the report was generated"
    )
    drift_type: DriftType = Field(..., description="Type of drift detected")
    overall_drift_detected: bool = Field(..., description="Whether overall drift was detected")
    drift_share: float = Field(
        ..., ge=0.0, le=1.0, description="Share of features with drift (0-1)"
    )
    severity: DriftSeverity = Field(..., description="Overall severity of drift")
    feature_results: list[FeatureDriftResult] = Field(
        default_factory=list, description="Per-feature drift results"
    )
    reference_rows: int = Field(..., description="Number of rows in reference dataset")
    current_rows: int = Field(..., description="Number of rows in current dataset")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations based on drift analysis"
    )


class ModelMetrics(BaseModel):
    """Performance metrics for a model."""

    accuracy: float | None = Field(None, ge=0.0, le=1.0, description="Classification accuracy")
    precision: float | None = Field(None, ge=0.0, le=1.0, description="Precision score")
    recall: float | None = Field(None, ge=0.0, le=1.0, description="Recall score")
    f1_score: float | None = Field(None, ge=0.0, le=1.0, description="F1 score")
    auc_roc: float | None = Field(None, ge=0.0, le=1.0, description="Area under ROC curve")
    log_loss: float | None = Field(None, ge=0.0, description="Log loss (cross-entropy)")
    mse: float | None = Field(None, ge=0.0, description="Mean squared error")
    rmse: float | None = Field(None, ge=0.0, description="Root mean squared error")
    mae: float | None = Field(None, ge=0.0, description="Mean absolute error")
    r2_score: float | None = Field(None, description="R-squared score")
    custom_metrics: dict[str, float] = Field(
        default_factory=dict, description="Additional custom metrics"
    )


class PerformanceSnapshot(BaseModel):
    """A snapshot of model performance at a point in time."""

    snapshot_id: str = Field(..., description="Unique identifier for this snapshot")
    model_name: str = Field(..., description="Name of the model")
    model_version: str | None = Field(None, description="Version of the model")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When snapshot was taken"
    )
    metrics: ModelMetrics = Field(..., description="Performance metrics")
    sample_size: int = Field(..., ge=0, description="Number of samples evaluated")
    data_start: datetime | None = Field(None, description="Start of evaluation period")
    data_end: datetime | None = Field(None, description="End of evaluation period")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PerformanceTrend(BaseModel):
    """Trend analysis of model performance over time."""

    model_name: str = Field(..., description="Name of the model")
    metric_name: str = Field(..., description="Name of the metric being tracked")
    time_window_days: int = Field(..., description="Number of days in the window")
    snapshots: list[PerformanceSnapshot] = Field(
        default_factory=list, description="Historical snapshots"
    )
    trend_direction: str = Field(..., description="Trend direction: improving, declining, stable")
    change_percentage: float = Field(..., description="Percentage change over the window")
    baseline_value: float | None = Field(None, description="Baseline metric value")
    current_value: float | None = Field(None, description="Most recent metric value")
    degradation_detected: bool = Field(..., description="Whether significant degradation detected")
    degradation_threshold: float = Field(
        default=0.05, description="Threshold for degradation detection"
    )


class AlertConfig(BaseModel):
    """Configuration for an alert."""

    alert_id: str = Field(..., description="Unique identifier for this alert config")
    name: str = Field(..., description="Human-readable name")
    description: str | None = Field(None, description="Alert description")
    enabled: bool = Field(default=True, description="Whether alert is active")
    channel: AlertChannel = Field(..., description="Notification channel")
    level: AlertLevel = Field(..., description="Alert severity level")
    metric_name: str | None = Field(None, description="Metric to monitor (for metric alerts)")
    threshold: float | None = Field(None, description="Threshold value for triggering")
    comparison: str = Field(
        default="lt", description="Comparison operator: lt, gt, lte, gte, eq, neq"
    )
    cooldown_minutes: int = Field(default=60, description="Minimum minutes between alerts")
    webhook_url: str | None = Field(None, description="Webhook URL for notifications")
    email_recipients: list[str] = Field(
        default_factory=list, description="Email addresses for notifications"
    )
    slack_channel: str | None = Field(None, description="Slack channel for notifications")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional config")


class Alert(BaseModel):
    """An alert instance that was triggered."""

    alert_id: str = Field(..., description="Unique identifier for this alert instance")
    config_id: str = Field(..., description="ID of the alert configuration")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When alert was triggered"
    )
    level: AlertLevel = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Detailed alert message")
    metric_name: str | None = Field(None, description="Metric that triggered alert")
    metric_value: float | None = Field(None, description="Value that triggered alert")
    threshold_value: float | None = Field(None, description="Threshold that was breached")
    model_name: str | None = Field(None, description="Model associated with alert")
    resolved: bool = Field(default=False, description="Whether alert has been resolved")
    resolved_at: datetime | None = Field(None, description="When alert was resolved")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")
    acknowledged_by: str | None = Field(None, description="Who acknowledged the alert")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class MonitoringConfig(BaseModel):
    """Configuration for the monitoring system."""

    model_name: str = Field(..., description="Name of the model to monitor")
    model_version: str | None = Field(None, description="Version of the model")
    reference_data_path: str | None = Field(None, description="Path to reference/baseline data")
    drift_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Drift score threshold for alerts"
    )
    degradation_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Performance degradation threshold"
    )
    check_interval_minutes: int = Field(
        default=60, description="How often to run monitoring checks"
    )
    enabled_checks: list[str] = Field(
        default_factory=lambda: ["drift", "performance"],
        description="Enabled monitoring checks",
    )
    alert_configs: list[AlertConfig] = Field(
        default_factory=list, description="Alert configurations"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional config")
