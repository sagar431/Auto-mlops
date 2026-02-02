"""
Model Performance Monitor.

Provides continuous monitoring of model performance metrics, trend analysis,
and degradation detection for ML models in production.
"""

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .models import (
    Alert,
    AlertConfig,
    AlertLevel,
    ModelMetrics,
    PerformanceSnapshot,
    PerformanceTrend,
)


class HealthStatus(str, Enum):
    """Overall health status of the model."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ModelMonitor:
    """
    Monitor model performance over time.

    Features:
    - Track multiple performance metrics (accuracy, precision, recall, F1, etc.)
    - Detect performance degradation
    - Generate performance trends and reports
    - Support for classification and regression models

    Example usage:
        from monitoring import ModelMonitor

        monitor = ModelMonitor(model_name="fraud_detector")

        # Record a performance snapshot
        metrics = monitor.calculate_metrics(
            y_true=actual_labels,
            y_pred=predictions,
            task_type="classification"
        )
        monitor.record_snapshot(metrics, sample_size=len(y_true))

        # Check for degradation
        trend = monitor.get_performance_trend("accuracy", days=7)
        if trend.degradation_detected:
            print(f"Performance degraded by {trend.change_percentage:.2%}")
    """

    def __init__(
        self,
        model_name: str,
        model_version: str | None = None,
        degradation_threshold: float = 0.05,
        baseline_metrics: ModelMetrics | None = None,
    ):
        """
        Initialize the model monitor.

        Args:
            model_name: Name of the model being monitored
            model_version: Version of the model
            degradation_threshold: Threshold for degradation detection (0-1)
            baseline_metrics: Baseline metrics to compare against
        """
        self.model_name = model_name
        self.model_version = model_version
        self.degradation_threshold = degradation_threshold
        self.baseline_metrics = baseline_metrics

        # Storage for snapshots
        self._snapshots: list[PerformanceSnapshot] = []
        self._max_snapshots = 1000
        self._storage_path: Path | None = None

    def calculate_metrics(
        self,
        y_true: Any,
        y_pred: Any,
        y_prob: Any | None = None,
        task_type: str = "classification",
        average: str = "weighted",
    ) -> ModelMetrics:
        """
        Calculate performance metrics for predictions.

        Args:
            y_true: Ground truth labels/values
            y_pred: Predicted labels/values
            y_prob: Prediction probabilities (for classification)
            task_type: "classification" or "regression"
            average: Averaging method for multiclass ("micro", "macro", "weighted")

        Returns:
            ModelMetrics with calculated values
        """
        import numpy as np

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if task_type == "classification":
            return self._calculate_classification_metrics(y_true, y_pred, y_prob, average)
        elif task_type == "regression":
            return self._calculate_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _calculate_classification_metrics(
        self,
        y_true: Any,
        y_pred: Any,
        y_prob: Any | None,
        average: str,
    ) -> ModelMetrics:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics = ModelMetrics()

        try:
            metrics.accuracy = float(accuracy_score(y_true, y_pred))
        except Exception:
            pass

        try:
            metrics.precision = float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )
        except Exception:
            pass

        try:
            metrics.recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        except Exception:
            pass

        try:
            metrics.f1_score = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        except Exception:
            pass

        if y_prob is not None:
            try:
                # Handle binary and multiclass
                if len(set(y_true)) == 2:
                    if y_prob.ndim > 1:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    metrics.auc_roc = float(roc_auc_score(y_true, y_prob_pos))
                else:
                    metrics.auc_roc = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                    )
            except Exception:
                pass

            try:
                metrics.log_loss = float(log_loss(y_true, y_prob))
            except Exception:
                pass

        return metrics

    def _calculate_regression_metrics(
        self,
        y_true: Any,
        y_pred: Any,
    ) -> ModelMetrics:
        """Calculate regression metrics."""
        import numpy as np
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        metrics = ModelMetrics()

        try:
            metrics.mse = float(mean_squared_error(y_true, y_pred))
            metrics.rmse = float(np.sqrt(metrics.mse))
        except Exception:
            pass

        try:
            metrics.mae = float(mean_absolute_error(y_true, y_pred))
        except Exception:
            pass

        try:
            metrics.r2_score = float(r2_score(y_true, y_pred))
        except Exception:
            pass

        return metrics

    def record_snapshot(
        self,
        metrics: ModelMetrics,
        sample_size: int,
        timestamp: datetime | None = None,
        data_start: datetime | None = None,
        data_end: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PerformanceSnapshot:
        """
        Record a performance snapshot.

        Args:
            metrics: Calculated performance metrics
            sample_size: Number of samples evaluated
            timestamp: Snapshot timestamp (defaults to now)
            data_start: Start of evaluation period
            data_end: End of evaluation period
            metadata: Additional metadata

        Returns:
            Created PerformanceSnapshot
        """
        snapshot = PerformanceSnapshot(
            snapshot_id=str(uuid.uuid4()),
            model_name=self.model_name,
            model_version=self.model_version,
            timestamp=timestamp or datetime.utcnow(),
            metrics=metrics,
            sample_size=sample_size,
            data_start=data_start,
            data_end=data_end,
            metadata=metadata or {},
        )

        self._snapshots.append(snapshot)

        # Maintain max size
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

        return snapshot

    def get_snapshots(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[PerformanceSnapshot]:
        """
        Get performance snapshots within a time range.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            limit: Maximum number of snapshots to return

        Returns:
            List of matching snapshots
        """
        snapshots = self._snapshots

        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]

        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]

        if limit:
            snapshots = snapshots[-limit:]

        return snapshots

    def get_performance_trend(
        self,
        metric_name: str,
        days: int = 7,
        end_time: datetime | None = None,
    ) -> PerformanceTrend:
        """
        Analyze performance trend for a metric.

        Args:
            metric_name: Name of the metric to analyze
            days: Number of days to include in trend
            end_time: End time for analysis (defaults to now)

        Returns:
            PerformanceTrend with trend analysis
        """
        end = end_time or datetime.utcnow()
        start = end - timedelta(days=days)

        snapshots = self.get_snapshots(start_time=start, end_time=end)

        if len(snapshots) < 2:
            return PerformanceTrend(
                model_name=self.model_name,
                metric_name=metric_name,
                time_window_days=days,
                snapshots=snapshots,
                trend_direction="stable",
                change_percentage=0.0,
                baseline_value=None,
                current_value=None,
                degradation_detected=False,
                degradation_threshold=self.degradation_threshold,
            )

        # Extract metric values
        values = []
        for s in snapshots:
            value = self._get_metric_value(s.metrics, metric_name)
            if value is not None:
                values.append((s.timestamp, value))

        if len(values) < 2:
            return PerformanceTrend(
                model_name=self.model_name,
                metric_name=metric_name,
                time_window_days=days,
                snapshots=snapshots,
                trend_direction="stable",
                change_percentage=0.0,
                baseline_value=None,
                current_value=None,
                degradation_detected=False,
                degradation_threshold=self.degradation_threshold,
            )

        values.sort(key=lambda x: x[0])
        baseline_value = values[0][1]
        current_value = values[-1][1]

        # Calculate change
        if baseline_value != 0:
            change_percentage = (current_value - baseline_value) / abs(baseline_value)
        else:
            change_percentage = 0.0

        # Determine trend direction
        if abs(change_percentage) < 0.01:
            trend_direction = "stable"
        elif change_percentage > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"

        # Check for degradation (for metrics where higher is better)
        degradation_detected = change_percentage < -self.degradation_threshold

        return PerformanceTrend(
            model_name=self.model_name,
            metric_name=metric_name,
            time_window_days=days,
            snapshots=snapshots,
            trend_direction=trend_direction,
            change_percentage=change_percentage,
            baseline_value=baseline_value,
            current_value=current_value,
            degradation_detected=degradation_detected,
            degradation_threshold=self.degradation_threshold,
        )

    def _get_metric_value(self, metrics: ModelMetrics, metric_name: str) -> float | None:
        """Extract a metric value by name."""
        # Check standard metrics
        if hasattr(metrics, metric_name):
            value = getattr(metrics, metric_name)
            return value

        # Check custom metrics
        if metric_name in metrics.custom_metrics:
            return metrics.custom_metrics[metric_name]

        return None

    def check_degradation(
        self,
        metric_name: str = "accuracy",
        days: int = 7,
    ) -> tuple[bool, PerformanceTrend]:
        """
        Check if model performance has degraded.

        Args:
            metric_name: Metric to check
            days: Time window for comparison

        Returns:
            Tuple of (is_degraded, trend)
        """
        trend = self.get_performance_trend(metric_name, days)
        return trend.degradation_detected, trend

    def compare_to_baseline(
        self,
        current_metrics: ModelMetrics,
    ) -> dict[str, dict[str, float | bool]]:
        """
        Compare current metrics to baseline.

        Args:
            current_metrics: Current performance metrics

        Returns:
            Dict with comparison for each metric
        """
        if not self.baseline_metrics:
            return {}

        comparisons = {}
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "mse",
            "rmse",
            "mae",
            "r2_score",
        ]

        for name in metric_names:
            baseline_val = self._get_metric_value(self.baseline_metrics, name)
            current_val = self._get_metric_value(current_metrics, name)

            if baseline_val is not None and current_val is not None:
                diff = current_val - baseline_val
                pct_change = diff / abs(baseline_val) if baseline_val != 0 else 0.0

                # Determine if degraded (higher is better for most, lower is better for errors)
                error_metrics = {"mse", "rmse", "mae", "log_loss"}
                if name in error_metrics:
                    degraded = diff > self.degradation_threshold * abs(baseline_val)
                else:
                    degraded = diff < -self.degradation_threshold * abs(baseline_val)

                comparisons[name] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "difference": diff,
                    "percent_change": pct_change,
                    "degraded": degraded,
                }

        return comparisons

    def set_baseline(self, metrics: ModelMetrics) -> None:
        """
        Set baseline metrics for comparison.

        Args:
            metrics: Baseline metrics to use
        """
        self.baseline_metrics = metrics

    def get_latest_metrics(self) -> ModelMetrics | None:
        """Get the most recent recorded metrics."""
        if not self._snapshots:
            return None
        return self._snapshots[-1].metrics

    def get_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Get a summary of model performance.

        Args:
            days: Number of days to include

        Returns:
            Summary dict with key statistics
        """
        snapshots = self.get_snapshots(start_time=datetime.utcnow() - timedelta(days=days))

        if not snapshots:
            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "snapshot_count": 0,
                "time_window_days": days,
            }

        # Aggregate metrics
        metrics_sum: dict[str, list[float]] = defaultdict(list)
        total_samples = 0

        for s in snapshots:
            total_samples += s.sample_size
            for metric_name in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "mse",
                "rmse",
                "mae",
                "r2_score",
            ]:
                val = self._get_metric_value(s.metrics, metric_name)
                if val is not None:
                    metrics_sum[metric_name].append(val)

        # Calculate averages
        avg_metrics = {}
        for name, values in metrics_sum.items():
            if values:
                avg_metrics[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "snapshot_count": len(snapshots),
            "time_window_days": days,
            "total_samples": total_samples,
            "metrics": avg_metrics,
            "latest_snapshot": snapshots[-1].timestamp.isoformat(),
        }

    def get_moving_average(
        self,
        metric_name: str,
        window_size: int = 5,
        days: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Calculate moving average for a metric.

        Args:
            metric_name: Name of the metric to analyze
            window_size: Number of snapshots for moving average
            days: Optional time window in days (None = all snapshots)

        Returns:
            List of dicts with timestamp and moving_average values
        """
        if days:
            snapshots = self.get_snapshots(start_time=datetime.utcnow() - timedelta(days=days))
        else:
            snapshots = self._snapshots

        if len(snapshots) < window_size:
            return []

        values = []
        for s in snapshots:
            val = self._get_metric_value(s.metrics, metric_name)
            if val is not None:
                values.append((s.timestamp, val))

        if len(values) < window_size:
            return []

        values.sort(key=lambda x: x[0])
        results = []

        for i in range(window_size - 1, len(values)):
            window_values = [v[1] for v in values[i - window_size + 1 : i + 1]]
            ma = sum(window_values) / len(window_values)
            results.append(
                {
                    "timestamp": values[i][0].isoformat(),
                    "moving_average": ma,
                    "raw_value": values[i][1],
                }
            )

        return results

    def get_percentiles(
        self,
        metric_name: str,
        percentiles: list[float] | None = None,
        days: int | None = None,
    ) -> dict[str, float | None]:
        """
        Calculate percentile statistics for a metric.

        Args:
            metric_name: Name of the metric
            percentiles: List of percentiles to calculate (default: [25, 50, 75, 90, 95])
            days: Optional time window in days

        Returns:
            Dict mapping percentile labels to values
        """
        import numpy as np

        if percentiles is None:
            percentiles = [25, 50, 75, 90, 95]

        if days:
            snapshots = self.get_snapshots(start_time=datetime.utcnow() - timedelta(days=days))
        else:
            snapshots = self._snapshots

        values = []
        for s in snapshots:
            val = self._get_metric_value(s.metrics, metric_name)
            if val is not None:
                values.append(val)

        if not values:
            return {f"p{int(p)}": None for p in percentiles}

        arr = np.array(values)
        return {f"p{int(p)}": float(np.percentile(arr, p)) for p in percentiles}

    def get_health_status(
        self,
        metrics_to_check: list[str] | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Get overall health status of the model.

        Evaluates multiple metrics and returns a health assessment.

        Args:
            metrics_to_check: List of metrics to evaluate (defaults to common metrics)
            days: Time window for trend analysis

        Returns:
            Dict with overall status and per-metric details
        """
        if metrics_to_check is None:
            metrics_to_check = ["accuracy", "f1_score", "precision", "recall"]

        if not self._snapshots:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "No performance data available",
                "metrics": {},
                "recommendations": ["Record performance snapshots to enable monitoring"],
            }

        metric_details = {}
        issues = []
        warnings = []

        for metric in metrics_to_check:
            degraded, trend = self.check_degradation(metric, days)
            latest_value = None

            if self._snapshots:
                latest_value = self._get_metric_value(self._snapshots[-1].metrics, metric)

            metric_details[metric] = {
                "current_value": latest_value,
                "trend_direction": trend.trend_direction,
                "change_percentage": trend.change_percentage,
                "degraded": degraded,
            }

            if degraded:
                issues.append(f"{metric} has degraded by {abs(trend.change_percentage):.1%}")
            elif trend.trend_direction == "declining" and abs(trend.change_percentage) > 0.02:
                warnings.append(f"{metric} is declining ({trend.change_percentage:.1%})")

        # Determine overall status
        if issues:
            status = HealthStatus.CRITICAL
            message = f"Critical issues detected: {len(issues)} metric(s) degraded"
        elif warnings:
            status = HealthStatus.WARNING
            message = f"Warnings: {len(warnings)} metric(s) showing decline"
        else:
            status = HealthStatus.HEALTHY
            message = "All monitored metrics are within acceptable ranges"

        recommendations = []
        if status == HealthStatus.CRITICAL:
            recommendations.append("Investigate root cause of metric degradation")
            recommendations.append("Consider retraining with recent data")
            recommendations.append("Check for data drift")
        elif status == HealthStatus.WARNING:
            recommendations.append("Monitor metrics closely for further decline")
            recommendations.append("Review recent data distribution changes")

        return {
            "status": status,
            "message": message,
            "metrics": metric_details,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "snapshot_count": len(self._snapshots),
            "evaluation_window_days": days,
        }

    def compare_versions(
        self,
        other_monitor: "ModelMonitor",
        metrics_to_compare: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare performance metrics between two model versions.

        Args:
            other_monitor: Another ModelMonitor instance to compare against
            metrics_to_compare: List of metrics to compare

        Returns:
            Dict with comparison results
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "mse",
                "rmse",
                "mae",
                "r2_score",
            ]

        this_latest = self.get_latest_metrics()
        other_latest = other_monitor.get_latest_metrics()

        if not this_latest or not other_latest:
            return {
                "comparison_valid": False,
                "message": "One or both monitors have no recorded metrics",
            }

        comparisons = {}
        better_count = 0
        worse_count = 0
        error_metrics = {"mse", "rmse", "mae", "log_loss"}

        for metric in metrics_to_compare:
            this_val = self._get_metric_value(this_latest, metric)
            other_val = self._get_metric_value(other_latest, metric)

            if this_val is not None and other_val is not None:
                diff = this_val - other_val
                if other_val != 0:
                    pct_diff = diff / abs(other_val)
                else:
                    pct_diff = 0.0

                # Determine which is better
                if metric in error_metrics:
                    is_better = diff < 0
                else:
                    is_better = diff > 0

                if is_better:
                    better_count += 1
                elif diff != 0:
                    worse_count += 1

                comparisons[metric] = {
                    "this_version": this_val,
                    "other_version": other_val,
                    "difference": diff,
                    "percent_difference": pct_diff,
                    "this_is_better": is_better,
                }

        return {
            "comparison_valid": True,
            "this_model": {
                "name": self.model_name,
                "version": self.model_version,
            },
            "other_model": {
                "name": other_monitor.model_name,
                "version": other_monitor.model_version,
            },
            "metrics": comparisons,
            "summary": {
                "metrics_compared": len(comparisons),
                "this_better_count": better_count,
                "other_better_count": worse_count,
                "recommendation": (
                    f"This version ({self.model_version}) is better"
                    if better_count > worse_count
                    else (
                        f"Other version ({other_monitor.model_version}) is better"
                        if worse_count > better_count
                        else "Versions are comparable"
                    )
                ),
            },
        }

    def save_snapshots(self, path: str | Path) -> bool:
        """
        Save snapshots to a JSON file.

        Args:
            path: File path to save snapshots

        Returns:
            True if successful
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "degradation_threshold": self.degradation_threshold,
            "snapshots": [s.model_dump() for s in self._snapshots],
            "saved_at": datetime.utcnow().isoformat(),
        }

        # Convert datetime objects to ISO strings for JSON serialization
        for snapshot in data["snapshots"]:
            if snapshot.get("timestamp"):
                snapshot["timestamp"] = snapshot["timestamp"].isoformat()
            if snapshot.get("data_start"):
                snapshot["data_start"] = snapshot["data_start"].isoformat()
            if snapshot.get("data_end"):
                snapshot["data_end"] = snapshot["data_end"].isoformat()

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._storage_path = path
        return True

    def load_snapshots(self, path: str | Path) -> int:
        """
        Load snapshots from a JSON file.

        Args:
            path: File path to load snapshots from

        Returns:
            Number of snapshots loaded
        """
        path = Path(path)
        if not path.exists():
            return 0

        with open(path) as f:
            data = json.load(f)

        loaded_snapshots = []
        for snap_data in data.get("snapshots", []):
            # Parse datetime strings
            if snap_data.get("timestamp"):
                snap_data["timestamp"] = datetime.fromisoformat(
                    snap_data["timestamp"].replace("Z", "+00:00")
                )
            if snap_data.get("data_start"):
                snap_data["data_start"] = datetime.fromisoformat(
                    snap_data["data_start"].replace("Z", "+00:00")
                )
            if snap_data.get("data_end"):
                snap_data["data_end"] = datetime.fromisoformat(
                    snap_data["data_end"].replace("Z", "+00:00")
                )

            # Reconstruct ModelMetrics
            if "metrics" in snap_data and isinstance(snap_data["metrics"], dict):
                snap_data["metrics"] = ModelMetrics(**snap_data["metrics"])

            loaded_snapshots.append(PerformanceSnapshot(**snap_data))

        self._snapshots = loaded_snapshots
        self._storage_path = path
        return len(loaded_snapshots)

    def clear_snapshots(self) -> int:
        """
        Clear all recorded snapshots.

        Returns:
            Number of snapshots cleared
        """
        count = len(self._snapshots)
        self._snapshots = []
        return count

    def get_metric_statistics(
        self,
        metric_name: str,
        days: int | None = None,
    ) -> dict[str, Any]:
        """
        Get comprehensive statistics for a metric.

        Args:
            metric_name: Name of the metric
            days: Optional time window in days

        Returns:
            Dict with count, mean, std, min, max, and percentiles
        """
        import numpy as np

        if days:
            snapshots = self.get_snapshots(start_time=datetime.utcnow() - timedelta(days=days))
        else:
            snapshots = self._snapshots

        values = []
        for s in snapshots:
            val = self._get_metric_value(s.metrics, metric_name)
            if val is not None:
                values.append(val)

        if not values:
            return {
                "metric_name": metric_name,
                "count": 0,
                "message": "No data available for this metric",
            }

        arr = np.array(values)
        return {
            "metric_name": metric_name,
            "count": len(values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "variance": float(np.var(arr)),
        }


class AlertManager:
    """
    Manage monitoring alerts.

    Handles alert configuration, triggering, and notification dispatch.
    """

    def __init__(self):
        """Initialize the alert manager."""
        self._configs: dict[str, AlertConfig] = {}
        self._alerts: list[Alert] = []
        self._last_alert_time: dict[str, datetime] = {}
        self._max_alerts = 1000

    def add_alert_config(self, config: AlertConfig) -> None:
        """
        Add an alert configuration.

        Args:
            config: Alert configuration to add
        """
        self._configs[config.alert_id] = config

    def remove_alert_config(self, alert_id: str) -> bool:
        """
        Remove an alert configuration.

        Args:
            alert_id: ID of the alert to remove

        Returns:
            True if removed, False if not found
        """
        if alert_id in self._configs:
            del self._configs[alert_id]
            return True
        return False

    def get_alert_configs(self) -> list[AlertConfig]:
        """Get all alert configurations."""
        return list(self._configs.values())

    def check_and_trigger(
        self,
        metric_name: str,
        metric_value: float,
        model_name: str | None = None,
    ) -> list[Alert]:
        """
        Check if any alerts should be triggered.

        Args:
            metric_name: Name of the metric
            metric_value: Current value of the metric
            model_name: Optional model name

        Returns:
            List of triggered alerts
        """
        triggered = []

        for config in self._configs.values():
            if not config.enabled:
                continue

            if config.metric_name and config.metric_name != metric_name:
                continue

            if config.threshold is None:
                continue

            # Check cooldown
            last_time = self._last_alert_time.get(config.alert_id)
            if last_time:
                cooldown = timedelta(minutes=config.cooldown_minutes)
                if datetime.utcnow() - last_time < cooldown:
                    continue

            # Check threshold
            should_trigger = self._check_threshold(
                metric_value, config.threshold, config.comparison
            )

            if should_trigger:
                alert = self._create_alert(config, metric_name, metric_value, model_name)
                triggered.append(alert)
                self._alerts.append(alert)
                self._last_alert_time[config.alert_id] = datetime.utcnow()

        # Maintain max alerts
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]

        return triggered

    def _check_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if value crosses threshold."""
        if comparison == "lt":
            return value < threshold
        elif comparison == "gt":
            return value > threshold
        elif comparison == "lte":
            return value <= threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "eq":
            return value == threshold
        elif comparison == "neq":
            return value != threshold
        return False

    def _create_alert(
        self,
        config: AlertConfig,
        metric_name: str,
        metric_value: float,
        model_name: str | None,
    ) -> Alert:
        """Create an alert instance."""
        return Alert(
            alert_id=str(uuid.uuid4()),
            config_id=config.alert_id,
            timestamp=datetime.utcnow(),
            level=config.level,
            title=f"{config.name}: {metric_name} threshold breached",
            message=f"Metric '{metric_name}' has value {metric_value:.4f} which "
            f"breaches threshold {config.threshold:.4f} ({config.comparison})",
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=config.threshold,
            model_name=model_name,
        )

    def trigger_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.WARNING,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """
        Manually trigger an alert.

        Args:
            title: Alert title
            message: Alert message
            level: Alert severity level
            model_name: Associated model name
            metadata: Additional metadata

        Returns:
            Created alert
        """
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            config_id="manual",
            timestamp=datetime.utcnow(),
            level=level,
            title=title,
            message=message,
            model_name=model_name,
            metadata=metadata or {},
        )

        self._alerts.append(alert)
        return alert

    def get_alerts(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        level: AlertLevel | None = None,
        resolved: bool | None = None,
        limit: int | None = None,
    ) -> list[Alert]:
        """
        Get alerts matching criteria.

        Args:
            start_time: Start of time range
            end_time: End of time range
            level: Filter by alert level
            resolved: Filter by resolved status
            limit: Maximum alerts to return

        Returns:
            List of matching alerts
        """
        alerts = self._alerts

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]

        if level:
            alerts = [a for a in alerts if a.level == level]

        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert
            acknowledged_by: Who acknowledged it

        Returns:
            True if found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: ID of the alert

        Returns:
            True if found and resolved
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return True
        return False

    def get_unresolved_count(self) -> dict[str, int]:
        """Get count of unresolved alerts by level."""
        counts: dict[str, int] = defaultdict(int)
        for alert in self._alerts:
            if not alert.resolved:
                counts[alert.level.value] += 1
        return dict(counts)
