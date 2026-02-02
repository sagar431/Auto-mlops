"""
Model Monitoring Module for MLOps Agent.

Provides data drift detection, model performance monitoring, and alerting
capabilities for ML models in production environments.

Components:
- DriftDetector: Data drift detection using Evidently AI integration
- ConceptDriftDetector: Concept drift detection for model-target relationships
- ModelMonitor: Model performance tracking and degradation detection
- AlertManager: Alert configuration, triggering, and notification

Models:
- DriftReport: Complete drift detection report with feature-level results
- FeatureDriftResult: Drift analysis for a single feature
- ModelMetrics: Classification and regression performance metrics
- PerformanceSnapshot: Point-in-time performance capture
- PerformanceTrend: Trend analysis over time
- Alert: Alert instance with metadata
- AlertConfig: Alert configuration settings

Example usage:
    # Drift Detection
    from monitoring import DriftDetector

    detector = DriftDetector(drift_threshold=0.1)
    report = detector.detect_drift(
        reference_data=training_df,
        current_data=production_df,
    )

    if report.overall_drift_detected:
        print(f"Drift detected! Severity: {report.severity}")
        for feature in report.feature_results:
            if feature.drift_detected:
                print(f"  - {feature.feature_name}: score={feature.drift_score:.3f}")

    # Model Performance Monitoring
    from monitoring import ModelMonitor

    monitor = ModelMonitor(
        model_name="fraud_detector",
        degradation_threshold=0.05,
    )

    # Record performance
    metrics = monitor.calculate_metrics(
        y_true=labels,
        y_pred=predictions,
        task_type="classification",
    )
    monitor.record_snapshot(metrics, sample_size=len(labels))

    # Check for degradation
    degraded, trend = monitor.check_degradation("accuracy", days=7)
    if degraded:
        print(f"Performance degraded by {trend.change_percentage:.2%}")

    # Alerting
    from monitoring import AlertManager, AlertConfig, AlertLevel, AlertChannel

    alert_manager = AlertManager()
    alert_manager.add_alert_config(AlertConfig(
        alert_id="accuracy_alert",
        name="Low Accuracy Alert",
        channel=AlertChannel.SLACK,
        level=AlertLevel.WARNING,
        metric_name="accuracy",
        threshold=0.85,
        comparison="lt",
        slack_channel="#ml-alerts",
    ))

    # Check and trigger alerts
    alerts = alert_manager.check_and_trigger(
        metric_name="accuracy",
        metric_value=0.82,
        model_name="fraud_detector",
    )

    # Concept Drift Detection
    from monitoring import ConceptDriftDetector

    concept_detector = ConceptDriftDetector()
    report = concept_detector.detect_concept_drift(
        reference_data=train_df,
        current_data=prod_df,
        target_column="label",
        prediction_column="prediction",
    )
"""

from .drift_detector import ConceptDriftDetector, DriftDetector
from .model_monitor import AlertManager, HealthStatus, ModelMonitor
from .models import (
    Alert,
    AlertChannel,
    AlertConfig,
    AlertLevel,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
    ModelMetrics,
    MonitoringConfig,
    PerformanceSnapshot,
    PerformanceTrend,
)

__all__ = [
    # Drift Detection
    "DriftDetector",
    "ConceptDriftDetector",
    # Model Monitoring
    "ModelMonitor",
    "AlertManager",
    "HealthStatus",
    # Drift Models
    "DriftReport",
    "FeatureDriftResult",
    "DriftType",
    "DriftSeverity",
    # Performance Models
    "ModelMetrics",
    "PerformanceSnapshot",
    "PerformanceTrend",
    # Alert Models
    "Alert",
    "AlertConfig",
    "AlertLevel",
    "AlertChannel",
    # Config Models
    "MonitoringConfig",
]
