"""
Unit tests for the monitoring module.

Tests drift detection, model monitoring, and alerting functionality.
"""

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from monitoring import (
    Alert,
    AlertChannel,
    AlertConfig,
    AlertLevel,
    AlertManager,
    ConceptDriftDetector,
    DriftDetector,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
    HealthStatus,
    ModelMetrics,
    ModelMonitor,
    MonitoringConfig,
    PerformanceSnapshot,
    PerformanceTrend,
)


class TestDriftType:
    """Tests for DriftType enum."""

    def test_drift_type_values(self):
        """Test DriftType enum values."""
        assert DriftType.DATA == "data"
        assert DriftType.CONCEPT == "concept"
        assert DriftType.PREDICTION == "prediction"
        assert DriftType.FEATURE == "feature"


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    def test_drift_severity_values(self):
        """Test DriftSeverity enum values."""
        assert DriftSeverity.NONE == "none"
        assert DriftSeverity.LOW == "low"
        assert DriftSeverity.MEDIUM == "medium"
        assert DriftSeverity.HIGH == "high"
        assert DriftSeverity.CRITICAL == "critical"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_level_values(self):
        """Test AlertLevel enum values."""
        assert AlertLevel.INFO == "info"
        assert AlertLevel.WARNING == "warning"
        assert AlertLevel.ERROR == "error"
        assert AlertLevel.CRITICAL == "critical"


class TestAlertChannel:
    """Tests for AlertChannel enum."""

    def test_alert_channel_values(self):
        """Test AlertChannel enum values."""
        assert AlertChannel.SLACK == "slack"
        assert AlertChannel.EMAIL == "email"
        assert AlertChannel.WEBHOOK == "webhook"
        assert AlertChannel.PAGERDUTY == "pagerduty"
        assert AlertChannel.LOG == "log"


class TestFeatureDriftResult:
    """Tests for FeatureDriftResult model."""

    def test_create_feature_drift_result(self):
        """Test creating a FeatureDriftResult."""
        result = FeatureDriftResult(
            feature_name="age",
            drift_detected=True,
            drift_score=0.75,
            stattest_name="ks",
            stattest_threshold=0.1,
            p_value=0.001,
        )
        assert result.feature_name == "age"
        assert result.drift_detected is True
        assert result.drift_score == 0.75
        assert result.stattest_name == "ks"
        assert result.stattest_threshold == 0.1
        assert result.p_value == 0.001

    def test_feature_drift_result_with_distributions(self):
        """Test FeatureDriftResult with distribution data."""
        result = FeatureDriftResult(
            feature_name="income",
            drift_detected=False,
            drift_score=0.05,
            stattest_name="ks",
            stattest_threshold=0.1,
            reference_distribution={"mean": 50000, "std": 10000},
            current_distribution={"mean": 51000, "std": 10500},
        )
        assert result.reference_distribution["mean"] == 50000
        assert result.current_distribution["mean"] == 51000


class TestDriftReport:
    """Tests for DriftReport model."""

    def test_create_drift_report(self):
        """Test creating a DriftReport."""
        report = DriftReport(
            report_id=str(uuid.uuid4()),
            dataset_name="test_dataset",
            drift_type=DriftType.DATA,
            overall_drift_detected=True,
            drift_share=0.4,
            severity=DriftSeverity.MEDIUM,
            reference_rows=1000,
            current_rows=500,
        )
        assert report.dataset_name == "test_dataset"
        assert report.drift_type == DriftType.DATA
        assert report.overall_drift_detected is True
        assert report.drift_share == 0.4
        assert report.severity == DriftSeverity.MEDIUM

    def test_drift_report_with_feature_results(self):
        """Test DriftReport with feature results."""
        feature1 = FeatureDriftResult(
            feature_name="age",
            drift_detected=True,
            drift_score=0.8,
            stattest_name="ks",
            stattest_threshold=0.1,
        )
        feature2 = FeatureDriftResult(
            feature_name="income",
            drift_detected=False,
            drift_score=0.05,
            stattest_name="ks",
            stattest_threshold=0.1,
        )
        report = DriftReport(
            report_id=str(uuid.uuid4()),
            dataset_name="test_dataset",
            drift_type=DriftType.DATA,
            overall_drift_detected=True,
            drift_share=0.5,
            severity=DriftSeverity.HIGH,
            feature_results=[feature1, feature2],
            reference_rows=1000,
            current_rows=500,
            recommendations=["Retrain the model"],
        )
        assert len(report.feature_results) == 2
        assert report.feature_results[0].feature_name == "age"
        assert len(report.recommendations) == 1


class TestModelMetrics:
    """Tests for ModelMetrics model."""

    def test_create_classification_metrics(self):
        """Test creating classification metrics."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.92,
            f1_score=0.925,
            auc_roc=0.98,
        )
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.93
        assert metrics.recall == 0.92
        assert metrics.f1_score == 0.925
        assert metrics.auc_roc == 0.98

    def test_create_regression_metrics(self):
        """Test creating regression metrics."""
        metrics = ModelMetrics(
            mse=0.05,
            rmse=0.224,
            mae=0.15,
            r2_score=0.92,
        )
        assert metrics.mse == 0.05
        assert metrics.rmse == 0.224
        assert metrics.mae == 0.15
        assert metrics.r2_score == 0.92

    def test_custom_metrics(self):
        """Test custom metrics field."""
        metrics = ModelMetrics(
            accuracy=0.90,
            custom_metrics={"top_5_accuracy": 0.98, "confusion_matrix_sum": 1000},
        )
        assert metrics.custom_metrics["top_5_accuracy"] == 0.98


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot model."""

    def test_create_snapshot(self):
        """Test creating a performance snapshot."""
        metrics = ModelMetrics(accuracy=0.90)
        snapshot = PerformanceSnapshot(
            snapshot_id=str(uuid.uuid4()),
            model_name="fraud_detector",
            model_version="1.0.0",
            metrics=metrics,
            sample_size=1000,
        )
        assert snapshot.model_name == "fraud_detector"
        assert snapshot.model_version == "1.0.0"
        assert snapshot.metrics.accuracy == 0.90
        assert snapshot.sample_size == 1000


class TestPerformanceTrend:
    """Tests for PerformanceTrend model."""

    def test_create_trend(self):
        """Test creating a performance trend."""
        trend = PerformanceTrend(
            model_name="fraud_detector",
            metric_name="accuracy",
            time_window_days=7,
            trend_direction="declining",
            change_percentage=-0.05,
            baseline_value=0.95,
            current_value=0.90,
            degradation_detected=True,
            degradation_threshold=0.03,
        )
        assert trend.trend_direction == "declining"
        assert trend.change_percentage == -0.05
        assert trend.degradation_detected is True


class TestAlertConfig:
    """Tests for AlertConfig model."""

    def test_create_alert_config(self):
        """Test creating an alert configuration."""
        config = AlertConfig(
            alert_id="accuracy_alert",
            name="Low Accuracy Alert",
            channel=AlertChannel.SLACK,
            level=AlertLevel.WARNING,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
            slack_channel="#ml-alerts",
        )
        assert config.alert_id == "accuracy_alert"
        assert config.channel == AlertChannel.SLACK
        assert config.level == AlertLevel.WARNING
        assert config.threshold == 0.85


class TestAlert:
    """Tests for Alert model."""

    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            config_id="accuracy_alert",
            level=AlertLevel.WARNING,
            title="Low Accuracy Alert",
            message="Accuracy dropped below threshold",
            metric_name="accuracy",
            metric_value=0.80,
            threshold_value=0.85,
            model_name="fraud_detector",
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_value == 0.80
        assert alert.resolved is False


class TestMonitoringConfig:
    """Tests for MonitoringConfig model."""

    def test_create_monitoring_config(self):
        """Test creating a monitoring configuration."""
        config = MonitoringConfig(
            model_name="fraud_detector",
            model_version="1.0.0",
            drift_threshold=0.15,
            degradation_threshold=0.1,
            check_interval_minutes=30,
        )
        assert config.model_name == "fraud_detector"
        assert config.drift_threshold == 0.15
        assert config.degradation_threshold == 0.1


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a DriftDetector instance."""
        return DriftDetector(drift_threshold=0.1, stattest="ks")

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.normal(40, 10, 1000),
                "income": np.random.normal(50000, 15000, 1000),
                "category": np.random.choice(["A", "B", "C"], 1000),
            }
        )

    @pytest.fixture
    def current_data_no_drift(self, reference_data):
        """Create current dataset with no drift."""
        np.random.seed(43)
        return pd.DataFrame(
            {
                "age": np.random.normal(40, 10, 500),
                "income": np.random.normal(50000, 15000, 500),
                "category": np.random.choice(["A", "B", "C"], 500),
            }
        )

    @pytest.fixture
    def current_data_with_drift(self):
        """Create current dataset with drift."""
        np.random.seed(44)
        return pd.DataFrame(
            {
                "age": np.random.normal(60, 15, 500),  # Shifted mean
                "income": np.random.normal(80000, 20000, 500),  # Shifted mean
                "category": np.random.choice(["A", "B", "D"], 500),  # Different categories
            }
        )

    def test_detector_init(self, detector):
        """Test DriftDetector initialization."""
        assert detector.drift_threshold == 0.1
        assert detector.stattest == "ks"
        assert detector.per_feature_stattest == {}

    def test_evidently_available_property(self, detector):
        """Test evidently_available property."""
        # Should return bool
        assert isinstance(detector.evidently_available, bool)

    def test_detect_drift_with_installed_evidently_api(
        self, detector, reference_data, current_data_no_drift
    ):
        """Test installed Evidently API versions do not leak import errors."""
        if not detector.evidently_available:
            pytest.skip("Evidently is not installed")

        report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
            dataset_name="evidently_api_compat",
        )

        assert isinstance(report, DriftReport)
        assert report.feature_results

    def test_detect_drift_no_drift(self, detector, reference_data, current_data_no_drift):
        """Test drift detection with no drift."""
        report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
            dataset_name="test_data",
        )
        assert isinstance(report, DriftReport)
        assert report.dataset_name == "test_data"
        assert report.drift_type == DriftType.DATA
        assert report.reference_rows == 1000
        assert report.current_rows == 500

    def test_detect_drift_with_drift(self, detector, reference_data, current_data_with_drift):
        """Test drift detection with drift present."""
        report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            dataset_name="test_data",
        )
        assert isinstance(report, DriftReport)
        # With such different distributions, we expect drift to be detected
        assert len(report.feature_results) > 0

    def test_detect_drift_specific_columns(self, detector, reference_data, current_data_with_drift):
        """Test drift detection on specific columns."""
        report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            feature_columns=["age", "income"],
            dataset_name="test_data",
        )
        # Only 2 columns analyzed
        assert len(report.feature_results) <= 2

    def test_calculate_severity_none(self, detector):
        """Test severity calculation with no drift."""
        severity = detector._calculate_severity(0.0, False)
        assert severity == DriftSeverity.NONE

    def test_calculate_severity_low(self, detector):
        """Test severity calculation with low drift."""
        severity = detector._calculate_severity(0.05, False)
        assert severity == DriftSeverity.LOW

    def test_calculate_severity_medium(self, detector):
        """Test severity calculation with medium drift."""
        severity = detector._calculate_severity(0.2, False)
        assert severity == DriftSeverity.MEDIUM

    def test_calculate_severity_high(self, detector):
        """Test severity calculation with high drift."""
        severity = detector._calculate_severity(0.4, True)
        assert severity == DriftSeverity.HIGH

    def test_calculate_severity_critical(self, detector):
        """Test severity calculation with critical drift."""
        severity = detector._calculate_severity(0.6, True)
        assert severity == DriftSeverity.CRITICAL

    def test_generate_recommendations_no_drift(self, detector):
        """Test recommendations when no drift."""
        recommendations = detector._generate_recommendations([], 0.0, DriftSeverity.NONE)
        assert len(recommendations) > 0
        assert "No significant drift" in recommendations[0]

    def test_generate_recommendations_with_drift(self, detector):
        """Test recommendations when drift detected."""
        feature_results = [
            FeatureDriftResult(
                feature_name="age",
                drift_detected=True,
                drift_score=0.8,
                stattest_name="ks",
                stattest_threshold=0.1,
            )
        ]
        recommendations = detector._generate_recommendations(
            feature_results, 0.5, DriftSeverity.HIGH
        )
        assert len(recommendations) > 0
        # Should recommend retraining
        assert any("retrain" in r.lower() for r in recommendations)

    def test_detect_prediction_drift(self, detector):
        """Test prediction drift detection."""
        np.random.seed(42)
        ref_preds = np.random.normal(0.5, 0.1, 1000)
        cur_preds = np.random.normal(0.7, 0.15, 500)

        report = detector.detect_prediction_drift(
            reference_predictions=ref_preds,
            current_predictions=cur_preds,
            prediction_column="prediction",
        )
        assert report.drift_type == DriftType.PREDICTION

    def test_detect_prediction_drift_dataframe(self, detector):
        """Test prediction drift detection with DataFrame input."""
        np.random.seed(42)
        ref_df = pd.DataFrame({"prediction": np.random.normal(0.5, 0.1, 1000)})
        cur_df = pd.DataFrame({"prediction": np.random.normal(0.5, 0.1, 500)})

        report = detector.detect_prediction_drift(
            reference_predictions=ref_df,
            current_predictions=cur_df,
        )
        assert report.drift_type == DriftType.PREDICTION


class TestConceptDriftDetector:
    """Tests for ConceptDriftDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a ConceptDriftDetector instance."""
        return ConceptDriftDetector(significance_level=0.05)

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset with target."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
                "target": np.random.binomial(1, 0.5, 1000),
            }
        )
        df["prediction"] = df["target"] + np.random.normal(0, 0.1, 1000)
        return df

    @pytest.fixture
    def current_data_no_drift(self):
        """Create current dataset without concept drift."""
        np.random.seed(43)
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "feature2": np.random.normal(5, 2, 500),
                "target": np.random.binomial(1, 0.5, 500),
            }
        )
        df["prediction"] = df["target"] + np.random.normal(0, 0.1, 500)
        return df

    @pytest.fixture
    def current_data_with_drift(self):
        """Create current dataset with concept drift."""
        np.random.seed(44)
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "feature2": np.random.normal(5, 2, 500),
                "target": np.random.binomial(1, 0.8, 500),  # Different target distribution
            }
        )
        df["prediction"] = np.random.normal(0.5, 0.3, 500)  # Predictions no longer match target
        return df

    def test_detector_init(self, detector):
        """Test ConceptDriftDetector initialization."""
        assert detector.significance_level == 0.05

    def test_detect_concept_drift_no_drift(self, detector, reference_data, current_data_no_drift):
        """Test concept drift detection with no drift."""
        report = detector.detect_concept_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
            target_column="target",
            prediction_column="prediction",
        )
        assert isinstance(report, DriftReport)
        assert report.drift_type == DriftType.CONCEPT

    def test_detect_concept_drift_with_drift(
        self, detector, reference_data, current_data_with_drift
    ):
        """Test concept drift detection with drift present."""
        report = detector.detect_concept_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            target_column="target",
            prediction_column="prediction",
        )
        assert isinstance(report, DriftReport)
        assert report.drift_type == DriftType.CONCEPT

    def test_detect_concept_drift_without_predictions(
        self, detector, reference_data, current_data_no_drift
    ):
        """Test concept drift detection without prediction column."""
        report = detector.detect_concept_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
            target_column="target",
        )
        assert isinstance(report, DriftReport)
        assert report.drift_type == DriftType.CONCEPT

    def test_detect_concept_drift_missing_target(self, detector):
        """Test error when target column missing."""
        ref_df = pd.DataFrame({"feature1": [1, 2, 3]})
        cur_df = pd.DataFrame({"feature1": [4, 5, 6]})

        with pytest.raises(ValueError, match="Target column"):
            detector.detect_concept_drift(
                reference_data=ref_df,
                current_data=cur_df,
                target_column="target",
            )

    def test_detect_concept_drift_categorical_target(self, detector):
        """Test concept drift detection with categorical target."""
        np.random.seed(42)
        ref_df = pd.DataFrame(
            {
                "feature": np.random.normal(0, 1, 1000),
                "target": np.random.choice(["A", "B", "C"], 1000),
            }
        )
        cur_df = pd.DataFrame(
            {
                "feature": np.random.normal(0, 1, 500),
                "target": np.random.choice(["A", "B", "C"], 500),
            }
        )

        report = detector.detect_concept_drift(
            reference_data=ref_df,
            current_data=cur_df,
            target_column="target",
        )
        assert isinstance(report, DriftReport)


class TestModelMonitor:
    """Tests for ModelMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a ModelMonitor instance."""
        return ModelMonitor(
            model_name="test_model",
            model_version="1.0.0",
            degradation_threshold=0.05,
        )

    def test_monitor_init(self, monitor):
        """Test ModelMonitor initialization."""
        assert monitor.model_name == "test_model"
        assert monitor.model_version == "1.0.0"
        assert monitor.degradation_threshold == 0.05
        assert monitor.baseline_metrics is None

    def test_calculate_classification_metrics(self, monitor):
        """Test classification metrics calculation."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])

        metrics = monitor.calculate_metrics(y_true, y_pred, task_type="classification")
        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None

    def test_calculate_classification_metrics_with_probs(self, monitor):
        """Test classification metrics with probabilities."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.8])

        metrics = monitor.calculate_metrics(
            y_true, y_pred, y_prob=y_prob, task_type="classification"
        )
        assert metrics.auc_roc is not None or metrics.log_loss is not None

    def test_calculate_regression_metrics(self, monitor):
        """Test regression metrics calculation."""
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        metrics = monitor.calculate_metrics(y_true, y_pred, task_type="regression")
        assert isinstance(metrics, ModelMetrics)
        assert metrics.mse is not None
        assert metrics.rmse is not None
        assert metrics.mae is not None
        assert metrics.r2_score is not None

    def test_calculate_invalid_task_type(self, monitor):
        """Test error for invalid task type."""
        with pytest.raises(ValueError, match="Unknown task type"):
            monitor.calculate_metrics([1], [1], task_type="invalid")

    def test_record_snapshot(self, monitor):
        """Test recording a performance snapshot."""
        metrics = ModelMetrics(accuracy=0.95)
        snapshot = monitor.record_snapshot(metrics, sample_size=1000)

        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.model_name == "test_model"
        assert snapshot.metrics.accuracy == 0.95
        assert snapshot.sample_size == 1000

    def test_record_snapshot_with_metadata(self, monitor):
        """Test recording snapshot with metadata."""
        metrics = ModelMetrics(accuracy=0.90)
        snapshot = monitor.record_snapshot(
            metrics,
            sample_size=500,
            metadata={"batch_id": "batch_001"},
        )
        assert snapshot.metadata["batch_id"] == "batch_001"

    def test_get_snapshots(self, monitor):
        """Test getting snapshots."""
        # Record some snapshots
        for i in range(5):
            metrics = ModelMetrics(accuracy=0.9 + i * 0.01)
            monitor.record_snapshot(metrics, sample_size=100)

        snapshots = monitor.get_snapshots()
        assert len(snapshots) == 5

    def test_get_snapshots_with_limit(self, monitor):
        """Test getting snapshots with limit."""
        for i in range(10):
            metrics = ModelMetrics(accuracy=0.9)
            monitor.record_snapshot(metrics, sample_size=100)

        snapshots = monitor.get_snapshots(limit=5)
        assert len(snapshots) == 5

    def test_get_snapshots_with_time_range(self, monitor):
        """Test getting snapshots within time range."""
        now = datetime.utcnow()
        # Record snapshot with past timestamp
        metrics = ModelMetrics(accuracy=0.9)
        monitor.record_snapshot(metrics, sample_size=100, timestamp=now - timedelta(days=1))

        # Record snapshot with current timestamp
        monitor.record_snapshot(metrics, sample_size=100, timestamp=now)

        # Query with time range
        snapshots = monitor.get_snapshots(start_time=now - timedelta(hours=1))
        assert len(snapshots) == 1

    def test_get_performance_trend(self, monitor):
        """Test getting performance trend."""
        now = datetime.utcnow()
        # Record snapshots with declining accuracy
        for i in range(5):
            metrics = ModelMetrics(accuracy=0.95 - i * 0.02)
            monitor.record_snapshot(metrics, sample_size=100, timestamp=now - timedelta(days=4 - i))

        trend = monitor.get_performance_trend("accuracy", days=7)
        assert isinstance(trend, PerformanceTrend)
        assert trend.metric_name == "accuracy"
        assert trend.time_window_days == 7

    def test_get_performance_trend_insufficient_data(self, monitor):
        """Test trend with insufficient data."""
        trend = monitor.get_performance_trend("accuracy", days=7)
        assert trend.trend_direction == "stable"
        assert trend.degradation_detected is False

    def test_check_degradation(self, monitor):
        """Test checking for degradation."""
        now = datetime.utcnow()
        # Record snapshots with significant decline
        monitor.record_snapshot(
            ModelMetrics(accuracy=0.95), sample_size=100, timestamp=now - timedelta(days=6)
        )
        monitor.record_snapshot(ModelMetrics(accuracy=0.85), sample_size=100, timestamp=now)

        degraded, trend = monitor.check_degradation("accuracy", days=7)
        assert degraded is True
        assert trend.trend_direction == "declining"

    def test_set_baseline(self, monitor):
        """Test setting baseline metrics."""
        baseline = ModelMetrics(accuracy=0.95, precision=0.93)
        monitor.set_baseline(baseline)
        assert monitor.baseline_metrics.accuracy == 0.95

    def test_compare_to_baseline(self, monitor):
        """Test comparing to baseline."""
        monitor.set_baseline(ModelMetrics(accuracy=0.95, precision=0.93))
        current = ModelMetrics(accuracy=0.90, precision=0.91)

        comparison = monitor.compare_to_baseline(current)
        assert "accuracy" in comparison
        assert comparison["accuracy"]["baseline"] == 0.95
        assert comparison["accuracy"]["current"] == 0.90

    def test_compare_to_baseline_no_baseline(self, monitor):
        """Test comparing without baseline set."""
        current = ModelMetrics(accuracy=0.90)
        comparison = monitor.compare_to_baseline(current)
        assert comparison == {}

    def test_get_latest_metrics(self, monitor):
        """Test getting latest metrics."""
        monitor.record_snapshot(ModelMetrics(accuracy=0.90), sample_size=100)
        monitor.record_snapshot(ModelMetrics(accuracy=0.92), sample_size=100)

        latest = monitor.get_latest_metrics()
        assert latest.accuracy == 0.92

    def test_get_latest_metrics_empty(self, monitor):
        """Test getting latest metrics when empty."""
        assert monitor.get_latest_metrics() is None

    def test_get_summary(self, monitor):
        """Test getting summary."""
        now = datetime.utcnow()
        for i in range(3):
            monitor.record_snapshot(
                ModelMetrics(accuracy=0.90 + i * 0.02),
                sample_size=100,
                timestamp=now - timedelta(days=i),
            )

        summary = monitor.get_summary(days=7)
        assert summary["model_name"] == "test_model"
        assert summary["snapshot_count"] == 3
        assert "metrics" in summary

    def test_get_summary_empty(self, monitor):
        """Test summary with no data."""
        summary = monitor.get_summary(days=7)
        assert summary["snapshot_count"] == 0

    def test_get_moving_average(self, monitor):
        """Test moving average calculation."""
        now = datetime.utcnow()
        for i in range(10):
            monitor.record_snapshot(
                ModelMetrics(accuracy=0.90 + i * 0.01),
                sample_size=100,
                timestamp=now - timedelta(days=9 - i),
            )

        ma = monitor.get_moving_average("accuracy", window_size=3)
        assert len(ma) > 0
        assert "moving_average" in ma[0]
        assert "raw_value" in ma[0]

    def test_get_moving_average_insufficient_data(self, monitor):
        """Test moving average with insufficient data."""
        monitor.record_snapshot(ModelMetrics(accuracy=0.90), sample_size=100)
        ma = monitor.get_moving_average("accuracy", window_size=5)
        assert len(ma) == 0

    def test_get_percentiles(self, monitor):
        """Test percentile calculation."""
        for i in range(20):
            monitor.record_snapshot(ModelMetrics(accuracy=0.80 + i * 0.01), sample_size=100)

        percentiles = monitor.get_percentiles("accuracy")
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert percentiles["p50"] is not None

    def test_get_percentiles_empty(self, monitor):
        """Test percentiles with no data."""
        percentiles = monitor.get_percentiles("accuracy")
        assert percentiles["p50"] is None

    def test_get_health_status_healthy(self, monitor):
        """Test health status when healthy."""
        now = datetime.utcnow()
        for i in range(5):
            monitor.record_snapshot(
                ModelMetrics(accuracy=0.95),
                sample_size=100,
                timestamp=now - timedelta(days=i),
            )

        health = monitor.get_health_status(days=7)
        assert health["status"] == HealthStatus.HEALTHY

    def test_get_health_status_no_data(self, monitor):
        """Test health status with no data."""
        health = monitor.get_health_status()
        assert health["status"] == HealthStatus.UNKNOWN

    def test_get_health_status_degraded(self, monitor):
        """Test health status when degraded."""
        now = datetime.utcnow()
        monitor.record_snapshot(
            ModelMetrics(accuracy=0.95),
            sample_size=100,
            timestamp=now - timedelta(days=6),
        )
        monitor.record_snapshot(
            ModelMetrics(accuracy=0.80),  # Significant drop
            sample_size=100,
            timestamp=now,
        )

        health = monitor.get_health_status(days=7)
        assert health["status"] in [HealthStatus.CRITICAL, HealthStatus.WARNING]

    def test_compare_versions(self, monitor):
        """Test comparing model versions."""
        other = ModelMonitor(model_name="test_model", model_version="2.0.0")

        monitor.record_snapshot(ModelMetrics(accuracy=0.90), sample_size=100)
        other.record_snapshot(ModelMetrics(accuracy=0.92), sample_size=100)

        comparison = monitor.compare_versions(other)
        assert comparison["comparison_valid"] is True
        assert "metrics" in comparison

    def test_compare_versions_no_data(self, monitor):
        """Test comparing versions with no data."""
        other = ModelMonitor(model_name="test_model", model_version="2.0.0")
        comparison = monitor.compare_versions(other)
        assert comparison["comparison_valid"] is False

    def test_save_and_load_snapshots(self, monitor, tmp_path):
        """Test saving and loading snapshots."""
        # Record some snapshots
        for i in range(5):
            monitor.record_snapshot(ModelMetrics(accuracy=0.90 + i * 0.01), sample_size=100)

        # Save
        save_path = tmp_path / "snapshots.json"
        result = monitor.save_snapshots(save_path)
        assert result is True
        assert save_path.exists()

        # Create new monitor and load
        new_monitor = ModelMonitor(model_name="test_model")
        loaded_count = new_monitor.load_snapshots(save_path)
        assert loaded_count == 5

    def test_load_snapshots_nonexistent_file(self, monitor, tmp_path):
        """Test loading from nonexistent file."""
        count = monitor.load_snapshots(tmp_path / "nonexistent.json")
        assert count == 0

    def test_clear_snapshots(self, monitor):
        """Test clearing snapshots."""
        for i in range(5):
            monitor.record_snapshot(ModelMetrics(accuracy=0.90), sample_size=100)

        count = monitor.clear_snapshots()
        assert count == 5
        assert len(monitor.get_snapshots()) == 0

    def test_get_metric_statistics(self, monitor):
        """Test metric statistics calculation."""
        for i in range(20):
            monitor.record_snapshot(ModelMetrics(accuracy=0.80 + i * 0.01), sample_size=100)

        stats = monitor.get_metric_statistics("accuracy")
        assert stats["count"] == 20
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_get_metric_statistics_empty(self, monitor):
        """Test statistics with no data."""
        stats = monitor.get_metric_statistics("accuracy")
        assert stats["count"] == 0

    def test_max_snapshots_limit(self, monitor):
        """Test that max snapshots limit is enforced."""
        monitor._max_snapshots = 10

        # Record more than max
        for i in range(15):
            monitor.record_snapshot(ModelMetrics(accuracy=0.90), sample_size=100)

        assert len(monitor._snapshots) == 10


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.fixture
    def manager(self):
        """Create an AlertManager instance."""
        return AlertManager()

    @pytest.fixture
    def alert_config(self):
        """Create an alert configuration."""
        return AlertConfig(
            alert_id="test_alert",
            name="Test Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
            cooldown_minutes=1,
        )

    def test_manager_init(self, manager):
        """Test AlertManager initialization."""
        assert len(manager._configs) == 0
        assert len(manager._alerts) == 0

    def test_add_alert_config(self, manager, alert_config):
        """Test adding alert configuration."""
        manager.add_alert_config(alert_config)
        assert "test_alert" in manager._configs

    def test_remove_alert_config(self, manager, alert_config):
        """Test removing alert configuration."""
        manager.add_alert_config(alert_config)
        result = manager.remove_alert_config("test_alert")
        assert result is True
        assert "test_alert" not in manager._configs

    def test_remove_nonexistent_config(self, manager):
        """Test removing nonexistent configuration."""
        result = manager.remove_alert_config("nonexistent")
        assert result is False

    def test_get_alert_configs(self, manager, alert_config):
        """Test getting all configurations."""
        manager.add_alert_config(alert_config)
        configs = manager.get_alert_configs()
        assert len(configs) == 1

    def test_check_and_trigger_below_threshold(self, manager, alert_config):
        """Test triggering alert when below threshold."""
        manager.add_alert_config(alert_config)
        triggered = manager.check_and_trigger("accuracy", 0.80, model_name="test_model")
        assert len(triggered) == 1
        assert triggered[0].metric_value == 0.80

    def test_check_and_trigger_above_threshold(self, manager, alert_config):
        """Test no alert when above threshold."""
        manager.add_alert_config(alert_config)
        triggered = manager.check_and_trigger("accuracy", 0.90, model_name="test_model")
        assert len(triggered) == 0

    def test_check_and_trigger_wrong_metric(self, manager, alert_config):
        """Test no alert for different metric."""
        manager.add_alert_config(alert_config)
        triggered = manager.check_and_trigger("f1_score", 0.50, model_name="test_model")
        assert len(triggered) == 0

    def test_check_threshold_comparisons(self, manager):
        """Test different threshold comparisons."""
        assert manager._check_threshold(5, 10, "lt") is True
        assert manager._check_threshold(15, 10, "gt") is True
        assert manager._check_threshold(10, 10, "lte") is True
        assert manager._check_threshold(10, 10, "gte") is True
        assert manager._check_threshold(10, 10, "eq") is True
        assert manager._check_threshold(5, 10, "neq") is True
        assert manager._check_threshold(10, 10, "invalid") is False

    def test_cooldown_period(self, manager, alert_config):
        """Test cooldown period between alerts."""
        manager.add_alert_config(alert_config)

        # First trigger
        triggered1 = manager.check_and_trigger("accuracy", 0.80)
        assert len(triggered1) == 1

        # Immediate second trigger should be blocked by cooldown
        triggered2 = manager.check_and_trigger("accuracy", 0.80)
        assert len(triggered2) == 0

    def test_disabled_alert(self, manager):
        """Test disabled alert is not triggered."""
        config = AlertConfig(
            alert_id="disabled",
            name="Disabled Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
            enabled=False,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
        )
        manager.add_alert_config(config)
        triggered = manager.check_and_trigger("accuracy", 0.80)
        assert len(triggered) == 0

    def test_trigger_manual_alert(self, manager):
        """Test manually triggering an alert."""
        alert = manager.trigger_alert(
            title="Manual Alert",
            message="This is a manual alert",
            level=AlertLevel.ERROR,
            model_name="test_model",
        )
        assert isinstance(alert, Alert)
        assert alert.title == "Manual Alert"
        assert alert.level == AlertLevel.ERROR

    def test_get_alerts(self, manager, alert_config):
        """Test getting alerts."""
        manager.add_alert_config(alert_config)
        manager.check_and_trigger("accuracy", 0.80)
        manager.check_and_trigger("accuracy", 0.75)

        # Need to wait for cooldown or adjust timestamp
        alerts = manager.get_alerts()
        assert len(alerts) >= 1

    def test_get_alerts_filtered(self, manager):
        """Test filtering alerts."""
        manager.trigger_alert("Warning", "msg", AlertLevel.WARNING)
        manager.trigger_alert("Error", "msg", AlertLevel.ERROR)

        warning_alerts = manager.get_alerts(level=AlertLevel.WARNING)
        assert all(a.level == AlertLevel.WARNING for a in warning_alerts)

    def test_acknowledge_alert(self, manager):
        """Test acknowledging an alert."""
        alert = manager.trigger_alert("Test", "message")
        result = manager.acknowledge_alert(alert.alert_id, "user@test.com")
        assert result is True

        # Verify it was acknowledged
        alerts = manager.get_alerts()
        acknowledged = [a for a in alerts if a.acknowledged]
        assert len(acknowledged) == 1

    def test_acknowledge_nonexistent_alert(self, manager):
        """Test acknowledging nonexistent alert."""
        result = manager.acknowledge_alert("nonexistent", "user@test.com")
        assert result is False

    def test_resolve_alert(self, manager):
        """Test resolving an alert."""
        alert = manager.trigger_alert("Test", "message")
        result = manager.resolve_alert(alert.alert_id)
        assert result is True

        # Verify it was resolved
        resolved_alerts = manager.get_alerts(resolved=True)
        assert len(resolved_alerts) == 1

    def test_resolve_nonexistent_alert(self, manager):
        """Test resolving nonexistent alert."""
        result = manager.resolve_alert("nonexistent")
        assert result is False

    def test_get_unresolved_count(self, manager):
        """Test counting unresolved alerts."""
        manager.trigger_alert("Warning 1", "msg", AlertLevel.WARNING)
        manager.trigger_alert("Warning 2", "msg", AlertLevel.WARNING)
        manager.trigger_alert("Error 1", "msg", AlertLevel.ERROR)

        counts = manager.get_unresolved_count()
        assert counts.get("warning", 0) == 2
        assert counts.get("error", 0) == 1

    def test_max_alerts_limit(self, manager, alert_config):
        """Test max alerts limit is enforced via check_and_trigger."""
        manager._max_alerts = 5
        manager.add_alert_config(alert_config)
        # Disable cooldown for this test
        alert_config.cooldown_minutes = 0

        for i in range(10):
            # Force trigger via check_and_trigger (which has the limit check)
            manager._last_alert_time.clear()
            manager.check_and_trigger("accuracy", 0.80)

        assert len(manager._alerts) <= 5


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.WARNING == "warning"
        assert HealthStatus.CRITICAL == "critical"
        assert HealthStatus.UNKNOWN == "unknown"
