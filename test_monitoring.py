"""
Tests for the Model Monitoring Module.

Tests for DriftDetector, ConceptDriftDetector, ModelMonitor, and AlertManager.
"""

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

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def reference_data():
    """Create reference dataset for drift detection."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def current_data_no_drift(reference_data):
    """Create current dataset with no drift."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def current_data_with_drift():
    """Create current dataset with significant drift."""
    np.random.seed(44)
    return pd.DataFrame(
        {
            "age": np.random.normal(50, 15, 1000),  # Shifted mean
            "income": np.random.normal(75000, 20000, 1000),  # Shifted mean
            "category": np.random.choice(["A", "D", "E"], 1000),  # Different categories
            "score": np.random.uniform(50, 150, 1000),  # Shifted range
        }
    )


@pytest.fixture
def classification_data():
    """Create data for classification metrics."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    # Predictions that mostly match but have some errors
    y_pred = y_true.copy()
    y_pred[:10] = 1 - y_pred[:10]  # 10% error
    y_prob = np.random.uniform(0.3, 0.7, 100)
    y_prob[y_pred == 1] = np.random.uniform(0.6, 0.9, (y_pred == 1).sum())
    y_prob[y_pred == 0] = np.random.uniform(0.1, 0.4, (y_pred == 0).sum())
    return y_true, y_pred, y_prob


@pytest.fixture
def regression_data():
    """Create data for regression metrics."""
    np.random.seed(42)
    y_true = np.random.uniform(0, 100, 100)
    y_pred = y_true + np.random.normal(0, 5, 100)  # Add some noise
    return y_true, y_pred


@pytest.fixture
def drift_detector():
    """Create a DriftDetector instance."""
    return DriftDetector(drift_threshold=0.05)


@pytest.fixture
def model_monitor():
    """Create a ModelMonitor instance."""
    return ModelMonitor(
        model_name="test_model",
        model_version="1.0.0",
        degradation_threshold=0.05,
    )


@pytest.fixture
def alert_manager():
    """Create an AlertManager instance."""
    return AlertManager()


# ============================================================================
# DriftDetector Tests
# ============================================================================


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_initialization(self):
        """Test DriftDetector initialization."""
        detector = DriftDetector()
        assert detector.drift_threshold == 0.1
        assert detector.stattest == "ks"

    def test_initialization_custom_params(self):
        """Test DriftDetector with custom parameters."""
        detector = DriftDetector(
            drift_threshold=0.05,
            stattest="psi",
            per_feature_stattest={"age": "ks"},
        )
        assert detector.drift_threshold == 0.05
        assert detector.stattest == "psi"
        assert detector.per_feature_stattest == {"age": "ks"}

    def test_detect_drift_no_drift(self, drift_detector, reference_data, current_data_no_drift):
        """Test drift detection when no drift exists."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
            dataset_name="test_dataset",
        )

        assert isinstance(report, DriftReport)
        assert report.dataset_name == "test_dataset"
        assert report.drift_type == DriftType.DATA
        assert report.reference_rows == 1000
        assert report.current_rows == 1000
        # With similar distributions, drift should be minimal
        assert len(report.feature_results) > 0

    def test_detect_drift_with_drift(self, drift_detector, reference_data, current_data_with_drift):
        """Test drift detection when drift exists."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            dataset_name="drifted_dataset",
        )

        assert isinstance(report, DriftReport)
        assert report.overall_drift_detected is True
        assert report.drift_share > 0
        assert report.severity != DriftSeverity.NONE

        # Check that some features show drift
        drifted_features = [f for f in report.feature_results if f.drift_detected]
        assert len(drifted_features) > 0

    def test_detect_drift_specific_columns(
        self, drift_detector, reference_data, current_data_with_drift
    ):
        """Test drift detection on specific columns."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            feature_columns=["age", "income"],
            dataset_name="specific_columns",
        )

        assert len(report.feature_results) == 2
        feature_names = {f.feature_name for f in report.feature_results}
        assert feature_names == {"age", "income"}

    def test_detect_drift_report_structure(
        self, drift_detector, reference_data, current_data_no_drift
    ):
        """Test that drift report has correct structure."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
        )

        assert report.report_id is not None
        assert isinstance(report.timestamp, datetime)
        assert 0 <= report.drift_share <= 1
        assert report.severity in DriftSeverity
        assert isinstance(report.recommendations, list)

    def test_feature_drift_result_structure(
        self, drift_detector, reference_data, current_data_no_drift
    ):
        """Test FeatureDriftResult structure."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
        )

        for feature in report.feature_results:
            assert isinstance(feature, FeatureDriftResult)
            assert feature.feature_name is not None
            assert isinstance(feature.drift_detected, bool)
            assert 0 <= feature.drift_score <= 1
            assert feature.stattest_name is not None

    def test_detect_prediction_drift(self, drift_detector):
        """Test prediction drift detection."""
        np.random.seed(42)
        ref_preds = np.random.normal(0.5, 0.1, 500)
        cur_preds = np.random.normal(0.7, 0.15, 500)  # Shifted predictions

        report = drift_detector.detect_prediction_drift(
            reference_predictions=ref_preds,
            current_predictions=cur_preds,
        )

        assert report.drift_type == DriftType.PREDICTION
        assert len(report.feature_results) == 1

    def test_severity_calculation(self):
        """Test drift severity calculation."""
        detector = DriftDetector()

        # Test different severity levels
        assert detector._calculate_severity(0, False) == DriftSeverity.NONE
        assert detector._calculate_severity(0.05, False) == DriftSeverity.LOW
        assert detector._calculate_severity(0.2, False) == DriftSeverity.MEDIUM
        assert detector._calculate_severity(0.4, True) == DriftSeverity.HIGH
        assert detector._calculate_severity(0.7, True) == DriftSeverity.CRITICAL

    def test_evidently_available_property(self):
        """Test evidently_available property."""
        detector = DriftDetector()
        # Property should return a boolean
        assert isinstance(detector.evidently_available, bool)

    def test_detect_drift_with_column_mapping(
        self, drift_detector, reference_data, current_data_with_drift
    ):
        """Test drift detection with explicit column type mapping."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            categorical_columns=["category"],
            numerical_columns=["age", "income", "score"],
            dataset_name="mapped_dataset",
        )

        assert isinstance(report, DriftReport)
        assert len(report.feature_results) == 4

    def test_detect_drift_with_empty_dataframes(self, drift_detector):
        """Test drift detection with empty dataframes raises error or handles gracefully."""
        empty_df = pd.DataFrame()
        data_df = pd.DataFrame({"a": [1, 2, 3]})

        # Test with empty reference
        try:
            drift_detector.detect_drift(
                reference_data=empty_df,
                current_data=data_df,
            )
        except (ValueError, KeyError):
            pass  # Expected behavior

    def test_detect_drift_recommendations(
        self, drift_detector, reference_data, current_data_with_drift
    ):
        """Test that recommendations are generated for drift reports."""
        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
        )

        # Should have recommendations when drift is detected
        assert isinstance(report.recommendations, list)
        if report.overall_drift_detected:
            assert len(report.recommendations) > 0

    def test_detect_drift_with_missing_columns(self, drift_detector, reference_data):
        """Test drift detection when current data has missing columns."""
        current_data = reference_data.drop(columns=["category"]).copy()

        report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            feature_columns=["age", "income", "category"],
        )

        # Should handle missing columns gracefully
        assert isinstance(report, DriftReport)

    def test_generate_recommendations_no_drift(self, drift_detector):
        """Test recommendation generation when no drift."""
        feature_results = [
            FeatureDriftResult(
                feature_name="test",
                drift_detected=False,
                drift_score=0.01,
                stattest_name="ks",
                stattest_threshold=0.05,
            )
        ]
        recommendations = drift_detector._generate_recommendations(
            feature_results, 0.0, DriftSeverity.NONE
        )
        assert "No significant drift detected" in recommendations[0]

    def test_generate_recommendations_high_drift(self, drift_detector):
        """Test recommendation generation for high drift."""
        feature_results = [
            FeatureDriftResult(
                feature_name="high_drift_feature",
                drift_detected=True,
                drift_score=0.8,
                stattest_name="ks",
                stattest_threshold=0.05,
            )
        ]
        recommendations = drift_detector._generate_recommendations(
            feature_results, 0.6, DriftSeverity.CRITICAL
        )
        assert len(recommendations) > 0
        # Should include retraining recommendation for critical drift
        assert any("retrain" in r.lower() for r in recommendations)


# ============================================================================
# ConceptDriftDetector Tests
# ============================================================================


class TestConceptDriftDetector:
    """Tests for ConceptDriftDetector class."""

    def test_initialization(self):
        """Test ConceptDriftDetector initialization."""
        detector = ConceptDriftDetector()
        assert detector.significance_level == 0.05

    def test_initialization_custom_params(self):
        """Test ConceptDriftDetector with custom parameters."""
        detector = ConceptDriftDetector(significance_level=0.01)
        assert detector.significance_level == 0.01

    def test_detect_concept_drift(self):
        """Test concept drift detection."""
        np.random.seed(42)
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "target": np.random.randint(0, 2, 500),
            }
        )
        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "target": np.random.randint(0, 2, 500),  # Same distribution
            }
        )

        detector = ConceptDriftDetector()
        report = detector.detect_concept_drift(
            reference_data=reference_data,
            current_data=current_data,
            target_column="target",
            dataset_name="concept_test",
        )

        assert isinstance(report, DriftReport)
        assert report.drift_type == DriftType.CONCEPT
        assert len(report.feature_results) > 0

    def test_detect_concept_drift_with_predictions(self):
        """Test concept drift detection with predictions."""
        np.random.seed(42)
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "target": np.random.randint(0, 2, 500),
                "prediction": np.random.randint(0, 2, 500),
            }
        )
        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "target": np.random.randint(0, 2, 500),
                "prediction": np.random.randint(0, 2, 500),
            }
        )

        detector = ConceptDriftDetector()
        report = detector.detect_concept_drift(
            reference_data=reference_data,
            current_data=current_data,
            target_column="target",
            prediction_column="prediction",
        )

        assert report.drift_type == DriftType.CONCEPT
        # Should have results for prediction_error and target
        assert len(report.feature_results) >= 1

    def test_concept_drift_missing_target(self):
        """Test error when target column is missing."""
        detector = ConceptDriftDetector()
        df = pd.DataFrame({"feature": [1, 2, 3]})

        with pytest.raises(ValueError, match="Target column"):
            detector.detect_concept_drift(
                reference_data=df,
                current_data=df,
                target_column="missing_column",
            )


# ============================================================================
# ModelMonitor Tests
# ============================================================================


class TestModelMonitor:
    """Tests for ModelMonitor class."""

    def test_initialization(self):
        """Test ModelMonitor initialization."""
        monitor = ModelMonitor(model_name="test_model")
        assert monitor.model_name == "test_model"
        assert monitor.degradation_threshold == 0.05

    def test_initialization_custom_params(self):
        """Test ModelMonitor with custom parameters."""
        baseline = ModelMetrics(accuracy=0.9)
        monitor = ModelMonitor(
            model_name="test_model",
            model_version="2.0.0",
            degradation_threshold=0.1,
            baseline_metrics=baseline,
        )
        assert monitor.model_version == "2.0.0"
        assert monitor.degradation_threshold == 0.1
        assert monitor.baseline_metrics.accuracy == 0.9

    def test_calculate_classification_metrics(self, model_monitor, classification_data):
        """Test classification metrics calculation."""
        y_true, y_pred, y_prob = classification_data

        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            task_type="classification",
        )

        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy is not None
        assert 0 <= metrics.accuracy <= 1
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None

    def test_calculate_regression_metrics(self, model_monitor, regression_data):
        """Test regression metrics calculation."""
        y_true, y_pred = regression_data

        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
        )

        assert isinstance(metrics, ModelMetrics)
        assert metrics.mse is not None
        assert metrics.mse >= 0
        assert metrics.rmse is not None
        assert metrics.mae is not None
        assert metrics.r2_score is not None

    def test_record_snapshot(self, model_monitor, classification_data):
        """Test recording performance snapshots."""
        y_true, y_pred, y_prob = classification_data
        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        snapshot = model_monitor.record_snapshot(
            metrics=metrics,
            sample_size=100,
            metadata={"batch_id": "batch_001"},
        )

        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.model_name == "test_model"
        assert snapshot.sample_size == 100
        assert snapshot.metadata["batch_id"] == "batch_001"

    def test_get_snapshots(self, model_monitor, classification_data):
        """Test retrieving snapshots."""
        y_true, y_pred, _ = classification_data
        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        # Record multiple snapshots
        for i in range(5):
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        snapshots = model_monitor.get_snapshots()
        assert len(snapshots) == 5

        # Test with limit
        limited = model_monitor.get_snapshots(limit=3)
        assert len(limited) == 3

    def test_get_snapshots_time_range(self, model_monitor, classification_data):
        """Test retrieving snapshots within time range."""
        y_true, y_pred, _ = classification_data
        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        now = datetime.utcnow()
        past = now - timedelta(days=2)

        model_monitor.record_snapshot(metrics=metrics, sample_size=100, timestamp=past)
        model_monitor.record_snapshot(metrics=metrics, sample_size=100, timestamp=now)

        # Get all snapshots
        all_snapshots = model_monitor.get_snapshots()
        assert len(all_snapshots) == 2

        # Get only recent
        recent = model_monitor.get_snapshots(start_time=now - timedelta(hours=1))
        assert len(recent) == 1

    def test_get_performance_trend(self, model_monitor):
        """Test performance trend analysis."""
        # Record snapshots with varying accuracy
        for i in range(10):
            metrics = ModelMetrics(accuracy=0.9 - i * 0.01)
            timestamp = datetime.utcnow() - timedelta(days=9 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        trend = model_monitor.get_performance_trend("accuracy", days=10)

        assert isinstance(trend, PerformanceTrend)
        assert trend.model_name == "test_model"
        assert trend.metric_name == "accuracy"
        assert trend.baseline_value is not None
        assert trend.current_value is not None
        assert trend.trend_direction in ["improving", "declining", "stable"]

    def test_check_degradation(self, model_monitor):
        """Test degradation detection."""
        # Record declining performance
        for i in range(10):
            metrics = ModelMetrics(accuracy=0.95 - i * 0.02)
            timestamp = datetime.utcnow() - timedelta(days=9 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        degraded, trend = model_monitor.check_degradation("accuracy", days=10)

        assert degraded is True
        assert trend.degradation_detected is True
        assert trend.change_percentage < 0

    def test_compare_to_baseline(self, model_monitor, classification_data):
        """Test comparison to baseline metrics."""
        y_true, y_pred, _ = classification_data

        # Set baseline
        baseline = ModelMetrics(accuracy=0.95, precision=0.93, recall=0.92)
        model_monitor.set_baseline(baseline)

        # Calculate current metrics
        current = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        comparison = model_monitor.compare_to_baseline(current)

        assert "accuracy" in comparison
        assert "baseline" in comparison["accuracy"]
        assert "current" in comparison["accuracy"]
        assert "difference" in comparison["accuracy"]
        assert "degraded" in comparison["accuracy"]

    def test_get_latest_metrics(self, model_monitor, classification_data):
        """Test getting latest metrics."""
        y_true, y_pred, _ = classification_data
        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )
        model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        latest = model_monitor.get_latest_metrics()
        assert latest is not None
        assert latest.accuracy == metrics.accuracy

    def test_get_summary(self, model_monitor, classification_data):
        """Test getting performance summary."""
        y_true, y_pred, _ = classification_data
        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        for _ in range(5):
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        summary = model_monitor.get_summary(days=7)

        assert summary["model_name"] == "test_model"
        assert summary["snapshot_count"] == 5
        assert "metrics" in summary
        assert "accuracy" in summary["metrics"]


# ============================================================================
# AlertManager Tests
# ============================================================================


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_initialization(self):
        """Test AlertManager initialization."""
        manager = AlertManager()
        assert manager._configs == {}
        assert manager._alerts == []

    def test_add_alert_config(self, alert_manager):
        """Test adding alert configuration."""
        config = AlertConfig(
            alert_id="test_alert",
            name="Test Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
        )

        alert_manager.add_alert_config(config)
        configs = alert_manager.get_alert_configs()

        assert len(configs) == 1
        assert configs[0].alert_id == "test_alert"

    def test_remove_alert_config(self, alert_manager):
        """Test removing alert configuration."""
        config = AlertConfig(
            alert_id="test_alert",
            name="Test Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
        )

        alert_manager.add_alert_config(config)
        assert len(alert_manager.get_alert_configs()) == 1

        result = alert_manager.remove_alert_config("test_alert")
        assert result is True
        assert len(alert_manager.get_alert_configs()) == 0

        # Remove non-existent
        result = alert_manager.remove_alert_config("non_existent")
        assert result is False

    def test_check_and_trigger(self, alert_manager):
        """Test alert checking and triggering."""
        config = AlertConfig(
            alert_id="accuracy_alert",
            name="Accuracy Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
            cooldown_minutes=0,  # No cooldown for testing
        )
        alert_manager.add_alert_config(config)

        # Should trigger (0.80 < 0.85)
        alerts = alert_manager.check_and_trigger(
            metric_name="accuracy",
            metric_value=0.80,
            model_name="test_model",
        )

        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert alerts[0].metric_value == 0.80

    def test_check_and_trigger_no_trigger(self, alert_manager):
        """Test alert not triggering when threshold not breached."""
        config = AlertConfig(
            alert_id="accuracy_alert",
            name="Accuracy Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
        )
        alert_manager.add_alert_config(config)

        # Should not trigger (0.90 >= 0.85)
        alerts = alert_manager.check_and_trigger(
            metric_name="accuracy",
            metric_value=0.90,
        )

        assert len(alerts) == 0

    def test_check_and_trigger_cooldown(self, alert_manager):
        """Test alert cooldown."""
        config = AlertConfig(
            alert_id="accuracy_alert",
            name="Accuracy Alert",
            channel=AlertChannel.LOG,
            level=AlertLevel.WARNING,
            metric_name="accuracy",
            threshold=0.85,
            comparison="lt",
            cooldown_minutes=60,
        )
        alert_manager.add_alert_config(config)

        # First trigger
        alerts1 = alert_manager.check_and_trigger(
            metric_name="accuracy",
            metric_value=0.80,
        )
        assert len(alerts1) == 1

        # Second trigger should be blocked by cooldown
        alerts2 = alert_manager.check_and_trigger(
            metric_name="accuracy",
            metric_value=0.75,
        )
        assert len(alerts2) == 0

    def test_trigger_alert_manually(self, alert_manager):
        """Test manual alert triggering."""
        alert = alert_manager.trigger_alert(
            title="Manual Alert",
            message="This is a manual alert",
            level=AlertLevel.ERROR,
            model_name="test_model",
            metadata={"source": "test"},
        )

        assert isinstance(alert, Alert)
        assert alert.title == "Manual Alert"
        assert alert.level == AlertLevel.ERROR
        assert alert.model_name == "test_model"

    def test_get_alerts(self, alert_manager):
        """Test getting alerts with filters."""
        # Create some alerts
        alert_manager.trigger_alert("Alert 1", "Message 1", AlertLevel.INFO)
        alert_manager.trigger_alert("Alert 2", "Message 2", AlertLevel.WARNING)
        alert_manager.trigger_alert("Alert 3", "Message 3", AlertLevel.ERROR)

        # Get all alerts
        all_alerts = alert_manager.get_alerts()
        assert len(all_alerts) == 3

        # Filter by level
        warnings = alert_manager.get_alerts(level=AlertLevel.WARNING)
        assert len(warnings) == 1

        # Filter by resolved
        unresolved = alert_manager.get_alerts(resolved=False)
        assert len(unresolved) == 3

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        alert = alert_manager.trigger_alert("Test", "Test message")
        alert_id = alert.alert_id

        result = alert_manager.acknowledge_alert(alert_id, "user@example.com")
        assert result is True

        alerts = alert_manager.get_alerts()
        assert alerts[0].acknowledged is True
        assert alerts[0].acknowledged_by == "user@example.com"

    def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        alert = alert_manager.trigger_alert("Test", "Test message")
        alert_id = alert.alert_id

        result = alert_manager.resolve_alert(alert_id)
        assert result is True

        alerts = alert_manager.get_alerts()
        assert alerts[0].resolved is True
        assert alerts[0].resolved_at is not None

    def test_get_unresolved_count(self, alert_manager):
        """Test getting unresolved alert count."""
        alert_manager.trigger_alert("Alert 1", "Message 1", AlertLevel.WARNING)
        alert_manager.trigger_alert("Alert 2", "Message 2", AlertLevel.WARNING)
        alert = alert_manager.trigger_alert("Alert 3", "Message 3", AlertLevel.ERROR)

        alert_manager.resolve_alert(alert.alert_id)

        counts = alert_manager.get_unresolved_count()
        assert counts.get("warning") == 2
        assert counts.get("error", 0) == 0


# ============================================================================
# Model Tests
# ============================================================================


class TestModels:
    """Tests for Pydantic models."""

    def test_model_metrics_defaults(self):
        """Test ModelMetrics default values."""
        metrics = ModelMetrics()
        assert metrics.accuracy is None
        assert metrics.custom_metrics == {}

    def test_model_metrics_with_values(self):
        """Test ModelMetrics with values."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.91,
            f1_score=0.92,
            custom_metrics={"weighted_f1": 0.925},
        )
        assert metrics.accuracy == 0.95
        assert metrics.custom_metrics["weighted_f1"] == 0.925

    def test_alert_config_validation(self):
        """Test AlertConfig validation."""
        config = AlertConfig(
            alert_id="test",
            name="Test",
            channel=AlertChannel.SLACK,
            level=AlertLevel.WARNING,
            threshold=0.5,
            comparison="lt",
            cooldown_minutes=30,
        )
        assert config.cooldown_minutes == 30
        assert config.enabled is True

    def test_drift_report_structure(self):
        """Test DriftReport structure."""
        feature_result = FeatureDriftResult(
            feature_name="age",
            drift_detected=True,
            drift_score=0.45,
            stattest_name="ks",
            stattest_threshold=0.05,
        )

        report = DriftReport(
            report_id="test-123",
            dataset_name="test_data",
            drift_type=DriftType.DATA,
            overall_drift_detected=True,
            drift_share=0.5,
            severity=DriftSeverity.MEDIUM,
            feature_results=[feature_result],
            reference_rows=1000,
            current_rows=1000,
        )

        assert report.report_id == "test-123"
        assert len(report.feature_results) == 1
        assert report.severity == DriftSeverity.MEDIUM

    def test_monitoring_config(self):
        """Test MonitoringConfig structure."""
        config = MonitoringConfig(
            model_name="test_model",
            model_version="1.0.0",
            drift_threshold=0.15,
            degradation_threshold=0.1,
            check_interval_minutes=30,
            enabled_checks=["drift", "performance", "alerts"],
        )

        assert config.model_name == "test_model"
        assert config.drift_threshold == 0.15
        assert "drift" in config.enabled_checks


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the monitoring module."""

    def test_full_monitoring_workflow(self, reference_data, current_data_with_drift):
        """Test complete monitoring workflow."""
        # Initialize components
        drift_detector = DriftDetector()
        model_monitor = ModelMonitor(model_name="integration_model")
        alert_manager = AlertManager()

        # Configure alerts
        alert_manager.add_alert_config(
            AlertConfig(
                alert_id="drift_alert",
                name="Drift Alert",
                channel=AlertChannel.LOG,
                level=AlertLevel.WARNING,
                metric_name="drift_share",
                threshold=0.3,
                comparison="gt",
                cooldown_minutes=0,
            )
        )

        # Detect drift
        drift_report = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
        )

        # Check for drift alerts
        if drift_report.drift_share > 0:
            alert_manager.check_and_trigger(
                metric_name="drift_share",
                metric_value=drift_report.drift_share,
                model_name="integration_model",
            )

        # Record model metrics
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        metrics = model_monitor.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )
        model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        # Get summary
        summary = model_monitor.get_summary()

        assert drift_report is not None
        assert summary["snapshot_count"] == 1

    def test_monitoring_with_sklearn_dependencies(self):
        """Test that sklearn integration works."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        # Generate synthetic data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Monitor performance
        monitor = ModelMonitor(model_name="logistic_regression")
        metrics = monitor.calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            task_type="classification",
        )

        assert metrics.accuracy is not None
        assert metrics.auc_roc is not None
        assert 0 <= metrics.accuracy <= 1

    def test_evidently_integration(self, reference_data, current_data_with_drift):
        """Test Evidently library integration for drift detection."""
        detector = DriftDetector(drift_threshold=0.05)

        # Skip if Evidently is not available
        if not detector.evidently_available:
            pytest.skip("Evidently not available")

        report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            dataset_name="evidently_test",
        )

        # Verify Evidently report structure
        assert isinstance(report, DriftReport)
        assert report.report_id is not None
        assert report.drift_type == DriftType.DATA

        # With shifted data, drift should be detected
        assert report.overall_drift_detected is True
        assert len(report.feature_results) > 0

    def test_fallback_drift_detection(self, reference_data, current_data_with_drift):
        """Test fallback drift detection without Evidently."""
        detector = DriftDetector()

        # Force fallback by directly calling the fallback method
        report = detector._detect_drift_fallback(
            reference_data=reference_data,
            current_data=current_data_with_drift,
            feature_columns=None,
            dataset_name="fallback_test",
        )

        assert isinstance(report, DriftReport)
        assert report.dataset_name == "fallback_test"
        assert len(report.feature_results) > 0

    def test_categorical_drift_detection(self):
        """Test drift detection for categorical features."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "category": np.random.choice(["A", "B", "C"], 500, p=[0.5, 0.3, 0.2]),
                "status": np.random.choice(["active", "inactive"], 500),
            }
        )
        # Current data with shifted distribution
        current = pd.DataFrame(
            {
                "category": np.random.choice(["A", "B", "C"], 500, p=[0.1, 0.2, 0.7]),
                "status": np.random.choice(["active", "pending"], 500),  # New category
            }
        )

        detector = DriftDetector(drift_threshold=0.05)
        report = detector.detect_drift(
            reference_data=reference,
            current_data=current,
            categorical_columns=["category", "status"],
        )

        assert isinstance(report, DriftReport)
        # Should detect drift in category distribution
        category_result = next(
            (f for f in report.feature_results if f.feature_name == "category"), None
        )
        if category_result:
            assert category_result.stattest_name in ["chisquare", "psi", "jensenshannon"]

    def test_drift_detection_with_nan_values(self):
        """Test drift detection handles NaN values correctly."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0] * 100,
                "feature2": np.random.normal(0, 1, 500),
            }
        )
        current = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0, 4.0, np.nan] * 100,
                "feature2": np.random.normal(0.5, 1, 500),
            }
        )

        detector = DriftDetector()
        report = detector.detect_drift(
            reference_data=reference,
            current_data=current,
        )

        assert isinstance(report, DriftReport)
        assert len(report.feature_results) == 2

    def test_drift_report_serialization(self, reference_data, current_data_no_drift):
        """Test that drift reports can be serialized to JSON."""
        detector = DriftDetector()
        report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_no_drift,
        )

        # Pydantic models should be serializable
        json_str = report.model_dump_json()
        assert isinstance(json_str, str)
        assert "report_id" in json_str
        assert "drift_type" in json_str

    def test_multiple_drift_checks(self, reference_data):
        """Test running multiple drift checks on the same detector."""
        detector = DriftDetector()

        # Create multiple current datasets with varying drift
        np.random.seed(42)
        current1 = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 500),  # No drift
                "income": np.random.normal(50000, 15000, 500),
                "category": np.random.choice(["A", "B", "C"], 500),
                "score": np.random.uniform(0, 100, 500),
            }
        )
        current2 = pd.DataFrame(
            {
                "age": np.random.normal(60, 10, 500),  # High drift
                "income": np.random.normal(80000, 15000, 500),
                "category": np.random.choice(["X", "Y", "Z"], 500),
                "score": np.random.uniform(50, 150, 500),
            }
        )

        report1 = detector.detect_drift(
            reference_data=reference_data, current_data=current1, dataset_name="check1"
        )
        report2 = detector.detect_drift(
            reference_data=reference_data, current_data=current2, dataset_name="check2"
        )

        # report2 should show more drift than report1
        assert report2.drift_share >= report1.drift_share
        assert report1.dataset_name == "check1"
        assert report2.dataset_name == "check2"


# ============================================================================
# New ModelMonitor Feature Tests
# ============================================================================


class TestModelMonitorNewFeatures:
    """Tests for new ModelMonitor features: moving average, percentiles, health check, etc."""

    def test_get_moving_average(self, model_monitor):
        """Test moving average calculation."""
        # Record 10 snapshots with increasing accuracy
        for i in range(10):
            metrics = ModelMetrics(accuracy=0.80 + i * 0.01)
            timestamp = datetime.utcnow() - timedelta(days=9 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        ma_results = model_monitor.get_moving_average("accuracy", window_size=3)

        assert len(ma_results) == 8  # 10 - (3 - 1)
        # Moving average should smooth the values
        for result in ma_results:
            assert "timestamp" in result
            assert "moving_average" in result
            assert "raw_value" in result
            assert 0 <= result["moving_average"] <= 1

    def test_get_moving_average_insufficient_data(self, model_monitor):
        """Test moving average with insufficient data."""
        # Only record 2 snapshots
        for i in range(2):
            metrics = ModelMetrics(accuracy=0.85)
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        ma_results = model_monitor.get_moving_average("accuracy", window_size=5)
        assert len(ma_results) == 0

    def test_get_percentiles(self, model_monitor):
        """Test percentile calculation."""
        # Record snapshots with varying accuracy
        np.random.seed(42)
        for _ in range(50):
            metrics = ModelMetrics(accuracy=np.random.uniform(0.75, 0.95))
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        percentiles = model_monitor.get_percentiles("accuracy")

        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles

        # p25 < p50 < p75 < p90 < p95
        assert percentiles["p25"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p75"]
        assert percentiles["p75"] <= percentiles["p90"]
        assert percentiles["p90"] <= percentiles["p95"]

    def test_get_percentiles_custom(self, model_monitor):
        """Test percentile calculation with custom percentiles."""
        for _ in range(20):
            metrics = ModelMetrics(accuracy=np.random.uniform(0.8, 0.9))
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        percentiles = model_monitor.get_percentiles("accuracy", percentiles=[10, 50, 99])

        assert "p10" in percentiles
        assert "p50" in percentiles
        assert "p99" in percentiles
        assert "p25" not in percentiles

    def test_get_percentiles_no_data(self, model_monitor):
        """Test percentile calculation with no data."""
        percentiles = model_monitor.get_percentiles("accuracy")

        assert percentiles["p25"] is None
        assert percentiles["p50"] is None

    def test_get_health_status_healthy(self, model_monitor):
        """Test health status when model is healthy."""
        # Record stable, good performance
        for i in range(10):
            metrics = ModelMetrics(accuracy=0.92, f1_score=0.90, precision=0.91, recall=0.89)
            timestamp = datetime.utcnow() - timedelta(days=9 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        health = model_monitor.get_health_status()

        assert health["status"] == HealthStatus.HEALTHY
        assert "All monitored metrics" in health["message"]
        assert len(health["issues"]) == 0
        assert len(health["warnings"]) == 0

    def test_get_health_status_critical(self, model_monitor):
        """Test health status when model has critical degradation."""
        # Record declining performance
        for i in range(10):
            metrics = ModelMetrics(accuracy=0.95 - i * 0.03)  # From 0.95 to 0.68
            timestamp = datetime.utcnow() - timedelta(days=9 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        health = model_monitor.get_health_status()

        assert health["status"] == HealthStatus.CRITICAL
        assert len(health["issues"]) > 0
        assert len(health["recommendations"]) > 0

    def test_get_health_status_no_data(self, model_monitor):
        """Test health status with no data."""
        health = model_monitor.get_health_status()

        assert health["status"] == HealthStatus.UNKNOWN
        assert "No performance data" in health["message"]

    def test_compare_versions(self):
        """Test version comparison between two monitors."""
        monitor_v1 = ModelMonitor(model_name="test_model", model_version="1.0.0")
        monitor_v2 = ModelMonitor(model_name="test_model", model_version="2.0.0")

        # V1 metrics
        metrics_v1 = ModelMetrics(accuracy=0.85, precision=0.84, recall=0.83, f1_score=0.835)
        monitor_v1.record_snapshot(metrics=metrics_v1, sample_size=100)

        # V2 metrics (better)
        metrics_v2 = ModelMetrics(accuracy=0.90, precision=0.89, recall=0.88, f1_score=0.885)
        monitor_v2.record_snapshot(metrics=metrics_v2, sample_size=100)

        comparison = monitor_v2.compare_versions(monitor_v1)

        assert comparison["comparison_valid"] is True
        assert comparison["this_model"]["version"] == "2.0.0"
        assert comparison["other_model"]["version"] == "1.0.0"
        assert "accuracy" in comparison["metrics"]
        assert comparison["metrics"]["accuracy"]["this_is_better"] is True
        assert comparison["summary"]["this_better_count"] > 0

    def test_compare_versions_no_data(self):
        """Test version comparison when one monitor has no data."""
        monitor_v1 = ModelMonitor(model_name="test_model", model_version="1.0.0")
        monitor_v2 = ModelMonitor(model_name="test_model", model_version="2.0.0")

        # Only v1 has data
        metrics_v1 = ModelMetrics(accuracy=0.85)
        monitor_v1.record_snapshot(metrics=metrics_v1, sample_size=100)

        comparison = monitor_v2.compare_versions(monitor_v1)

        assert comparison["comparison_valid"] is False

    def test_save_and_load_snapshots(self, model_monitor, tmp_path):
        """Test saving and loading snapshots to/from disk."""
        # Record some snapshots
        for i in range(5):
            metrics = ModelMetrics(accuracy=0.85 + i * 0.01, precision=0.84 + i * 0.01)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100 + i,
                metadata={"batch": i},
            )

        # Save
        save_path = tmp_path / "snapshots.json"
        result = model_monitor.save_snapshots(save_path)
        assert result is True
        assert save_path.exists()

        # Create new monitor and load
        new_monitor = ModelMonitor(model_name="test_model")
        loaded_count = new_monitor.load_snapshots(save_path)

        assert loaded_count == 5
        assert len(new_monitor._snapshots) == 5
        assert new_monitor._snapshots[0].sample_size == 100
        assert new_monitor._snapshots[4].sample_size == 104

    def test_load_snapshots_nonexistent(self, model_monitor, tmp_path):
        """Test loading from nonexistent file."""
        result = model_monitor.load_snapshots(tmp_path / "nonexistent.json")
        assert result == 0

    def test_clear_snapshots(self, model_monitor):
        """Test clearing all snapshots."""
        # Record some snapshots
        for _ in range(5):
            metrics = ModelMetrics(accuracy=0.85)
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        assert len(model_monitor._snapshots) == 5

        cleared = model_monitor.clear_snapshots()

        assert cleared == 5
        assert len(model_monitor._snapshots) == 0

    def test_get_metric_statistics(self, model_monitor):
        """Test comprehensive metric statistics."""
        np.random.seed(42)
        for _ in range(30):
            metrics = ModelMetrics(accuracy=np.random.uniform(0.80, 0.95))
            model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        stats = model_monitor.get_metric_statistics("accuracy")

        assert stats["metric_name"] == "accuracy"
        assert stats["count"] == 30
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "p25" in stats
        assert "p75" in stats
        assert "p95" in stats
        assert "variance" in stats

        # Validate ranges
        assert stats["min"] <= stats["p25"] <= stats["median"]
        assert stats["median"] <= stats["p75"] <= stats["max"]

    def test_get_metric_statistics_no_data(self, model_monitor):
        """Test metric statistics with no data."""
        stats = model_monitor.get_metric_statistics("accuracy")

        assert stats["count"] == 0
        assert "No data available" in stats["message"]

    def test_get_moving_average_with_days_filter(self, model_monitor):
        """Test moving average with days filter."""
        # Record snapshots over 20 days
        for i in range(20):
            metrics = ModelMetrics(accuracy=0.80 + (i % 5) * 0.02)
            timestamp = datetime.utcnow() - timedelta(days=19 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        # Get MA for last 7 days only
        ma_results = model_monitor.get_moving_average("accuracy", window_size=3, days=7)

        # Should only include snapshots from last 7 days
        assert len(ma_results) <= 7

    def test_health_status_with_custom_metrics(self, model_monitor):
        """Test health status with custom metrics list."""
        for i in range(10):
            metrics = ModelMetrics(
                accuracy=0.90,
                mse=0.01 + i * 0.005,  # Increasing MSE (bad)
                r2_score=0.95 - i * 0.02,  # Decreasing R2 (bad)
            )
            timestamp = datetime.utcnow() - timedelta(days=9 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        # Check only accuracy (should be healthy)
        health_acc = model_monitor.get_health_status(metrics_to_check=["accuracy"])
        assert health_acc["status"] == HealthStatus.HEALTHY

    def test_compare_versions_error_metrics(self):
        """Test version comparison for error metrics (lower is better)."""
        monitor_v1 = ModelMonitor(model_name="regression_model", model_version="1.0.0")
        monitor_v2 = ModelMonitor(model_name="regression_model", model_version="2.0.0")

        # V1 metrics (higher error)
        metrics_v1 = ModelMetrics(mse=0.05, rmse=0.22, mae=0.15)
        monitor_v1.record_snapshot(metrics=metrics_v1, sample_size=100)

        # V2 metrics (lower error = better)
        metrics_v2 = ModelMetrics(mse=0.02, rmse=0.14, mae=0.10)
        monitor_v2.record_snapshot(metrics=metrics_v2, sample_size=100)

        comparison = monitor_v2.compare_versions(monitor_v1)

        assert comparison["metrics"]["mse"]["this_is_better"] is True
        assert comparison["metrics"]["rmse"]["this_is_better"] is True
        assert comparison["metrics"]["mae"]["this_is_better"] is True

    def test_save_snapshots_creates_directory(self, model_monitor, tmp_path):
        """Test that save_snapshots creates parent directories."""
        metrics = ModelMetrics(accuracy=0.85)
        model_monitor.record_snapshot(metrics=metrics, sample_size=100)

        nested_path = tmp_path / "deep" / "nested" / "dir" / "snapshots.json"
        result = model_monitor.save_snapshots(nested_path)

        assert result is True
        assert nested_path.exists()

    def test_percentiles_with_days_filter(self, model_monitor):
        """Test percentiles with days filter."""
        # Record 30 days of data
        for i in range(30):
            metrics = ModelMetrics(accuracy=0.80 + (i / 30) * 0.15)  # 0.80 to 0.95
            timestamp = datetime.utcnow() - timedelta(days=29 - i)
            model_monitor.record_snapshot(
                metrics=metrics,
                sample_size=100,
                timestamp=timestamp,
            )

        # Get percentiles for all data
        all_percentiles = model_monitor.get_percentiles("accuracy")

        # Get percentiles for last 7 days only
        recent_percentiles = model_monitor.get_percentiles("accuracy", days=7)

        # Recent percentiles should generally be higher (later data has higher accuracy)
        assert recent_percentiles["p50"] >= all_percentiles["p50"] - 0.1
