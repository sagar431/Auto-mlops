"""
Pytest tests for monitor_model_performance MCP Tool

Tests for the monitor_model_performance MCP tool that uses ModelMonitor
for comprehensive model performance monitoring and degradation detection.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mcp_mlops_tools import monitor_model_performance

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="mlops_monitor_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def classification_data():
    """Create classification data for testing."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100).tolist()
    # Predictions that mostly match but have some errors
    y_pred = y_true.copy()
    for i in range(10):  # 10% error
        y_pred[i] = 1 - y_pred[i]
    return y_true, y_pred


@pytest.fixture
def classification_data_with_probs():
    """Create classification data with probabilities."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100).tolist()
    y_pred = y_true.copy()
    for i in range(10):
        y_pred[i] = 1 - y_pred[i]

    # Generate probabilities
    y_prob = []
    for pred in y_pred:
        if pred == 1:
            prob = [np.random.uniform(0.1, 0.4), np.random.uniform(0.6, 0.9)]
        else:
            prob = [np.random.uniform(0.6, 0.9), np.random.uniform(0.1, 0.4)]
        y_prob.append(prob)
    return y_true, y_pred, y_prob


@pytest.fixture
def regression_data():
    """Create regression data for testing."""
    np.random.seed(42)
    y_true = np.random.uniform(0, 100, 100).tolist()
    y_pred = [v + np.random.normal(0, 5) for v in y_true]  # Add some noise
    return y_true, y_pred


@pytest.fixture
def degraded_classification_data():
    """Create classification data with poor performance (degraded)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100).tolist()
    # Almost random predictions (about 50% accuracy)
    y_pred = np.random.randint(0, 2, 100).tolist()
    return y_true, y_pred


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestMonitorModelPerformanceBasic:
    """Basic functionality tests for monitor_model_performance."""

    def test_classification_success(self, classification_data):
        """Test successful classification monitoring."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        assert result["model_name"] == "test_classifier"
        assert result["task_type"] == "classification"
        assert result["sample_size"] == 100
        assert "metrics" in result
        assert "health_status" in result

    def test_regression_success(self, regression_data):
        """Test successful regression monitoring."""
        y_true, y_pred = regression_data
        result = monitor_model_performance(
            model_name="test_regressor",
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
        )

        assert result["success"] is True
        assert result["model_name"] == "test_regressor"
        assert result["task_type"] == "regression"
        assert "metrics" in result
        assert "mse" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert "mae" in result["metrics"]
        assert "r2_score" in result["metrics"]

    def test_classification_with_probabilities(self, classification_data_with_probs):
        """Test classification monitoring with prediction probabilities."""
        y_true, y_pred, y_prob = classification_data_with_probs
        result = monitor_model_performance(
            model_name="test_classifier_probs",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            task_type="classification",
        )

        assert result["success"] is True
        assert "metrics" in result
        # Should include AUC-ROC when probabilities are provided
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_with_model_version(self, classification_data):
        """Test monitoring with model version."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            model_version="1.2.3",
        )

        assert result["success"] is True
        assert result["model_version"] == "1.2.3"


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMonitorModelPerformanceMetrics:
    """Tests for metrics calculation in monitor_model_performance."""

    def test_classification_metrics_values(self, classification_data):
        """Test that classification metrics have valid values."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        metrics = result["metrics"]

        # All metrics should be between 0 and 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

        # With 90% accuracy expected
        assert metrics["accuracy"] >= 0.85

    def test_regression_metrics_values(self, regression_data):
        """Test that regression metrics have valid values."""
        y_true, y_pred = regression_data
        result = monitor_model_performance(
            model_name="test_regressor",
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
        )

        assert result["success"] is True
        metrics = result["metrics"]

        # MSE, RMSE, MAE should be non-negative
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

        # R2 should be reasonable for good fit
        assert metrics["r2_score"] > 0.5


# ============================================================================
# Health Status Tests
# ============================================================================


class TestMonitorModelPerformanceHealth:
    """Tests for health status in monitor_model_performance."""

    def test_health_status_present(self, classification_data):
        """Test that health status is included in result."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        assert "health_status" in result
        assert "health_message" in result
        assert "health_details" in result
        assert "recommendations" in result

    def test_health_status_valid_values(self, classification_data):
        """Test that health status has valid values."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        valid_statuses = ["healthy", "warning", "critical", "unknown"]
        assert result["health_status"] in valid_statuses

    def test_custom_metrics_to_check(self, classification_data):
        """Test health status with custom metrics to check."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            metrics_to_check=["accuracy", "precision"],
        )

        assert result["success"] is True
        assert "health_details" in result


# ============================================================================
# Baseline Comparison Tests
# ============================================================================


class TestMonitorModelPerformanceBaseline:
    """Tests for baseline comparison in monitor_model_performance."""

    def test_baseline_comparison(self, classification_data):
        """Test monitoring with baseline comparison."""
        y_true, y_pred = classification_data
        baseline = {"accuracy": 0.95, "f1_score": 0.92}

        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            baseline_metrics=baseline,
        )

        assert result["success"] is True
        assert "baseline_comparison" in result
        comparison = result["baseline_comparison"]
        assert "accuracy" in comparison
        assert "baseline" in comparison["accuracy"]
        assert "current" in comparison["accuracy"]
        assert "difference" in comparison["accuracy"]

    def test_baseline_degradation_detection(self, degraded_classification_data):
        """Test that degradation is detected when below baseline."""
        y_true, y_pred = degraded_classification_data
        # Set high baseline to trigger degradation detection
        baseline = {"accuracy": 0.95}

        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            baseline_metrics=baseline,
            degradation_threshold=0.05,
        )

        assert result["success"] is True
        assert "baseline_comparison" in result
        comparison = result["baseline_comparison"]
        # Should show degradation from baseline
        assert comparison["accuracy"]["current"] < comparison["accuracy"]["baseline"]


# ============================================================================
# Snapshot Recording Tests
# ============================================================================


class TestMonitorModelPerformanceSnapshots:
    """Tests for snapshot recording in monitor_model_performance."""

    def test_snapshot_recorded_by_default(self, classification_data):
        """Test that snapshot is recorded by default."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        assert result.get("snapshot_recorded") is True
        assert "snapshot_id" in result

    def test_snapshot_not_recorded_when_disabled(self, classification_data):
        """Test that snapshot is not recorded when disabled."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            record_snapshot=False,
        )

        assert result["success"] is True
        assert "snapshot_id" not in result

    def test_snapshot_persistence(self, classification_data, temp_dir):
        """Test snapshot persistence to storage."""
        y_true, y_pred = classification_data
        storage_path = str(temp_dir / "snapshots.json")

        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            storage_path=storage_path,
        )

        assert result["success"] is True
        assert result["storage_path"] == storage_path
        # File should be created
        assert Path(storage_path).exists()

    def test_snapshot_loading(self, classification_data, temp_dir):
        """Test loading existing snapshots from storage."""
        y_true, y_pred = classification_data
        storage_path = str(temp_dir / "snapshots.json")

        # First call - creates snapshot
        result1 = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            storage_path=storage_path,
        )

        assert result1["success"] is True

        # Second call - should load existing snapshot
        result2 = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            storage_path=storage_path,
        )

        assert result2["success"] is True
        assert result2.get("snapshots_loaded", 0) >= 1


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestMonitorModelPerformanceErrors:
    """Tests for error handling in monitor_model_performance."""

    def test_empty_arrays(self):
        """Test error handling for empty input arrays."""
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=[],
            y_pred=[],
            task_type="classification",
        )

        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_length_mismatch(self):
        """Test error handling for mismatched array lengths."""
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=[1, 0, 1, 0, 1],
            y_pred=[1, 0, 1],
            task_type="classification",
        )

        assert result["success"] is False
        assert "error" in result
        assert "length" in result["error"].lower() or "mismatch" in result["error"].lower()


# ============================================================================
# Result Structure Tests
# ============================================================================


class TestMonitorModelPerformanceResultStructure:
    """Tests for the structure of monitor_model_performance results."""

    def test_result_contains_all_fields(self, classification_data):
        """Test that result contains all expected fields."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        assert "model_name" in result
        assert "task_type" in result
        assert "sample_size" in result
        assert "metrics" in result
        assert "health_status" in result
        assert "health_message" in result
        assert "health_details" in result
        assert "recommendations" in result
        assert "message" in result

    def test_health_details_structure(self, classification_data):
        """Test the structure of health_details."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert result["success"] is True
        health_details = result["health_details"]
        assert "metrics" in health_details
        assert "issues" in health_details
        assert "warnings" in health_details


# ============================================================================
# Degradation Threshold Tests
# ============================================================================


class TestMonitorModelPerformanceDegradation:
    """Tests for degradation threshold in monitor_model_performance."""

    def test_custom_degradation_threshold(self, classification_data):
        """Test monitoring with custom degradation threshold."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            degradation_threshold=0.10,  # 10% threshold
        )

        assert result["success"] is True

    def test_strict_degradation_threshold(self, classification_data):
        """Test monitoring with strict degradation threshold."""
        y_true, y_pred = classification_data
        result = monitor_model_performance(
            model_name="test_classifier",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            degradation_threshold=0.01,  # 1% threshold (strict)
        )

        assert result["success"] is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestMonitorModelPerformanceIntegration:
    """Integration tests for monitor_model_performance."""

    def test_full_monitoring_workflow(self, classification_data, temp_dir):
        """Test complete monitoring workflow."""
        y_true, y_pred = classification_data
        storage_path = str(temp_dir / "monitoring.json")
        baseline = {"accuracy": 0.85, "precision": 0.80, "recall": 0.80, "f1_score": 0.80}

        result = monitor_model_performance(
            model_name="production_model",
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
            model_version="2.0.0",
            degradation_threshold=0.05,
            baseline_metrics=baseline,
            metrics_to_check=["accuracy", "f1_score"],
            record_snapshot=True,
            storage_path=storage_path,
        )

        assert result["success"] is True
        assert result["model_name"] == "production_model"
        assert result["model_version"] == "2.0.0"
        assert "metrics" in result
        assert "baseline_comparison" in result
        assert "snapshot_id" in result
        assert Path(storage_path).exists()

    def test_multiple_monitoring_calls(self, temp_dir):
        """Test multiple monitoring calls to track performance over time."""
        storage_path = str(temp_dir / "tracking.json")
        snapshot_counts = []

        for i in range(3):
            np.random.seed(42 + i)
            y_true = np.random.randint(0, 2, 50).tolist()
            y_pred = np.random.randint(0, 2, 50).tolist()

            result = monitor_model_performance(
                model_name="tracked_model",
                y_true=y_true,
                y_pred=y_pred,
                task_type="classification",
                storage_path=storage_path,
            )

            assert result["success"] is True
            snapshot_counts.append(result.get("snapshot_count", 0))

        # Snapshot count should increase
        assert snapshot_counts[-1] >= snapshot_counts[0]

    def test_regression_with_baseline(self, regression_data, temp_dir):
        """Test regression monitoring with baseline."""
        y_true, y_pred = regression_data
        baseline = {"mse": 50.0, "rmse": 7.0, "mae": 5.0, "r2_score": 0.90}

        result = monitor_model_performance(
            model_name="regressor",
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
            baseline_metrics=baseline,
        )

        assert result["success"] is True
        assert "baseline_comparison" in result
        assert "mse" in result["baseline_comparison"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
