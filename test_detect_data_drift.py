"""
Pytest tests for detect_data_drift MCP Tool

Tests for the detect_data_drift MCP tool that uses Evidently AI
for comprehensive drift detection between reference and current datasets.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcp_mlops_tools import detect_data_drift

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="mlops_drift_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reference_csv(temp_dir):
    """Create a reference CSV for drift detection (training data)."""
    csv_path = temp_dir / "reference.csv"
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 500),
            "income": np.random.normal(50000, 15000, 500),
            "category": np.random.choice(["A", "B", "C"], 500),
            "score": np.random.uniform(0, 100, 500),
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def current_no_drift_csv(temp_dir):
    """Create a current CSV with no significant drift."""
    csv_path = temp_dir / "current_no_drift.csv"
    np.random.seed(43)
    df = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 500),
            "income": np.random.normal(50000, 15000, 500),
            "category": np.random.choice(["A", "B", "C"], 500),
            "score": np.random.uniform(0, 100, 500),
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def current_with_drift_csv(temp_dir):
    """Create a current CSV with significant drift."""
    csv_path = temp_dir / "current_with_drift.csv"
    np.random.seed(44)
    df = pd.DataFrame(
        {
            "age": np.random.normal(50, 15, 500),  # Shifted mean
            "income": np.random.normal(75000, 20000, 500),  # Shifted mean
            "category": np.random.choice(["A", "D", "E"], 500),  # Different categories
            "score": np.random.uniform(50, 150, 500),  # Shifted range
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def reference_parquet(temp_dir):
    """Create a reference Parquet file for drift detection."""
    parquet_path = temp_dir / "reference.parquet"
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 300),
            "feature2": np.random.normal(100, 25, 300),
            "label": np.random.choice(["X", "Y", "Z"], 300),
        }
    )
    df.to_parquet(parquet_path)
    return str(parquet_path)


@pytest.fixture
def current_parquet(temp_dir):
    """Create a current Parquet file with drift."""
    parquet_path = temp_dir / "current.parquet"
    np.random.seed(45)
    df = pd.DataFrame(
        {
            "feature1": np.random.normal(2, 1.5, 300),  # Shifted
            "feature2": np.random.normal(150, 30, 300),  # Shifted
            "label": np.random.choice(["X", "Y", "W"], 300),  # Changed categories
        }
    )
    df.to_parquet(parquet_path)
    return str(parquet_path)


@pytest.fixture
def reference_json(temp_dir):
    """Create a reference JSON file for drift detection."""
    json_path = temp_dir / "reference.json"
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "value": np.random.normal(50, 10, 100),
            "count": np.random.randint(1, 100, 100),
        }
    )
    df.to_json(json_path)
    return str(json_path)


@pytest.fixture
def current_json(temp_dir):
    """Create a current JSON file with drift."""
    json_path = temp_dir / "current.json"
    np.random.seed(46)
    df = pd.DataFrame(
        {
            "value": np.random.normal(80, 20, 100),  # Shifted
            "count": np.random.randint(50, 200, 100),  # Shifted
        }
    )
    df.to_json(json_path)
    return str(json_path)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestDetectDataDriftBasic:
    """Basic functionality tests for detect_data_drift."""

    def test_detect_drift_csv_success(self, reference_csv, current_no_drift_csv):
        """Test successful drift detection on CSV files."""
        result = detect_data_drift(reference_csv, current_no_drift_csv)

        assert result["success"] is True
        assert "report_id" in result
        assert "overall_drift_detected" in result
        assert "drift_share" in result
        assert "severity" in result
        assert "feature_results" in result

    def test_detect_drift_with_drift(self, reference_csv, current_with_drift_csv):
        """Test drift detection when significant drift exists."""
        result = detect_data_drift(reference_csv, current_with_drift_csv)

        assert result["success"] is True
        assert result["overall_drift_detected"] is True
        assert result["drift_share"] > 0
        assert result["severity"] != "none"
        assert len(result["drifted_features"]) > 0

    def test_detect_drift_parquet(self, reference_parquet, current_parquet):
        """Test drift detection on Parquet files."""
        result = detect_data_drift(reference_parquet, current_parquet)

        assert result["success"] is True
        assert "overall_drift_detected" in result
        assert "feature_results" in result

    def test_detect_drift_json(self, reference_json, current_json):
        """Test drift detection on JSON files."""
        result = detect_data_drift(reference_json, current_json)

        assert result["success"] is True
        assert "overall_drift_detected" in result


# ============================================================================
# Parameter Tests
# ============================================================================


class TestDetectDataDriftParameters:
    """Tests for detect_data_drift parameters."""

    def test_custom_feature_columns(self, reference_csv, current_with_drift_csv):
        """Test drift detection on specific columns only."""
        result = detect_data_drift(
            reference_csv, current_with_drift_csv, feature_columns=["age", "income"]
        )

        assert result["success"] is True
        assert result["total_features_checked"] == 2
        feature_names = [f["feature_name"] for f in result["feature_results"]]
        assert set(feature_names) == {"age", "income"}

    def test_custom_drift_threshold(self, reference_csv, current_no_drift_csv):
        """Test drift detection with custom threshold."""
        # With very strict threshold (lower p-value required to NOT detect drift)
        result_strict = detect_data_drift(
            reference_csv, current_no_drift_csv, drift_threshold=0.001
        )

        # With very loose threshold (higher p-value required to NOT detect drift)
        result_loose = detect_data_drift(reference_csv, current_no_drift_csv, drift_threshold=0.5)

        assert result_strict["success"] is True
        assert result_loose["success"] is True
        # Both should work with different thresholds
        # Looser threshold (higher) will detect MORE drift (more features fail the test)
        # The actual drift_share depends on the statistical tests
        assert 0 <= result_strict["drift_share"] <= 1
        assert 0 <= result_loose["drift_share"] <= 1

    def test_custom_dataset_name(self, reference_csv, current_no_drift_csv):
        """Test drift detection with custom dataset name."""
        result = detect_data_drift(
            reference_csv, current_no_drift_csv, dataset_name="my_production_data"
        )

        assert result["success"] is True
        assert result["dataset_name"] == "my_production_data"

    def test_categorical_columns_specification(self, reference_csv, current_with_drift_csv):
        """Test drift detection with explicit categorical column specification."""
        result = detect_data_drift(
            reference_csv, current_with_drift_csv, categorical_columns=["category"]
        )

        assert result["success"] is True
        assert "feature_results" in result

    def test_numerical_columns_specification(self, reference_csv, current_with_drift_csv):
        """Test drift detection with explicit numerical column specification."""
        result = detect_data_drift(
            reference_csv,
            current_with_drift_csv,
            numerical_columns=["age", "income", "score"],
        )

        assert result["success"] is True
        assert "feature_results" in result


# ============================================================================
# Result Structure Tests
# ============================================================================


class TestDetectDataDriftResultStructure:
    """Tests for the structure of detect_data_drift results."""

    def test_result_contains_report_metadata(self, reference_csv, current_no_drift_csv):
        """Test that result contains report metadata."""
        result = detect_data_drift(reference_csv, current_no_drift_csv)

        assert result["success"] is True
        assert "report_id" in result
        assert "timestamp" in result
        assert "drift_type" in result
        assert result["drift_type"] == "data"

    def test_result_contains_dataset_info(self, reference_csv, current_no_drift_csv):
        """Test that result contains dataset information."""
        result = detect_data_drift(reference_csv, current_no_drift_csv)

        assert result["success"] is True
        assert "reference_rows" in result
        assert "current_rows" in result
        assert result["reference_rows"] == 500
        assert result["current_rows"] == 500

    def test_result_contains_feature_results(self, reference_csv, current_with_drift_csv):
        """Test that result contains detailed feature results."""
        result = detect_data_drift(reference_csv, current_with_drift_csv)

        assert result["success"] is True
        assert "feature_results" in result
        assert len(result["feature_results"]) > 0

        for feature in result["feature_results"]:
            assert "feature_name" in feature
            assert "drift_detected" in feature
            assert "drift_score" in feature
            assert "stattest_name" in feature
            assert "stattest_threshold" in feature

    def test_result_contains_recommendations(self, reference_csv, current_with_drift_csv):
        """Test that result contains recommendations when drift is detected."""
        result = detect_data_drift(reference_csv, current_with_drift_csv)

        assert result["success"] is True
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_result_contains_summary(self, reference_csv, current_with_drift_csv):
        """Test that result contains drift summary."""
        result = detect_data_drift(reference_csv, current_with_drift_csv)

        assert result["success"] is True
        assert "drifted_features" in result
        assert "total_features_checked" in result
        assert "drifted_features_count" in result
        assert "message" in result

    def test_result_contains_evidently_status(self, reference_csv, current_no_drift_csv):
        """Test that result indicates if Evidently was used."""
        result = detect_data_drift(reference_csv, current_no_drift_csv)

        assert result["success"] is True
        assert "evidently_available" in result
        assert isinstance(result["evidently_available"], bool)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestDetectDataDriftErrorHandling:
    """Tests for error handling in detect_data_drift."""

    def test_nonexistent_reference_path(self, current_no_drift_csv):
        """Test error handling for non-existent reference file."""
        result = detect_data_drift("/nonexistent/reference.csv", current_no_drift_csv)

        assert result["success"] is False
        assert "error" in result
        assert "reference" in result["error"].lower() or "not exist" in result["error"].lower()

    def test_nonexistent_current_path(self, reference_csv):
        """Test error handling for non-existent current file."""
        result = detect_data_drift(reference_csv, "/nonexistent/current.csv")

        assert result["success"] is False
        assert "error" in result
        assert "current" in result["error"].lower() or "not exist" in result["error"].lower()

    def test_unsupported_file_type(self, temp_dir, reference_csv):
        """Test error handling for unsupported file type."""
        txt_file = temp_dir / "data.txt"
        txt_file.write_text("some text data")

        result = detect_data_drift(reference_csv, str(txt_file))

        assert result["success"] is False
        assert "error" in result
        assert "unsupported" in result["error"].lower() or "type" in result["error"].lower()


# ============================================================================
# Severity Tests
# ============================================================================


class TestDetectDataDriftSeverity:
    """Tests for drift severity calculation."""

    def test_severity_none_when_no_drift(self, reference_csv, current_no_drift_csv):
        """Test that severity is low or none when no significant drift."""
        result = detect_data_drift(reference_csv, current_no_drift_csv, drift_threshold=0.01)

        assert result["success"] is True
        # Severity should be low when datasets are similar
        assert result["severity"] in ["none", "low", "medium"]

    def test_severity_increases_with_drift(self, reference_csv, current_with_drift_csv):
        """Test that severity increases with more drift."""
        result = detect_data_drift(reference_csv, current_with_drift_csv)

        assert result["success"] is True
        # With significant drift, severity should be higher
        assert result["severity"] in ["medium", "high", "critical"]

    def test_severity_values_are_valid(self, reference_csv, current_no_drift_csv):
        """Test that severity values are from the expected set."""
        result = detect_data_drift(reference_csv, current_no_drift_csv)

        assert result["success"] is True
        valid_severities = ["none", "low", "medium", "high", "critical"]
        assert result["severity"] in valid_severities


# ============================================================================
# Integration Tests
# ============================================================================


class TestDetectDataDriftIntegration:
    """Integration tests for detect_data_drift."""

    def test_drift_detection_workflow(self, reference_csv, current_with_drift_csv, temp_dir):
        """Test complete drift detection workflow."""
        # First check with drifted data
        drift_result = detect_data_drift(
            reference_csv,
            current_with_drift_csv,
            dataset_name="production_data",
        )

        assert drift_result["success"] is True
        assert drift_result["overall_drift_detected"] is True

        # Verify recommendations are actionable
        assert len(drift_result["recommendations"]) > 0

        # Check drifted features are identified
        assert drift_result["drifted_features_count"] > 0

    def test_multiple_drift_checks(self, reference_csv, temp_dir):
        """Test running multiple drift checks sequentially."""
        # Create multiple current datasets with varying drift
        results = []

        for seed, drift_amount in [(100, 0), (101, 10), (102, 30)]:
            np.random.seed(seed)
            current_path = temp_dir / f"current_{seed}.csv"
            df = pd.DataFrame(
                {
                    "age": np.random.normal(35 + drift_amount, 10, 300),
                    "income": np.random.normal(50000 + drift_amount * 1000, 15000, 300),
                    "category": np.random.choice(["A", "B", "C"], 300),
                    "score": np.random.uniform(drift_amount, 100 + drift_amount, 300),
                }
            )
            df.to_csv(current_path, index=False)

            result = detect_data_drift(reference_csv, str(current_path))
            results.append(result)

        # All should succeed
        assert all(r["success"] for r in results)

        # Drift share should generally increase with drift amount
        drift_shares = [r["drift_share"] for r in results]
        # At minimum, the last one (most drift) should have higher or equal drift share
        assert drift_shares[2] >= drift_shares[0] or drift_shares[2] > 0.3

    def test_drift_detection_with_missing_values(self, temp_dir):
        """Test drift detection handles datasets with missing values."""
        # Create reference with some nulls
        ref_path = temp_dir / "ref_with_nulls.csv"
        np.random.seed(42)
        ref_df = pd.DataFrame(
            {
                "value1": [1.0, 2.0, np.nan, 4.0, 5.0] * 50,
                "value2": np.random.normal(0, 1, 250),
            }
        )
        ref_df.to_csv(ref_path, index=False)

        # Create current with different null pattern
        cur_path = temp_dir / "cur_with_nulls.csv"
        cur_df = pd.DataFrame(
            {
                "value1": [1.0, np.nan, 3.0, 4.0, np.nan] * 50,
                "value2": np.random.normal(0.5, 1.2, 250),
            }
        )
        cur_df.to_csv(cur_path, index=False)

        result = detect_data_drift(str(ref_path), str(cur_path))

        assert result["success"] is True
        assert "feature_results" in result


# ============================================================================
# Feature-Specific Tests
# ============================================================================


class TestDetectDataDriftFeatures:
    """Tests for specific feature behavior in detect_data_drift."""

    def test_numerical_feature_drift(self, temp_dir):
        """Test drift detection for numerical features."""
        # Reference with normal distribution
        ref_path = temp_dir / "ref_numeric.csv"
        np.random.seed(42)
        ref_df = pd.DataFrame(
            {
                "numeric1": np.random.normal(100, 10, 500),
                "numeric2": np.random.uniform(0, 50, 500),
            }
        )
        ref_df.to_csv(ref_path, index=False)

        # Current with shifted distributions
        cur_path = temp_dir / "cur_numeric.csv"
        cur_df = pd.DataFrame(
            {
                "numeric1": np.random.normal(150, 20, 500),  # Shifted
                "numeric2": np.random.uniform(25, 75, 500),  # Shifted
            }
        )
        cur_df.to_csv(cur_path, index=False)

        result = detect_data_drift(
            str(ref_path), str(cur_path), numerical_columns=["numeric1", "numeric2"]
        )

        assert result["success"] is True
        assert result["overall_drift_detected"] is True

    def test_categorical_feature_drift(self, temp_dir):
        """Test drift detection for categorical features."""
        # Reference with certain distribution
        ref_path = temp_dir / "ref_categorical.csv"
        np.random.seed(42)
        ref_df = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], 500, p=[0.5, 0.3, 0.2]),
                "cat2": np.random.choice(["X", "Y"], 500, p=[0.7, 0.3]),
            }
        )
        ref_df.to_csv(ref_path, index=False)

        # Current with completely different distribution
        cur_path = temp_dir / "cur_categorical.csv"
        cur_df = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], 500, p=[0.1, 0.1, 0.8]),
                "cat2": np.random.choice(["X", "Z"], 500, p=[0.3, 0.7]),  # New category
            }
        )
        cur_df.to_csv(cur_path, index=False)

        result = detect_data_drift(
            str(ref_path), str(cur_path), categorical_columns=["cat1", "cat2"]
        )

        assert result["success"] is True
        # Should detect drift due to distribution changes
        assert "feature_results" in result

    def test_mixed_feature_types(self, reference_csv, current_with_drift_csv):
        """Test drift detection with both numerical and categorical features."""
        result = detect_data_drift(
            reference_csv,
            current_with_drift_csv,
            categorical_columns=["category"],
            numerical_columns=["age", "income", "score"],
        )

        assert result["success"] is True
        assert len(result["feature_results"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
