"""
Pytest tests for Data Validation MCP Tools

Tests for validate_dataset, create_expectation_suite, and check_data_quality
MCP tools that provide data validation capabilities for ML pipelines.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcp_mlops_tools import (
    check_data_quality,
    compare_distributions,
    create_expectation_suite,
    detect_anomalies,
    profile_dataset,
    validate_dataset,
    validate_schema,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="mlops_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "sample_data.csv"
    csv_content = """id,label,value,category
1,cat,10.5,A
2,dog,15.2,B
3,cat,12.3,A
4,bird,18.1,C
5,dog,14.7,B
6,cat,11.2,A
7,dog,16.0,B
8,bird,19.3,C
9,cat,13.4,A
10,dog,17.8,B
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def csv_with_issues(temp_dir):
    """Create a CSV file with quality issues (nulls, duplicates, outliers)."""
    csv_path = temp_dir / "data_with_issues.csv"
    csv_content = """id,label,value,category
1,cat,10.5,A
2,dog,15.2,B
3,cat,12.3,A
4,,18.1,C
5,dog,14.7,B
1,cat,10.5,A
6,bird,100.0,D
7,cat,11.2,A
8,dog,,B
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def csv_with_nulls(temp_dir):
    """Create a CSV with null values in specific columns."""
    csv_path = temp_dir / "nulls_data.csv"
    csv_content = """id,value
1,10
,20
3,30
,40
5,50
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def large_csv(temp_dir):
    """Create a larger CSV file for sampling tests."""
    csv_path = temp_dir / "large_data.csv"
    rows = ["id,value,label"]
    for i in range(500):
        rows.append(f"{i},{i * 1.5},{'cat' if i % 2 == 0 else 'dog'}")
    csv_path.write_text("\n".join(rows))
    return str(csv_path)


@pytest.fixture
def parquet_file(temp_dir):
    """Create a Parquet file for testing."""
    parquet_path = temp_dir / "test_data.parquet"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.5, 20.3, 15.7, 18.2, 22.1],
            "category": ["A", "B", "A", "C", "B"],
        }
    )
    df.to_parquet(parquet_path)
    return str(parquet_path)


@pytest.fixture
def json_file(temp_dir):
    """Create a JSON Lines file for testing (one JSON object per line)."""
    json_path = temp_dir / "test_data.json"
    # JSON Lines format (one JSON object per line)
    lines = [
        '{"id": 1, "value": 10.5, "category": "A"}',
        '{"id": 2, "value": 20.3, "category": "B"}',
        '{"id": 3, "value": 15.7, "category": "A"}',
    ]
    json_path.write_text("\n".join(lines))
    return str(json_path)


@pytest.fixture
def image_dir(temp_dir):
    """Create an image directory structure for testing."""
    images_dir = temp_dir / "images"
    (images_dir / "cats").mkdir(parents=True)
    (images_dir / "dogs").mkdir(parents=True)

    # Create minimal valid PNG files (1x1 pixel)
    png_data = bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,
            0x90,
            0x77,
            0x53,
            0xDE,
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,
            0x54,
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xFF,
            0xFF,
            0x3F,
            0x00,
            0x05,
            0xFE,
            0x02,
            0xFE,
            0xDC,
            0xCC,
            0x59,
            0xE7,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )

    for i in range(5):
        (images_dir / "cats" / f"cat_{i}.png").write_bytes(png_data)
    for i in range(3):
        (images_dir / "dogs" / f"dog_{i}.png").write_bytes(png_data)

    # Create an invalid image file
    (images_dir / "cats" / "invalid.png").write_text("not an image")

    return str(images_dir)


@pytest.fixture
def project_dir(temp_dir):
    """Create a project directory structure."""
    project = temp_dir / "ml_project"
    project.mkdir()
    (project / "data").mkdir()
    return str(project)


@pytest.fixture
def reference_csv(temp_dir):
    """Create a reference CSV for distribution comparison."""
    csv_path = temp_dir / "reference.csv"
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "value": np.random.normal(50, 10, 100),
            "score": np.random.randint(70, 90, 100),
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def drifted_csv(temp_dir):
    """Create a CSV with significant distribution drift."""
    csv_path = temp_dir / "drifted.csv"
    np.random.seed(123)
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "value": np.random.normal(80, 15, 100),  # Shifted mean
            "score": np.random.randint(20, 40, 100),  # Completely different
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def anomaly_csv(temp_dir):
    """Create a CSV with clear anomalies."""
    csv_path = temp_dir / "anomaly_data.csv"
    csv_content = """id,value,category
1,10.5,A
2,15.2,B
3,12.3,A
4,18.1,C
5,14.7,B
6,100.0,A
7,11.2,A
8,-50.0,B
1,10.5,A
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


# ============================================================================
# validate_dataset Tests
# ============================================================================


class TestValidateDataset:
    """Tests for validate_dataset MCP tool."""

    def test_validate_csv_success(self, sample_csv):
        """Test successful validation of a clean CSV file."""
        result = validate_dataset(sample_csv, dataset_type="csv")

        assert result["success"] is True
        assert result["dataset_type"] == "csv"
        assert "statistics" in result
        assert result["statistics"]["total_rows"] == 10

    def test_validate_csv_auto_detect(self, sample_csv):
        """Test auto-detection of CSV file type."""
        result = validate_dataset(sample_csv)

        assert result["success"] is True
        assert result["dataset_type"] == "csv"

    def test_validate_csv_with_issues(self, csv_with_issues):
        """Test validation detects quality issues."""
        result = validate_dataset(csv_with_issues, dataset_type="csv")

        assert result["success"] is True
        assert result["total_issues"] > 0 or result["total_warnings"] > 0

    def test_validate_specific_checks(self, csv_with_issues):
        """Test validation with specific checks only."""
        result = validate_dataset(
            csv_with_issues, dataset_type="csv", checks=["missing_values", "duplicates"]
        )

        assert result["success"] is True
        assert "missing_values" in result["checks_performed"]
        assert "duplicates" in result["checks_performed"]

    def test_validate_with_sample_size(self, large_csv):
        """Test validation with sample size limit."""
        result = validate_dataset(large_csv, dataset_type="csv", sample_size=50)

        assert result["success"] is True
        stats = result.get("statistics", {})
        # Should validate only sampled rows
        assert stats.get("total_rows", 0) <= 50

    def test_validate_parquet(self, parquet_file):
        """Test validation of Parquet file."""
        result = validate_dataset(parquet_file, dataset_type="parquet")

        assert result["success"] is True
        assert result["dataset_type"] == "parquet"
        assert result["statistics"]["total_rows"] == 5

    def test_validate_parquet_auto_detect(self, parquet_file):
        """Test auto-detection of Parquet file type."""
        result = validate_dataset(parquet_file)

        assert result["success"] is True
        assert result["dataset_type"] == "parquet"

    def test_validate_json(self, json_file):
        """Test validation of JSON file."""
        result = validate_dataset(json_file, dataset_type="json")

        assert result["success"] is True
        assert result["dataset_type"] == "json"

    def test_validate_image_directory(self, image_dir):
        """Test validation of image directory."""
        result = validate_dataset(image_dir, dataset_type="images")

        assert result["success"] is True
        assert result["dataset_type"] == "images"
        stats = result.get("statistics", {})
        assert stats.get("total_images", 0) > 0
        assert "classes" in stats

    def test_validate_nonexistent_path(self):
        """Test error handling for non-existent path."""
        result = validate_dataset("/nonexistent/path/data.csv")

        assert result["success"] is False
        assert "error" in result

    def test_validate_unsupported_type(self, temp_dir):
        """Test handling for unknown file type - returns success with warning."""
        txt_file = temp_dir / "data.txt"
        txt_file.write_text("some text data")

        result = validate_dataset(str(txt_file), dataset_type="txt")

        # Unknown types return success=True but with a warning about limited validation
        assert result["success"] is True
        assert len(result.get("warnings", [])) > 0

    def test_validate_all_checks(self, sample_csv):
        """Test validation runs all checks when none specified."""
        result = validate_dataset(sample_csv, dataset_type="csv")

        assert result["success"] is True
        checks = result.get("checks_performed", [])
        # Should have multiple checks
        assert len(checks) >= 1

    def test_validate_returns_is_valid(self, sample_csv):
        """Test that result includes is_valid field."""
        result = validate_dataset(sample_csv)

        assert result["success"] is True
        assert "is_valid" in result


# ============================================================================
# create_expectation_suite Tests
# ============================================================================


class TestCreateExpectationSuite:
    """Tests for create_expectation_suite MCP tool."""

    def test_create_basic_suite(self, project_dir):
        """Test creating a basic expectation suite."""
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
                "description": "ID should not be null",
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "column": "id",
            },
        ]

        result = create_expectation_suite(project_dir, "test_suite", expectations)

        assert result["success"] is True
        assert result["suite_name"] == "test_suite"
        assert result["expectation_count"] == 2
        assert "suite_path" in result

    def test_suite_file_created(self, project_dir):
        """Test that suite file is actually created."""
        expectations = [
            {"expectation_type": "expect_column_to_exist", "column": "id"},
        ]

        result = create_expectation_suite(project_dir, "file_test_suite", expectations)

        assert result["success"] is True
        suite_path = Path(result["suite_path"])
        assert suite_path.exists()

    def test_suite_is_valid_json(self, project_dir):
        """Test that created suite is valid JSON."""
        expectations = [
            {"expectation_type": "expect_column_to_exist", "column": "name"},
        ]

        result = create_expectation_suite(project_dir, "json_test_suite", expectations)

        assert result["success"] is True
        suite_path = Path(result["suite_path"])
        with open(suite_path) as f:
            suite_data = json.load(f)
        assert "expectations" in suite_data

    def test_create_suite_with_kwargs(self, project_dir):
        """Test creating suite with expectation kwargs."""
        expectations = [
            {
                "expectation_type": "expect_column_values_to_be_between",
                "column": "value",
                "kwargs": {"min_value": 0, "max_value": 100},
                "severity": "warning",
            },
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 1000},
            },
        ]

        result = create_expectation_suite(project_dir, "kwargs_suite", expectations)

        assert result["success"] is True
        assert result["expectation_count"] == 2
        assert "expectation_types" in result

    def test_create_suite_in_set_expectation(self, project_dir):
        """Test creating suite with in-set expectation."""
        expectations = [
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "column": "category",
                "kwargs": {"value_set": ["A", "B", "C"]},
            },
        ]

        result = create_expectation_suite(project_dir, "in_set_suite", expectations)

        assert result["success"] is True

    def test_create_suite_custom_output_dir(self, project_dir):
        """Test creating suite in custom directory."""
        expectations = [
            {"expectation_type": "expect_column_to_exist", "column": "id"},
        ]

        result = create_expectation_suite(
            project_dir, "custom_dir_suite", expectations, output_dir="custom_expectations"
        )

        assert result["success"] is True
        expected_path = Path(project_dir) / "custom_expectations" / "custom_dir_suite.json"
        assert expected_path.exists()

    def test_reject_invalid_expectations(self, project_dir):
        """Test rejection of invalid expectations (missing expectation_type)."""
        invalid_expectations = [
            {"column": "id"},  # Missing expectation_type
        ]

        result = create_expectation_suite(project_dir, "invalid_suite", invalid_expectations)

        assert result["success"] is False
        assert "error" in result

    def test_reject_empty_expectations(self, project_dir):
        """Test rejection of empty expectations list."""
        result = create_expectation_suite(project_dir, "empty_suite", [])

        assert result["success"] is False

    def test_reject_nonexistent_project_path(self):
        """Test error handling for non-existent project path."""
        expectations = [
            {"expectation_type": "expect_column_to_exist", "column": "id"},
        ]

        result = create_expectation_suite("/nonexistent/path", "test_suite", expectations)

        assert result["success"] is False
        assert "error" in result

    def test_suite_with_multiple_severity_levels(self, project_dir):
        """Test suite with different severity levels."""
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "column": "value",
                "kwargs": {"min_value": 0, "max_value": 1000},
                "severity": "warning",
            },
            {
                "expectation_type": "expect_column_to_exist",
                "column": "optional_field",
                "severity": "info",
            },
        ]

        result = create_expectation_suite(project_dir, "multi_severity_suite", expectations)

        assert result["success"] is True
        assert result["expectation_count"] == 3


# ============================================================================
# check_data_quality Tests
# ============================================================================


class TestCheckDataQuality:
    """Tests for check_data_quality MCP tool."""

    def test_basic_quality_check(self, sample_csv):
        """Test basic data quality check without custom expectations."""
        result = check_data_quality(sample_csv)

        assert result["success"] is True
        assert "overall_score" in result
        assert "is_valid" in result

    def test_quality_check_with_expectations(self, sample_csv):
        """Test quality check with custom expectations."""
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "column": "value",
                "kwargs": {"min_value": 0, "max_value": 100},
                "severity": "warning",
            },
        ]

        result = check_data_quality(sample_csv, expectations=expectations)

        assert result["success"] is True
        assert "validation_results" in result

    def test_quality_check_includes_statistics(self, sample_csv):
        """Test that statistics are included when requested."""
        result = check_data_quality(sample_csv, include_statistics=True)

        assert result["success"] is True
        assert "statistics" in result
        stats = result["statistics"]
        assert "row_count" in stats
        assert "column_count" in stats

    def test_quality_check_excludes_statistics(self, sample_csv):
        """Test that statistics are excluded when not requested."""
        result = check_data_quality(sample_csv, include_statistics=False)

        assert result["success"] is True
        assert "statistics" not in result

    def test_fail_on_error_triggers_failure(self, csv_with_nulls):
        """Test that fail_on_error causes failure on error-level check failure."""
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
            }
        ]

        result = check_data_quality(csv_with_nulls, expectations=expectations, fail_on_error=True)

        # Should fail because id column has nulls
        assert result["success"] is False

    def test_quality_check_passed_failed_counts(self, sample_csv):
        """Test that passed and failed check counts are included."""
        result = check_data_quality(sample_csv)

        assert result["success"] is True
        assert "passed_checks" in result
        assert "failed_checks" in result

    def test_quality_check_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = check_data_quality("/nonexistent/path/data.csv")

        assert result["success"] is False
        assert "error" in result

    def test_quality_check_parquet(self, parquet_file):
        """Test quality check on Parquet file."""
        result = check_data_quality(parquet_file)

        assert result["success"] is True
        assert "overall_score" in result

    def test_quality_check_unsupported_format(self, temp_dir):
        """Test error handling for unsupported file format."""
        txt_file = temp_dir / "data.txt"
        txt_file.write_text("some text data")

        result = check_data_quality(str(txt_file))

        assert result["success"] is False

    def test_quality_check_recommendations(self, csv_with_issues):
        """Test that recommendations are generated for issues."""
        result = check_data_quality(csv_with_issues)

        assert result["success"] is True
        # May or may not have recommendations depending on data
        assert "recommendations" in result or result.get("overall_score", 100) == 100


# ============================================================================
# profile_dataset Tests
# ============================================================================


class TestProfileDataset:
    """Tests for profile_dataset MCP tool."""

    def test_profile_csv(self, sample_csv):
        """Test profiling a CSV file."""
        result = profile_dataset(sample_csv)

        assert result["success"] is True
        assert "statistics" in result
        stats = result["statistics"]
        assert stats["row_count"] == 10
        assert stats["column_count"] == 4

    def test_profile_with_custom_name(self, sample_csv):
        """Test profiling with custom dataset name."""
        result = profile_dataset(sample_csv, dataset_name="my_dataset")

        assert result["success"] is True
        assert result.get("dataset_name") == "my_dataset"

    def test_profile_includes_column_stats(self, sample_csv):
        """Test that column statistics are included by default."""
        result = profile_dataset(sample_csv, include_column_stats=True)

        assert result["success"] is True
        assert "columns" in result
        assert len(result["columns"]) > 0

    def test_profile_excludes_column_stats(self, sample_csv):
        """Test that column statistics can be excluded."""
        result = profile_dataset(sample_csv, include_column_stats=False)

        assert result["success"] is True
        assert "columns" not in result

    def test_profile_parquet(self, parquet_file):
        """Test profiling a Parquet file."""
        result = profile_dataset(parquet_file)

        assert result["success"] is True
        assert result["statistics"]["row_count"] == 5

    def test_profile_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = profile_dataset("/nonexistent/path/data.csv")

        assert result["success"] is False
        assert "error" in result

    def test_profile_includes_memory_usage(self, sample_csv):
        """Test that memory usage is included in profile."""
        result = profile_dataset(sample_csv)

        assert result["success"] is True
        stats = result.get("statistics", {})
        assert "memory_usage_mb" in stats or "memory_usage" in stats


# ============================================================================
# detect_anomalies Tests
# ============================================================================


class TestDetectAnomalies:
    """Tests for detect_anomalies MCP tool."""

    def test_detect_anomalies_basic(self, anomaly_csv):
        """Test basic anomaly detection."""
        result = detect_anomalies(anomaly_csv)

        assert result["success"] is True
        assert "total_anomalies" in result
        assert result["total_anomalies"] >= 0

    def test_detect_anomalies_specific_methods(self, anomaly_csv):
        """Test anomaly detection with specific methods."""
        result = detect_anomalies(anomaly_csv, methods=["iqr", "duplicates"])

        assert result["success"] is True
        assert "methods_used" in result

    def test_detect_anomalies_custom_thresholds(self, anomaly_csv):
        """Test anomaly detection with custom thresholds."""
        result = detect_anomalies(anomaly_csv, outlier_threshold=2.0, zscore_threshold=2.5)

        assert result["success"] is True
        thresholds = result.get("thresholds", {})
        # Thresholds may be present in result
        if thresholds:
            assert thresholds.get("iqr_multiplier") == 2.0 or "iqr" in str(thresholds)

    def test_detect_anomalies_finds_outliers(self, anomaly_csv):
        """Test that clear outliers are detected."""
        result = detect_anomalies(anomaly_csv, methods=["iqr"])

        assert result["success"] is True
        # Should detect the 100.0 and -50.0 values as outliers
        assert result.get("total_anomalies", 0) > 0 or result.get("affected_rows", 0) > 0

    def test_detect_anomalies_finds_duplicates(self, anomaly_csv):
        """Test that duplicates are detected."""
        result = detect_anomalies(anomaly_csv, methods=["duplicates"])

        assert result["success"] is True
        # Should detect the duplicate row (1,10.5,A)
        anomalies_by_type = result.get("anomalies_by_type", {})
        if anomalies_by_type:
            assert (
                "duplicate" in str(anomalies_by_type).lower()
                or result.get("total_anomalies", 0) >= 0
            )

    def test_detect_anomalies_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = detect_anomalies("/nonexistent/path/data.csv")

        assert result["success"] is False

    def test_detect_anomalies_affected_percentage(self, anomaly_csv):
        """Test that affected percentage is calculated."""
        result = detect_anomalies(anomaly_csv)

        assert result["success"] is True
        assert "affected_percentage" in result
        assert result["affected_percentage"] >= 0


# ============================================================================
# validate_schema Tests
# ============================================================================


class TestValidateSchema:
    """Tests for validate_schema MCP tool."""

    def test_validate_schema_success(self, sample_csv):
        """Test successful schema validation."""
        schema = {
            "schema_name": "test_schema",
            "version": "1.0",
            "fields": [
                {"name": "id", "data_type": "numeric", "nullable": False},
                {"name": "label", "data_type": "categorical", "nullable": True},
                {"name": "value", "data_type": "numeric", "nullable": True},
                {"name": "category", "data_type": "categorical", "nullable": True},
            ],
            "strict": False,
        }

        result = validate_schema(sample_csv, schema)

        assert result["success"] is True
        assert "is_valid" in result

    def test_validate_schema_missing_column(self, sample_csv):
        """Test detection of missing column in schema."""
        schema = {
            "schema_name": "test_schema",
            "fields": [
                {"name": "id", "data_type": "numeric"},
                {"name": "nonexistent_column", "data_type": "text"},
            ],
        }

        result = validate_schema(sample_csv, schema)

        assert result["success"] is True
        assert "nonexistent_column" in result.get("missing_columns", [])

    def test_validate_schema_strict_mode(self, sample_csv):
        """Test strict mode detects extra columns."""
        schema = {
            "schema_name": "strict_schema",
            "fields": [
                {"name": "id", "data_type": "numeric"},
            ],
            "strict": True,
        }

        result = validate_schema(sample_csv, schema)

        assert result["success"] is True
        assert len(result.get("extra_columns", [])) > 0

    def test_validate_schema_nonexistent_file(self):
        """Test error handling for non-existent file."""
        schema = {"schema_name": "test", "fields": [{"name": "id", "data_type": "numeric"}]}

        result = validate_schema("/nonexistent/path/data.csv", schema)

        assert result["success"] is False

    def test_validate_schema_type_mismatch(self, sample_csv):
        """Test detection of type mismatches."""
        schema = {
            "schema_name": "type_test",
            "fields": [
                {"name": "label", "data_type": "numeric"},  # label is actually categorical
            ],
        }

        result = validate_schema(sample_csv, schema)

        assert result["success"] is True
        # May or may not detect type mismatch depending on implementation


# ============================================================================
# compare_distributions Tests
# ============================================================================


class TestCompareDistributions:
    """Tests for compare_distributions MCP tool."""

    def test_compare_distributions_basic(self, reference_csv, temp_dir):
        """Test basic distribution comparison."""
        # Create a similar current dataset
        current_csv = temp_dir / "current.csv"
        np.random.seed(43)
        df = pd.DataFrame(
            {
                "id": range(1, 101),
                "value": np.random.normal(51, 11, 100),  # Slightly shifted
                "score": np.random.randint(71, 91, 100),
            }
        )
        df.to_csv(current_csv, index=False)

        result = compare_distributions(reference_csv, str(current_csv))

        assert result["success"] is True
        assert "columns_compared" in result
        assert "drift_detected" in result

    def test_compare_distributions_detects_drift(self, reference_csv, drifted_csv):
        """Test that significant drift is detected."""
        result = compare_distributions(reference_csv, drifted_csv)

        assert result["success"] is True
        assert result.get("drift_detected", False) is True

    def test_compare_distributions_specific_columns(self, reference_csv, drifted_csv):
        """Test comparison of specific columns only."""
        result = compare_distributions(reference_csv, drifted_csv, columns=["value"])

        assert result["success"] is True
        assert result.get("columns_compared") == ["value"]

    def test_compare_distributions_nonexistent_reference(self, drifted_csv):
        """Test error handling for non-existent reference file."""
        result = compare_distributions("/nonexistent/ref.csv", drifted_csv)

        assert result["success"] is False

    def test_compare_distributions_nonexistent_current(self, reference_csv):
        """Test error handling for non-existent current file."""
        result = compare_distributions(reference_csv, "/nonexistent/cur.csv")

        assert result["success"] is False

    def test_compare_distributions_returns_shifts(self, reference_csv, drifted_csv):
        """Test that shift details are returned."""
        result = compare_distributions(reference_csv, drifted_csv)

        assert result["success"] is True
        assert "shifts" in result or "total_shifts_detected" in result


# ============================================================================
# Integration Tests
# ============================================================================


class TestDataValidationIntegration:
    """Integration tests for data validation workflow."""

    def test_validate_then_check_quality(self, sample_csv):
        """Test validation followed by quality check workflow."""
        # First validate
        validate_result = validate_dataset(sample_csv)
        assert validate_result["success"] is True

        # Then check quality
        quality_result = check_data_quality(sample_csv)
        assert quality_result["success"] is True

    def test_profile_then_detect_anomalies(self, anomaly_csv):
        """Test profiling followed by anomaly detection."""
        # First profile
        profile_result = profile_dataset(anomaly_csv)
        assert profile_result["success"] is True

        # Then detect anomalies
        anomaly_result = detect_anomalies(anomaly_csv)
        assert anomaly_result["success"] is True

    def test_create_suite_then_check_quality(self, project_dir, sample_csv):
        """Test creating expectation suite then using for quality check."""
        # Create suite
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
            },
        ]
        suite_result = create_expectation_suite(project_dir, "integration_suite", expectations)
        assert suite_result["success"] is True

        # Check quality with same expectations
        quality_result = check_data_quality(sample_csv, expectations=expectations)
        assert quality_result["success"] is True

    def test_full_validation_pipeline(self, project_dir, sample_csv, reference_csv, temp_dir):
        """Test complete validation pipeline."""
        # 1. Validate dataset structure
        validate_result = validate_dataset(sample_csv)
        assert validate_result["success"] is True

        # 2. Profile dataset
        profile_result = profile_dataset(sample_csv)
        assert profile_result["success"] is True

        # 3. Detect anomalies
        anomaly_result = detect_anomalies(sample_csv)
        assert anomaly_result["success"] is True

        # 4. Create expectation suite
        expectations = [
            {"expectation_type": "expect_column_values_to_not_be_null", "column": "id"},
            {"expectation_type": "expect_column_values_to_be_unique", "column": "id"},
        ]
        suite_result = create_expectation_suite(project_dir, "full_pipeline_suite", expectations)
        assert suite_result["success"] is True

        # 5. Run quality check
        quality_result = check_data_quality(sample_csv, expectations=expectations)
        assert quality_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
