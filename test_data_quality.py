"""
Tests for Data Quality Module

Tests for DataValidator, DataProfiler, and related models.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_quality import (
    AnomalyDetectionResult,
    AnomalyType,
    ColumnStatistics,
    DataProfiler,
    DataQualityReport,
    DataSchema,
    DatasetStatistics,
    DataType,
    DataValidator,
    SchemaField,
    SchemaValidationResult,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "name": [f"item_{i}" for i in range(100)],
            "price": np.random.uniform(10, 100, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "quantity": np.random.randint(1, 50, 100),
            "rating": np.random.uniform(1, 5, 100),
        }
    )


@pytest.fixture
def dataframe_with_nulls():
    """Create a DataFrame with null values."""
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "name": [f"item_{i}" if i % 5 != 0 else None for i in range(100)],
            "price": [i * 10.0 if i % 10 != 0 else None for i in range(100)],
            "category": np.random.choice(["A", "B", "C", None], 100),
        }
    )
    return df


@pytest.fixture
def dataframe_with_outliers():
    """Create a DataFrame with outliers."""
    np.random.seed(42)
    normal_values = np.random.normal(50, 10, 95)
    outliers = [200, -50, 300, -100, 500]  # Clear outliers
    values = np.concatenate([normal_values, outliers])
    np.random.shuffle(values)
    return pd.DataFrame({"value": values})


@pytest.fixture
def dataframe_with_duplicates():
    """Create a DataFrame with duplicate rows."""
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 2, 4, 5, 1],
            "name": ["a", "b", "c", "a", "b", "d", "e", "a"],
            "value": [10, 20, 30, 10, 20, 40, 50, 10],
        }
    )
    return data


# ============================================================================
# DataValidator Tests
# ============================================================================


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_dataframe_basic(self, sample_dataframe):
        """Test basic DataFrame validation."""
        validator = DataValidator()
        report = validator.validate_dataframe(sample_dataframe, dataset_name="test_data")

        assert isinstance(report, DataQualityReport)
        assert report.dataset_name == "test_data"
        assert report.statistics.row_count == 100
        assert report.statistics.column_count == 6
        assert report.overall_score >= 0 and report.overall_score <= 100

    def test_null_check_rule(self, dataframe_with_nulls):
        """Test null value checking."""
        validator = DataValidator()
        validator.add_null_check("name", max_null_percentage=10.0)
        validator.add_null_check("price", max_null_percentage=5.0)

        report = validator.validate_dataframe(
            dataframe_with_nulls, dataset_name="null_test", include_builtin=False
        )

        # Find the null check results
        name_result = next((r for r in report.validation_results if r.column == "name"), None)
        price_result = next((r for r in report.validation_results if r.column == "price"), None)

        assert name_result is not None
        # 20% null rate should pass 10% threshold? No, it should fail
        assert name_result.status == ValidationStatus.FAILED

        assert price_result is not None
        # 10% null rate should fail 5% threshold
        assert price_result.status == ValidationStatus.FAILED

    def test_range_check_rule(self, sample_dataframe):
        """Test range validation."""
        validator = DataValidator()
        validator.add_range_check("price", min_value=0, max_value=200)
        validator.add_range_check("quantity", min_value=1, max_value=100)

        report = validator.validate_dataframe(
            sample_dataframe, dataset_name="range_test", include_builtin=False
        )

        # All values should be within range
        for result in report.validation_results:
            assert result.status == ValidationStatus.PASSED

    def test_range_check_fails(self, sample_dataframe):
        """Test range validation failure."""
        validator = DataValidator()
        # Set tight range that will fail
        validator.add_range_check("price", min_value=50, max_value=60)

        report = validator.validate_dataframe(
            sample_dataframe, dataset_name="range_fail_test", include_builtin=False
        )

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows > 0

    def test_pattern_check_rule(self):
        """Test pattern validation."""
        df = pd.DataFrame(
            {
                "email": ["test@example.com", "invalid", "user@domain.org", "bad@"],
                "phone": ["123-456-7890", "invalid", "555-123-4567", "1234567890"],
            }
        )

        validator = DataValidator()
        validator.add_pattern_check("email", r"^[\w.-]+@[\w.-]+\.\w+$")

        report = validator.validate_dataframe(
            df, dataset_name="pattern_test", include_builtin=False
        )

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows == 2  # "invalid" and "bad@"

    def test_allowed_values_check(self, sample_dataframe):
        """Test allowed values validation."""
        validator = DataValidator()
        validator.add_allowed_values_check("category", ["A", "B", "C"])

        report = validator.validate_dataframe(
            sample_dataframe, dataset_name="allowed_test", include_builtin=False
        )

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_allowed_values_check_fails(self, sample_dataframe):
        """Test allowed values validation failure."""
        validator = DataValidator()
        validator.add_allowed_values_check("category", ["A", "B"])  # Missing C

        report = validator.validate_dataframe(
            sample_dataframe, dataset_name="allowed_fail_test", include_builtin=False
        )

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED

    def test_duplicate_rows_check(self, dataframe_with_duplicates):
        """Test duplicate row detection."""
        validator = DataValidator()
        # Add built-in duplicate check
        from data_quality.models import ValidationRule

        validator.add_rule(
            ValidationRule(
                rule_id="no_dups",
                rule_name="No Duplicates",
                rule_type="row_duplicates",
                parameters={"max_duplicate_percentage": 0.0},
                severity=ValidationSeverity.WARNING,
            )
        )

        report = validator.validate_dataframe(
            dataframe_with_duplicates, dataset_name="dup_test", include_builtin=False
        )

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows > 0

    def test_clear_rules(self, sample_dataframe):
        """Test clearing validation rules."""
        validator = DataValidator()
        validator.add_null_check("name")
        validator.add_range_check("price", min_value=0)

        validator.clear_rules()

        report = validator.validate_dataframe(
            sample_dataframe, dataset_name="clear_test", include_builtin=False
        )

        # Should have no results after clearing (except built-in if enabled)
        assert len(report.validation_results) == 0

    def test_quality_score_calculation(self, sample_dataframe):
        """Test quality score calculation."""
        validator = DataValidator()
        # Add rules that will pass
        validator.add_range_check("price", min_value=0, max_value=1000)

        report = validator.validate_dataframe(
            sample_dataframe, dataset_name="score_test", include_builtin=False
        )

        # High score when all checks pass
        assert report.overall_score >= 90

    def test_recommendations_generated(self, dataframe_with_nulls):
        """Test that recommendations are generated."""
        validator = DataValidator()
        report = validator.validate_dataframe(dataframe_with_nulls, dataset_name="rec_test")

        # Should have recommendations due to null values
        assert isinstance(report.recommendations, list)


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestSchemaValidation:
    """Tests for schema validation functionality."""

    def test_valid_schema(self, sample_dataframe):
        """Test validation against a valid schema."""
        schema = DataSchema(
            schema_name="test_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC, nullable=False),
                SchemaField(name="name", data_type=DataType.TEXT),
                SchemaField(name="price", data_type=DataType.NUMERIC, min_value=0),
                SchemaField(name="category", data_type=DataType.CATEGORICAL),
                SchemaField(name="quantity", data_type=DataType.NUMERIC),
                SchemaField(name="rating", data_type=DataType.NUMERIC, min_value=1, max_value=5),
            ],
        )

        validator = DataValidator()
        result = validator.validate_schema(sample_dataframe, schema)

        assert isinstance(result, SchemaValidationResult)
        assert result.is_valid
        assert len(result.missing_columns) == 0

    def test_missing_columns(self, sample_dataframe):
        """Test detection of missing columns."""
        schema = DataSchema(
            schema_name="test_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC),
                SchemaField(name="nonexistent_column", data_type=DataType.TEXT),
            ],
        )

        validator = DataValidator()
        result = validator.validate_schema(sample_dataframe, schema)

        assert not result.is_valid
        assert "nonexistent_column" in result.missing_columns

    def test_extra_columns_strict(self, sample_dataframe):
        """Test detection of extra columns in strict mode."""
        schema = DataSchema(
            schema_name="test_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC),
            ],
            strict=True,
        )

        validator = DataValidator()
        result = validator.validate_schema(sample_dataframe, schema)

        assert not result.is_valid
        assert len(result.extra_columns) > 0

    def test_nullable_constraint(self, dataframe_with_nulls):
        """Test nullable constraint validation."""
        schema = DataSchema(
            schema_name="test_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC, nullable=False),
                SchemaField(name="name", data_type=DataType.TEXT, nullable=False),
            ],
        )

        validator = DataValidator()
        result = validator.validate_schema(dataframe_with_nulls, schema)

        assert not result.is_valid
        assert len(result.constraint_violations) > 0

    def test_allowed_values_constraint(self):
        """Test allowed values constraint in schema."""
        df = pd.DataFrame(
            {
                "status": ["active", "inactive", "pending", "unknown"],
            }
        )

        schema = DataSchema(
            schema_name="test_schema",
            fields=[
                SchemaField(
                    name="status",
                    data_type=DataType.CATEGORICAL,
                    allowed_values=["active", "inactive", "pending"],
                ),
            ],
        )

        validator = DataValidator()
        result = validator.validate_schema(df, schema)

        assert not result.is_valid
        assert len(result.constraint_violations) > 0


# ============================================================================
# DataProfiler Tests
# ============================================================================


class TestDataProfiler:
    """Tests for DataProfiler class."""

    def test_profile_basic(self, sample_dataframe):
        """Test basic profiling."""
        profiler = DataProfiler()
        stats = profiler.profile(sample_dataframe)

        assert isinstance(stats, DatasetStatistics)
        assert stats.row_count == 100
        assert stats.column_count == 6
        assert len(stats.columns) == 6

    def test_column_statistics(self, sample_dataframe):
        """Test column-level statistics."""
        profiler = DataProfiler()
        stats = profiler.profile(sample_dataframe)

        # Find price column stats
        price_stats = next((c for c in stats.columns if c.column_name == "price"), None)

        assert price_stats is not None
        assert price_stats.data_type == DataType.NUMERIC
        assert price_stats.null_count == 0
        assert price_stats.mean is not None
        assert price_stats.std is not None
        assert price_stats.min_value is not None
        assert price_stats.max_value is not None

    def test_categorical_stats(self, sample_dataframe):
        """Test categorical column statistics."""
        profiler = DataProfiler()
        stats = profiler.profile(sample_dataframe)

        # Find category column stats
        cat_stats = next((c for c in stats.columns if c.column_name == "category"), None)

        assert cat_stats is not None
        assert cat_stats.data_type == DataType.CATEGORICAL
        assert len(cat_stats.top_values) > 0

    def test_missing_value_stats(self, dataframe_with_nulls):
        """Test missing value statistics."""
        profiler = DataProfiler()
        stats = profiler.profile(dataframe_with_nulls)

        assert stats.total_missing > 0
        assert stats.missing_percentage > 0

    def test_duplicate_detection(self, dataframe_with_duplicates):
        """Test duplicate row detection in profile."""
        profiler = DataProfiler()
        stats = profiler.profile(dataframe_with_duplicates)

        assert stats.duplicate_rows > 0
        assert stats.duplicate_percentage > 0

    def test_detect_anomalies_outliers(self, dataframe_with_outliers):
        """Test outlier detection."""
        profiler = DataProfiler()
        result = profiler.detect_anomalies(dataframe_with_outliers, methods=["iqr"])

        assert isinstance(result, AnomalyDetectionResult)
        assert result.total_anomalies > 0
        assert AnomalyType.OUTLIER.value in result.anomalies_by_type

    def test_detect_anomalies_missing(self, dataframe_with_nulls):
        """Test missing value anomaly detection."""
        profiler = DataProfiler()
        result = profiler.detect_anomalies(dataframe_with_nulls, methods=["missing"])

        assert isinstance(result, AnomalyDetectionResult)

    def test_detect_anomalies_duplicates(self, dataframe_with_duplicates):
        """Test duplicate anomaly detection."""
        profiler = DataProfiler()
        result = profiler.detect_anomalies(dataframe_with_duplicates, methods=["duplicates"])

        assert result.total_anomalies > 0
        assert AnomalyType.DUPLICATE.value in result.anomalies_by_type

    def test_detect_anomalies_all_methods(self, sample_dataframe):
        """Test all anomaly detection methods together."""
        profiler = DataProfiler()
        result = profiler.detect_anomalies(sample_dataframe)

        assert isinstance(result, AnomalyDetectionResult)
        assert result.affected_percentage >= 0

    def test_compare_distributions(self):
        """Test distribution comparison."""
        np.random.seed(42)
        # Reference: normal distribution centered at 50
        df_ref = pd.DataFrame({"value": np.random.normal(50, 10, 1000)})
        # Current: shifted distribution centered at 70
        df_cur = pd.DataFrame({"value": np.random.normal(70, 10, 1000)})

        profiler = DataProfiler()
        anomalies = profiler.compare_distributions(df_ref, df_cur)

        assert len(anomalies) > 0
        assert anomalies[0].anomaly_type == AnomalyType.DISTRIBUTION_SHIFT


# ============================================================================
# File Validation Tests
# ============================================================================


class TestFileValidation:
    """Tests for file-based validation."""

    def test_validate_csv_file(self, sample_dataframe):
        """Test validating a CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)

            validator = DataValidator()
            report = validator.validate_file(f.name)

            assert isinstance(report, DataQualityReport)
            assert report.statistics.row_count == 100

            # Cleanup
            Path(f.name).unlink()

    def test_validate_parquet_file(self, sample_dataframe):
        """Test validating a Parquet file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            sample_dataframe.to_parquet(f.name, index=False)

            validator = DataValidator()
            report = validator.validate_file(f.name)

            assert isinstance(report, DataQualityReport)
            assert report.statistics.row_count == 100

            # Cleanup
            Path(f.name).unlink()

    def test_validate_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        validator = DataValidator()

        with pytest.raises(FileNotFoundError):
            validator.validate_file("/nonexistent/path/data.csv")

    def test_validate_unsupported_format(self, sample_dataframe):
        """Test error handling for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test data")

            validator = DataValidator()

            with pytest.raises(ValueError, match="Unsupported file format"):
                validator.validate_file(f.name)

            # Cleanup
            Path(f.name).unlink()


# ============================================================================
# Model Tests
# ============================================================================


class TestModels:
    """Tests for Pydantic models."""

    def test_validation_result_model(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="Test Rule",
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.ERROR,
            message="Test passed",
        )

        assert result.rule_id == "test_rule"
        assert result.status == ValidationStatus.PASSED

    def test_column_statistics_model(self):
        """Test ColumnStatistics model."""
        stats = ColumnStatistics(
            column_name="test_col",
            data_type=DataType.NUMERIC,
            total_count=100,
            null_count=5,
            null_percentage=5.0,
            unique_count=95,
            unique_percentage=95.0,
            mean=50.0,
            median=49.5,
            std=10.0,
            min_value=10.0,
            max_value=90.0,
            q1=40.0,
            q3=60.0,
        )

        assert stats.column_name == "test_col"
        assert stats.mean == 50.0

    def test_data_schema_model(self):
        """Test DataSchema model."""
        schema = DataSchema(
            schema_name="test_schema",
            version="1.0",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC, nullable=False, unique=True),
                SchemaField(name="name", data_type=DataType.TEXT),
            ],
        )

        assert schema.schema_name == "test_schema"
        assert len(schema.fields) == 2
        assert schema.fields[0].unique is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_validation_workflow(self, sample_dataframe):
        """Test a complete validation workflow."""
        # 1. Create validator with rules
        validator = DataValidator()
        validator.add_null_check("price", max_null_percentage=5.0)
        validator.add_range_check("price", min_value=0, max_value=200)
        validator.add_range_check("rating", min_value=1, max_value=5)
        validator.add_allowed_values_check("category", ["A", "B", "C"])

        # 2. Run validation
        report = validator.validate_dataframe(sample_dataframe, dataset_name="test_data")

        # 3. Check report
        assert report.overall_score >= 80
        assert report.statistics.row_count == 100

        # 4. Profile data
        profiler = DataProfiler()
        anomalies = profiler.detect_anomalies(sample_dataframe)

        assert isinstance(anomalies, AnomalyDetectionResult)

    def test_data_quality_pipeline(self):
        """Test a complete data quality pipeline."""
        # Create a dataset with known issues
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "id": range(1, 101),
                "value": list(np.random.normal(50, 10, 95)) + [200, -50, 300, -100, 500],
                "category": ["A"] * 40 + ["B"] * 30 + ["C"] * 25 + [None] * 5,
            }
        )

        # Define schema
        schema = DataSchema(
            schema_name="pipeline_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC, nullable=False, unique=True),
                SchemaField(name="value", data_type=DataType.NUMERIC, min_value=0, max_value=100),
                SchemaField(
                    name="category",
                    data_type=DataType.CATEGORICAL,
                    allowed_values=["A", "B", "C"],
                ),
            ],
        )

        # Validate schema
        validator = DataValidator()
        schema_result = validator.validate_schema(df, schema)

        # Schema validation should fail due to value constraints and nulls
        assert not schema_result.is_valid

        # Add strict validation rules that will fail
        validator.add_null_check("category", max_null_percentage=0.0)
        validator.add_range_check("value", min_value=0, max_value=100)

        # Run quality checks with strict rules
        quality_report = validator.validate_dataframe(df, include_builtin=False)
        assert quality_report.overall_score < 100  # Should have issues due to strict rules

        # Profile for anomalies
        profiler = DataProfiler()
        anomalies = profiler.detect_anomalies(df)

        # Should detect outliers in value column
        assert anomalies.total_anomalies > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
