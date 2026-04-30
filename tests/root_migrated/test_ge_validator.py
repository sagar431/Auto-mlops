"""
Tests for Great Expectations Validator Integration

Tests for GreatExpectationsValidator and ExpectationConfig classes.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_quality import (
    DataQualityReport,
    DataSchema,
    DataType,
    ExpectationConfig,
    GreatExpectationsValidator,
    SchemaField,
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
    outliers = [200, -50, 300, -100, 500]
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


@pytest.fixture
def ge_validator():
    """Create a GreatExpectationsValidator instance."""
    return GreatExpectationsValidator()


# ============================================================================
# ExpectationConfig Tests
# ============================================================================


class TestExpectationConfig:
    """Tests for ExpectationConfig class."""

    def test_basic_config(self):
        """Test basic ExpectationConfig creation."""
        config = ExpectationConfig(
            expectation_type="expect_column_values_to_not_be_null",
            column="name",
        )
        assert config.expectation_type == "expect_column_values_to_not_be_null"
        assert config.column == "name"
        assert config.kwargs == {}
        assert config.severity == ValidationSeverity.ERROR

    def test_config_with_kwargs(self):
        """Test ExpectationConfig with kwargs."""
        config = ExpectationConfig(
            expectation_type="expect_column_values_to_be_between",
            column="price",
            kwargs={"min_value": 0, "max_value": 100},
            severity=ValidationSeverity.WARNING,
        )
        assert config.kwargs == {"min_value": 0, "max_value": 100}
        assert config.severity == ValidationSeverity.WARNING

    def test_config_with_description(self):
        """Test ExpectationConfig with custom description."""
        config = ExpectationConfig(
            expectation_type="expect_column_values_to_not_be_null",
            column="id",
            description="ID must not be null",
        )
        assert config.description == "ID must not be null"


# ============================================================================
# GreatExpectationsValidator Basic Tests
# ============================================================================


class TestGreatExpectationsValidatorBasic:
    """Basic tests for GreatExpectationsValidator."""

    def test_validator_creation(self, ge_validator):
        """Test validator can be created."""
        assert ge_validator is not None
        assert isinstance(ge_validator.ge_available, bool)

    def test_add_expectation(self, ge_validator):
        """Test adding expectations."""
        ge_validator.add_expectation(
            "expect_column_values_to_not_be_null",
            column="id",
        )
        assert len(ge_validator._expectations) == 1

    def test_add_multiple_expectations(self, ge_validator):
        """Test adding multiple expectations."""
        ge_validator.add_expectation("expect_column_values_to_not_be_null", column="id")
        ge_validator.add_expectation("expect_column_values_to_be_unique", column="id")
        ge_validator.add_expectation(
            "expect_column_values_to_be_between",
            column="price",
            kwargs={"min_value": 0, "max_value": 1000},
        )
        assert len(ge_validator._expectations) == 3

    def test_clear_expectations(self, ge_validator):
        """Test clearing expectations."""
        ge_validator.add_expectation("expect_column_values_to_not_be_null", column="id")
        ge_validator.add_expectation("expect_column_values_to_be_unique", column="id")
        ge_validator.clear_expectations()
        assert len(ge_validator._expectations) == 0


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Tests for helper methods that add specific expectations."""

    def test_add_not_null_expectation(self, ge_validator):
        """Test add_not_null_expectation method."""
        ge_validator.add_not_null_expectation("id")
        assert len(ge_validator._expectations) == 1
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_values_to_not_be_null"
        assert exp.column == "id"

    def test_add_not_null_with_mostly(self, ge_validator):
        """Test add_not_null_expectation with mostly parameter."""
        ge_validator.add_not_null_expectation("name", mostly=0.95)
        exp = ge_validator._expectations[0]
        assert exp.kwargs.get("mostly") == 0.95

    def test_add_unique_expectation(self, ge_validator):
        """Test add_unique_expectation method."""
        ge_validator.add_unique_expectation("id")
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_values_to_be_unique"
        assert exp.column == "id"

    def test_add_range_expectation(self, ge_validator):
        """Test add_range_expectation method."""
        ge_validator.add_range_expectation("price", min_value=0, max_value=1000)
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_values_to_be_between"
        assert exp.kwargs["min_value"] == 0
        assert exp.kwargs["max_value"] == 1000

    def test_add_range_with_strict_bounds(self, ge_validator):
        """Test add_range_expectation with strict bounds."""
        ge_validator.add_range_expectation(
            "price", min_value=0, max_value=100, strict_min=True, strict_max=True
        )
        exp = ge_validator._expectations[0]
        assert exp.kwargs.get("strict_min") is True
        assert exp.kwargs.get("strict_max") is True

    def test_add_in_set_expectation(self, ge_validator):
        """Test add_in_set_expectation method."""
        ge_validator.add_in_set_expectation("category", ["A", "B", "C"])
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_values_to_be_in_set"
        assert exp.kwargs["value_set"] == ["A", "B", "C"]

    def test_add_regex_expectation(self, ge_validator):
        """Test add_regex_expectation method."""
        ge_validator.add_regex_expectation("email", r"^[\w.-]+@[\w.-]+\.\w+$")
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_values_to_match_regex"
        assert exp.kwargs["regex"] == r"^[\w.-]+@[\w.-]+\.\w+$"

    def test_add_column_type_expectation(self, ge_validator):
        """Test add_column_type_expectation method."""
        ge_validator.add_column_type_expectation("id", ["int64", "int32"])
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_values_to_be_in_type_list"
        assert exp.kwargs["type_list"] == ["int64", "int32"]

    def test_add_table_row_count_expectation(self, ge_validator):
        """Test add_table_row_count_expectation method."""
        ge_validator.add_table_row_count_expectation(min_value=10, max_value=1000)
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_table_row_count_to_be_between"
        assert exp.kwargs["min_value"] == 10
        assert exp.kwargs["max_value"] == 1000

    def test_add_column_exists_expectation(self, ge_validator):
        """Test add_column_exists_expectation method."""
        ge_validator.add_column_exists_expectation("id")
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_column_to_exist"
        assert exp.column == "id"

    def test_add_no_duplicates_expectation(self, ge_validator):
        """Test add_no_duplicates_expectation method."""
        ge_validator.add_no_duplicates_expectation(column_list=["id", "name"])
        exp = ge_validator._expectations[0]
        assert exp.expectation_type == "expect_compound_columns_to_be_unique"
        assert exp.kwargs["column_list"] == ["id", "name"]


# ============================================================================
# Validation Tests (Fallback Mode)
# ============================================================================


class TestFallbackValidation:
    """Tests for fallback validation when GE is not available."""

    def test_validate_basic(self, ge_validator, sample_dataframe):
        """Test basic validation."""
        ge_validator.add_not_null_expectation("id")
        report = ge_validator.validate(sample_dataframe, dataset_name="test_data")

        assert isinstance(report, DataQualityReport)
        assert report.dataset_name == "test_data"
        assert len(report.validation_results) == 1

    def test_validate_not_null_passes(self, ge_validator, sample_dataframe):
        """Test not null validation passes for clean data."""
        ge_validator.add_not_null_expectation("id")
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_not_null_fails(self, ge_validator, dataframe_with_nulls):
        """Test not null validation fails for data with nulls."""
        ge_validator.add_not_null_expectation("name")
        report = ge_validator.validate(dataframe_with_nulls)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows > 0

    def test_validate_not_null_with_mostly(self, ge_validator, dataframe_with_nulls):
        """Test not null validation with mostly threshold."""
        ge_validator.add_not_null_expectation("name", mostly=0.75)
        report = ge_validator.validate(dataframe_with_nulls)

        result = report.validation_results[0]
        # 80% non-null should pass 75% threshold
        assert result.status == ValidationStatus.PASSED

    def test_validate_unique_passes(self, ge_validator, sample_dataframe):
        """Test unique validation passes for unique column."""
        ge_validator.add_unique_expectation("id")
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_unique_fails(self, ge_validator, dataframe_with_duplicates):
        """Test unique validation fails for non-unique column."""
        ge_validator.add_unique_expectation("id")
        report = ge_validator.validate(dataframe_with_duplicates)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows > 0

    def test_validate_range_passes(self, ge_validator, sample_dataframe):
        """Test range validation passes."""
        ge_validator.add_range_expectation("price", min_value=0, max_value=200)
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_range_fails(self, ge_validator, dataframe_with_outliers):
        """Test range validation fails for outliers."""
        ge_validator.add_range_expectation("value", min_value=0, max_value=100)
        report = ge_validator.validate(dataframe_with_outliers)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows > 0

    def test_validate_in_set_passes(self, ge_validator, sample_dataframe):
        """Test in set validation passes."""
        ge_validator.add_in_set_expectation("category", ["A", "B", "C"])
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_in_set_fails(self, ge_validator, sample_dataframe):
        """Test in set validation fails when values outside set."""
        ge_validator.add_in_set_expectation("category", ["A", "B"])  # Missing C
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED

    def test_validate_regex_passes(self, ge_validator):
        """Test regex validation passes."""
        df = pd.DataFrame({"email": ["test@example.com", "user@domain.org", "admin@test.io"]})
        ge_validator.add_regex_expectation("email", r"^[\w.-]+@[\w.-]+\.\w+$")
        report = ge_validator.validate(df)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_regex_fails(self, ge_validator):
        """Test regex validation fails."""
        df = pd.DataFrame({"email": ["test@example.com", "invalid_email", "user@domain.org"]})
        ge_validator.add_regex_expectation("email", r"^[\w.-]+@[\w.-]+\.\w+$")
        report = ge_validator.validate(df)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_rows == 1

    def test_validate_row_count_passes(self, ge_validator, sample_dataframe):
        """Test row count validation passes."""
        ge_validator.add_table_row_count_expectation(min_value=50, max_value=200)
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_row_count_fails(self, ge_validator, sample_dataframe):
        """Test row count validation fails."""
        ge_validator.add_table_row_count_expectation(min_value=500, max_value=1000)
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED

    def test_validate_column_exists_passes(self, ge_validator, sample_dataframe):
        """Test column exists validation passes."""
        ge_validator.add_column_exists_expectation("id")
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_column_exists_fails(self, ge_validator, sample_dataframe):
        """Test column exists validation fails."""
        ge_validator.add_column_exists_expectation("nonexistent_column")
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED

    def test_validate_compound_unique_passes(self, ge_validator, sample_dataframe):
        """Test compound unique validation passes."""
        ge_validator.add_no_duplicates_expectation()
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.PASSED

    def test_validate_compound_unique_fails(self, ge_validator, dataframe_with_duplicates):
        """Test compound unique validation fails."""
        ge_validator.add_no_duplicates_expectation()
        report = ge_validator.validate(dataframe_with_duplicates)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED


# ============================================================================
# Report Quality Tests
# ============================================================================


class TestReportQuality:
    """Tests for report quality metrics."""

    def test_quality_score_high(self, ge_validator, sample_dataframe):
        """Test quality score is high for clean data."""
        ge_validator.add_not_null_expectation("id")
        ge_validator.add_unique_expectation("id")
        ge_validator.add_range_expectation("price", min_value=0, max_value=200)
        report = ge_validator.validate(sample_dataframe)

        assert report.overall_score >= 90

    def test_quality_score_low(self, ge_validator, dataframe_with_nulls):
        """Test quality score is low for data with issues."""
        ge_validator.add_not_null_expectation("name")
        ge_validator.add_not_null_expectation("price")
        report = ge_validator.validate(dataframe_with_nulls)

        assert report.overall_score < 100

    def test_passed_failed_counts(self, ge_validator, sample_dataframe):
        """Test passed/failed counts are correct."""
        ge_validator.add_not_null_expectation("id")
        ge_validator.add_unique_expectation("id")
        ge_validator.add_in_set_expectation("category", ["A", "B"])  # Will fail
        report = ge_validator.validate(sample_dataframe)

        assert report.passed_checks == 2
        assert report.failed_checks == 1

    def test_recommendations_generated(self, ge_validator, dataframe_with_nulls):
        """Test recommendations are generated."""
        ge_validator.add_not_null_expectation("name")
        report = ge_validator.validate(dataframe_with_nulls)

        assert isinstance(report.recommendations, list)

    def test_statistics_included(self, ge_validator, sample_dataframe):
        """Test statistics are included in report."""
        ge_validator.add_not_null_expectation("id")
        report = ge_validator.validate(sample_dataframe, include_statistics=True)

        assert report.statistics is not None
        assert report.statistics.row_count == 100
        assert report.statistics.column_count == 6


# ============================================================================
# Schema-Based Validation Tests
# ============================================================================


class TestSchemaBasedValidation:
    """Tests for schema-based validation."""

    def test_from_schema(self, sample_dataframe):
        """Test configuring validator from schema."""
        schema = DataSchema(
            schema_name="test_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC, nullable=False, unique=True),
                SchemaField(name="name", data_type=DataType.TEXT, nullable=False),
                SchemaField(name="price", data_type=DataType.NUMERIC, min_value=0, max_value=200),
                SchemaField(
                    name="category",
                    data_type=DataType.CATEGORICAL,
                    allowed_values=["A", "B", "C"],
                ),
            ],
        )

        validator = GreatExpectationsValidator()
        validator.from_schema(schema)

        # Should have expectations for all fields
        assert len(validator._expectations) > 0

        report = validator.validate(sample_dataframe)
        assert isinstance(report, DataQualityReport)

    def test_schema_with_pattern(self):
        """Test schema with regex pattern."""
        df = pd.DataFrame({"email": ["test@example.com", "user@domain.org"]})

        schema = DataSchema(
            schema_name="email_schema",
            fields=[
                SchemaField(
                    name="email",
                    data_type=DataType.TEXT,
                    pattern=r"^[\w.-]+@[\w.-]+\.\w+$",
                ),
            ],
        )

        validator = GreatExpectationsValidator()
        validator.from_schema(schema)
        report = validator.validate(df)

        # All emails are valid
        assert all(r.status == ValidationStatus.PASSED for r in report.validation_results)


# ============================================================================
# File Validation Tests
# ============================================================================


class TestFileValidation:
    """Tests for file-based validation."""

    def test_validate_csv_file(self, ge_validator, sample_dataframe):
        """Test validating a CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)

            ge_validator.add_not_null_expectation("id")
            ge_validator.add_range_expectation("price", min_value=0, max_value=200)
            report = ge_validator.validate_file(f.name)

            assert isinstance(report, DataQualityReport)
            assert report.passed_checks == 2

            Path(f.name).unlink()

    def test_validate_parquet_file(self, ge_validator, sample_dataframe):
        """Test validating a Parquet file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            sample_dataframe.to_parquet(f.name, index=False)

            ge_validator.add_not_null_expectation("id")
            report = ge_validator.validate_file(f.name)

            assert isinstance(report, DataQualityReport)
            assert report.passed_checks == 1

            Path(f.name).unlink()

    def test_validate_nonexistent_file(self, ge_validator):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ge_validator.validate_file("/nonexistent/path/data.csv")

    def test_validate_unsupported_format(self, ge_validator):
        """Test error handling for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test data")

            with pytest.raises(ValueError, match="Unsupported file format"):
                ge_validator.validate_file(f.name)

            Path(f.name).unlink()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, ge_validator):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame({"id": [], "name": []})
        ge_validator.add_not_null_expectation("id")
        report = ge_validator.validate(df)

        assert isinstance(report, DataQualityReport)
        assert report.statistics.row_count == 0

    def test_missing_column(self, ge_validator, sample_dataframe):
        """Test validation with missing column."""
        ge_validator.add_not_null_expectation("nonexistent_column")
        report = ge_validator.validate(sample_dataframe)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.SKIPPED

    def test_all_null_column(self, ge_validator):
        """Test validation of all-null column."""
        df = pd.DataFrame({"id": [1, 2, 3], "nulls": [None, None, None]})
        ge_validator.add_not_null_expectation("nulls")
        report = ge_validator.validate(df)

        result = report.validation_results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.failed_percentage == 100.0

    def test_single_row_dataframe(self, ge_validator):
        """Test validation of single-row DataFrame."""
        df = pd.DataFrame({"id": [1], "name": ["test"]})
        ge_validator.add_unique_expectation("id")
        ge_validator.add_not_null_expectation("name")
        report = ge_validator.validate(df)

        assert all(r.status == ValidationStatus.PASSED for r in report.validation_results)

    def test_no_expectations(self, ge_validator, sample_dataframe):
        """Test validation with no expectations configured."""
        report = ge_validator.validate(sample_dataframe)

        assert isinstance(report, DataQualityReport)
        assert len(report.validation_results) == 0
        assert report.overall_score == 100.0

    def test_multiple_failures(self, ge_validator, dataframe_with_nulls):
        """Test handling multiple validation failures."""
        ge_validator.add_not_null_expectation("name")
        ge_validator.add_not_null_expectation("price")
        ge_validator.add_not_null_expectation("category")
        report = ge_validator.validate(dataframe_with_nulls)

        assert report.failed_checks >= 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_validation_workflow(self, sample_dataframe):
        """Test a complete validation workflow."""
        validator = GreatExpectationsValidator()

        # Add various expectations
        validator.add_column_exists_expectation("id")
        validator.add_not_null_expectation("id")
        validator.add_unique_expectation("id")
        validator.add_range_expectation("price", min_value=0, max_value=200)
        validator.add_range_expectation("rating", min_value=1, max_value=5)
        validator.add_in_set_expectation("category", ["A", "B", "C"])
        validator.add_table_row_count_expectation(min_value=50, max_value=200)

        report = validator.validate(sample_dataframe, dataset_name="integration_test")

        assert report.dataset_name == "integration_test"
        assert report.overall_score >= 80
        assert len(report.validation_results) == 7

    def test_schema_then_custom_expectations(self, sample_dataframe):
        """Test combining schema-based and custom expectations."""
        schema = DataSchema(
            schema_name="base_schema",
            fields=[
                SchemaField(name="id", data_type=DataType.NUMERIC, nullable=False),
                SchemaField(name="price", data_type=DataType.NUMERIC, min_value=0),
            ],
        )

        validator = GreatExpectationsValidator()
        validator.from_schema(schema)

        # Add additional custom expectations
        validator.add_unique_expectation("id")
        validator.add_in_set_expectation("category", ["A", "B", "C"])

        report = validator.validate(sample_dataframe)
        assert isinstance(report, DataQualityReport)

    def test_validation_with_warnings(self, sample_dataframe):
        """Test validation with warning-level expectations."""
        validator = GreatExpectationsValidator()

        validator.add_not_null_expectation("id", severity=ValidationSeverity.ERROR)
        validator.add_in_set_expectation(
            "category", ["A"], severity=ValidationSeverity.WARNING
        )  # Will fail

        report = validator.validate(sample_dataframe)

        assert report.passed_checks == 1
        assert report.warning_count == 1
        assert report.failed_checks == 0  # Warnings don't count as errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
