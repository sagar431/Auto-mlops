"""
Data Quality Models

Pydantic models for data quality validation, profiling, and reporting.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataType(str, Enum):
    """Supported data types for validation."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ColumnStatistics(BaseModel):
    """Statistical summary for a single column."""

    column_name: str = Field(..., description="Name of the column")
    data_type: DataType = Field(..., description="Detected data type")
    total_count: int = Field(..., description="Total number of values")
    null_count: int = Field(..., description="Number of null/missing values")
    null_percentage: float = Field(..., description="Percentage of null values")
    unique_count: int = Field(..., description="Number of unique values")
    unique_percentage: float = Field(..., description="Percentage of unique values")

    # Numeric statistics (only for numeric columns)
    mean: float | None = Field(None, description="Mean value")
    median: float | None = Field(None, description="Median value")
    std: float | None = Field(None, description="Standard deviation")
    min_value: float | None = Field(None, description="Minimum value")
    max_value: float | None = Field(None, description="Maximum value")
    q1: float | None = Field(None, description="First quartile (25%)")
    q3: float | None = Field(None, description="Third quartile (75%)")

    # Categorical statistics
    top_values: list[dict[str, Any]] = Field(
        default_factory=list, description="Most frequent values with counts"
    )


class DatasetStatistics(BaseModel):
    """Statistical summary for an entire dataset."""

    row_count: int = Field(..., description="Total number of rows")
    column_count: int = Field(..., description="Total number of columns")
    total_cells: int = Field(..., description="Total number of cells")
    total_missing: int = Field(..., description="Total missing values across all columns")
    missing_percentage: float = Field(..., description="Percentage of missing values")
    duplicate_rows: int = Field(..., description="Number of duplicate rows")
    duplicate_percentage: float = Field(..., description="Percentage of duplicate rows")
    memory_usage_bytes: int = Field(..., description="Memory usage in bytes")
    columns: list[ColumnStatistics] = Field(
        default_factory=list, description="Statistics for each column"
    )


class ValidationRule(BaseModel):
    """A single validation rule definition."""

    rule_id: str = Field(..., description="Unique identifier for the rule")
    rule_name: str = Field(..., description="Human-readable rule name")
    rule_type: str = Field(..., description="Type of validation rule")
    column: str | None = Field(None, description="Column this rule applies to, if any")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Rule-specific parameters")
    severity: ValidationSeverity = Field(
        ValidationSeverity.ERROR, description="Severity if rule fails"
    )


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    rule_id: str = Field(..., description="ID of the rule that was checked")
    rule_name: str = Field(..., description="Name of the rule")
    status: ValidationStatus = Field(..., description="Pass/fail status")
    severity: ValidationSeverity = Field(..., description="Severity level")
    column: str | None = Field(None, description="Column checked, if applicable")
    message: str = Field(..., description="Description of the result")
    actual_value: Any = Field(None, description="Actual value found")
    expected_value: Any = Field(None, description="Expected value or threshold")
    failed_rows: int = Field(0, description="Number of rows that failed this check")
    failed_percentage: float = Field(0.0, description="Percentage of rows that failed")


class DataQualityReport(BaseModel):
    """Complete data quality report for a dataset."""

    report_id: str = Field(..., description="Unique report identifier")
    dataset_name: str = Field(..., description="Name or path of the dataset")
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp when report was generated",
    )
    statistics: DatasetStatistics = Field(..., description="Dataset statistics")
    validation_results: list[ValidationResult] = Field(
        default_factory=list, description="Results of all validation checks"
    )
    passed_checks: int = Field(0, description="Number of passed validation checks")
    failed_checks: int = Field(0, description="Number of failed validation checks")
    warning_count: int = Field(0, description="Number of warnings")
    overall_score: float = Field(..., description="Overall data quality score (0-100)")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommended actions to improve data quality"
    )


class AnomalyType(str, Enum):
    """Types of data anomalies."""

    OUTLIER = "outlier"
    MISSING_PATTERN = "missing_pattern"
    DUPLICATE = "duplicate"
    FORMAT_ERROR = "format_error"
    DISTRIBUTION_SHIFT = "distribution_shift"
    UNEXPECTED_VALUE = "unexpected_value"


class AnomalyRecord(BaseModel):
    """A single detected anomaly."""

    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    column: str | None = Field(None, description="Column where anomaly was found")
    row_indices: list[int] = Field(default_factory=list, description="Row indices affected")
    description: str = Field(..., description="Description of the anomaly")
    severity: ValidationSeverity = Field(..., description="Severity level")
    value: Any = Field(None, description="The anomalous value(s)")
    expected_range: str | None = Field(None, description="Expected range or pattern")


class AnomalyDetectionResult(BaseModel):
    """Results from anomaly detection analysis."""

    dataset_name: str = Field(..., description="Name or path of the dataset")
    analyzed_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of analysis",
    )
    total_anomalies: int = Field(..., description="Total number of anomalies found")
    anomalies_by_type: dict[str, int] = Field(
        default_factory=dict, description="Count of anomalies by type"
    )
    anomalies: list[AnomalyRecord] = Field(
        default_factory=list, description="List of detected anomalies"
    )
    affected_rows: int = Field(..., description="Number of rows with anomalies")
    affected_percentage: float = Field(..., description="Percentage of rows with anomalies")


class SchemaField(BaseModel):
    """Schema definition for a single field."""

    name: str = Field(..., description="Field name")
    data_type: DataType = Field(..., description="Expected data type")
    nullable: bool = Field(True, description="Whether null values are allowed")
    unique: bool = Field(False, description="Whether values must be unique")
    min_value: float | None = Field(None, description="Minimum allowed value")
    max_value: float | None = Field(None, description="Maximum allowed value")
    allowed_values: list[Any] | None = Field(
        None, description="List of allowed values for categorical fields"
    )
    pattern: str | None = Field(None, description="Regex pattern for string validation")


class DataSchema(BaseModel):
    """Schema definition for a dataset."""

    schema_name: str = Field(..., description="Name of the schema")
    version: str = Field("1.0", description="Schema version")
    fields: list[SchemaField] = Field(..., description="Field definitions")
    strict: bool = Field(False, description="If true, extra columns are not allowed")


class SchemaValidationResult(BaseModel):
    """Result of schema validation."""

    schema_name: str = Field(..., description="Name of the schema used")
    dataset_name: str = Field(..., description="Name of the dataset validated")
    validated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of validation",
    )
    is_valid: bool = Field(..., description="Overall validation result")
    missing_columns: list[str] = Field(
        default_factory=list, description="Required columns that are missing"
    )
    extra_columns: list[str] = Field(default_factory=list, description="Columns not in schema")
    type_mismatches: list[dict[str, Any]] = Field(
        default_factory=list, description="Columns with wrong data types"
    )
    constraint_violations: list[ValidationResult] = Field(
        default_factory=list, description="Constraint validation failures"
    )
