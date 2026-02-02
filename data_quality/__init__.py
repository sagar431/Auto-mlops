"""
Data Quality Module for MLOps Agent

Provides data profiling, validation, and quality checks for ML pipelines.

Components:
- DataValidator: Input validation and data quality checks against rules and schemas
- DataProfiler: Statistical profiling and anomaly detection

Models:
- DataQualityReport: Complete quality report with statistics and validation results
- ValidationResult: Result of a single validation check
- AnomalyDetectionResult: Results from anomaly detection analysis
- DatasetStatistics: Statistical summary of a dataset
- ColumnStatistics: Statistics for a single column
- DataSchema: Schema definition for dataset validation
- SchemaValidationResult: Result of schema validation

Example usage:
    from data_quality import DataValidator, DataProfiler

    # Validate a DataFrame
    validator = DataValidator()
    validator.add_null_check("price", max_null_percentage=5.0)
    validator.add_range_check("price", min_value=0)
    report = validator.validate_dataframe(df)

    # Profile a dataset
    profiler = DataProfiler()
    stats = profiler.profile(df)
    anomalies = profiler.detect_anomalies(df)
"""

from .models import (
    AnomalyDetectionResult,
    AnomalyRecord,
    AnomalyType,
    ColumnStatistics,
    DataQualityReport,
    DataSchema,
    DatasetStatistics,
    DataType,
    SchemaField,
    SchemaValidationResult,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    ValidationStatus,
)
from .profiler import DataProfiler
from .validators import DataValidator

__all__ = [
    # Core classes
    "DataValidator",
    "DataProfiler",
    # Report models
    "DataQualityReport",
    "ValidationResult",
    "AnomalyDetectionResult",
    "AnomalyRecord",
    # Statistics models
    "DatasetStatistics",
    "ColumnStatistics",
    # Schema models
    "DataSchema",
    "SchemaField",
    "SchemaValidationResult",
    # Rule models
    "ValidationRule",
    # Enums
    "DataType",
    "ValidationSeverity",
    "ValidationStatus",
    "AnomalyType",
]
