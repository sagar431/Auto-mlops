"""
Data Validators

Provides validation rules and checks for ML dataset quality.
"""

import re
import uuid
from pathlib import Path
from typing import Any

from .models import (
    DataQualityReport,
    DataSchema,
    DatasetStatistics,
    DataType,
    SchemaValidationResult,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    ValidationStatus,
)


class DataValidator:
    """
    Validates datasets against quality rules and schema constraints.

    Supports:
    - Null/missing value checks
    - Uniqueness validation
    - Range validation for numeric columns
    - Pattern matching for string columns
    - Schema validation
    - Custom validation rules
    """

    def __init__(self):
        self._rules: list[ValidationRule] = []
        self._builtin_rules = self._create_builtin_rules()

    def _create_builtin_rules(self) -> dict[str, ValidationRule]:
        """Create built-in validation rules."""
        return {
            "no_nulls": ValidationRule(
                rule_id="no_nulls",
                rule_name="No Null Values",
                rule_type="null_check",
                parameters={"max_null_percentage": 0.0},
                severity=ValidationSeverity.ERROR,
            ),
            "max_nulls_10": ValidationRule(
                rule_id="max_nulls_10",
                rule_name="Max 10% Null Values",
                rule_type="null_check",
                parameters={"max_null_percentage": 10.0},
                severity=ValidationSeverity.WARNING,
            ),
            "unique_values": ValidationRule(
                rule_id="unique_values",
                rule_name="Unique Values",
                rule_type="uniqueness",
                parameters={"require_unique": True},
                severity=ValidationSeverity.ERROR,
            ),
            "no_duplicates": ValidationRule(
                rule_id="no_duplicates",
                rule_name="No Duplicate Rows",
                rule_type="row_duplicates",
                parameters={"max_duplicate_percentage": 0.0},
                severity=ValidationSeverity.WARNING,
            ),
        }

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self._rules.append(rule)

    def add_null_check(
        self,
        column: str,
        max_null_percentage: float = 0.0,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """Add a null value check for a column."""
        rule = ValidationRule(
            rule_id=f"null_check_{column}_{uuid.uuid4().hex[:8]}",
            rule_name=f"Null Check: {column}",
            rule_type="null_check",
            column=column,
            parameters={"max_null_percentage": max_null_percentage},
            severity=severity,
        )
        self._rules.append(rule)

    def add_range_check(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """Add a range check for a numeric column."""
        rule = ValidationRule(
            rule_id=f"range_check_{column}_{uuid.uuid4().hex[:8]}",
            rule_name=f"Range Check: {column}",
            rule_type="range",
            column=column,
            parameters={"min_value": min_value, "max_value": max_value},
            severity=severity,
        )
        self._rules.append(rule)

    def add_pattern_check(
        self,
        column: str,
        pattern: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """Add a regex pattern check for a string column."""
        rule = ValidationRule(
            rule_id=f"pattern_check_{column}_{uuid.uuid4().hex[:8]}",
            rule_name=f"Pattern Check: {column}",
            rule_type="pattern",
            column=column,
            parameters={"pattern": pattern},
            severity=severity,
        )
        self._rules.append(rule)

    def add_allowed_values_check(
        self,
        column: str,
        allowed_values: list[Any],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """Add an allowed values check for a categorical column."""
        rule = ValidationRule(
            rule_id=f"allowed_values_{column}_{uuid.uuid4().hex[:8]}",
            rule_name=f"Allowed Values: {column}",
            rule_type="allowed_values",
            column=column,
            parameters={"allowed_values": allowed_values},
            severity=severity,
        )
        self._rules.append(rule)

    def clear_rules(self) -> None:
        """Clear all custom validation rules."""
        self._rules.clear()

    def validate_dataframe(
        self,
        df: Any,
        dataset_name: str = "dataset",
        include_builtin: bool = True,
    ) -> DataQualityReport:
        """
        Validate a pandas DataFrame against all configured rules.

        Args:
            df: pandas DataFrame to validate
            dataset_name: Name for the dataset in the report
            include_builtin: Whether to include built-in rules

        Returns:
            DataQualityReport with validation results
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        results: list[ValidationResult] = []
        rules_to_check = list(self._rules)

        if include_builtin:
            # Add built-in rules for each column
            for col in df.columns:
                rules_to_check.append(
                    ValidationRule(
                        rule_id=f"null_check_{col}",
                        rule_name=f"Null Check: {col}",
                        rule_type="null_check",
                        column=col,
                        parameters={"max_null_percentage": 100.0},
                        severity=ValidationSeverity.INFO,
                    )
                )

        # Run validation rules
        for rule in rules_to_check:
            result = self._run_rule(df, rule)
            results.append(result)

        # Calculate statistics
        stats = self._calculate_statistics(df)

        # Calculate summary metrics
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(
            1
            for r in results
            if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.ERROR
        )
        warnings = sum(
            1
            for r in results
            if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.WARNING
        )

        # Calculate quality score
        total_checks = len(results)
        if total_checks > 0:
            # Weight errors more than warnings
            error_penalty = failed * 10
            warning_penalty = warnings * 3
            max_penalty = total_checks * 10
            score = max(0, 100 - (error_penalty + warning_penalty) / max_penalty * 100)
        else:
            score = 100.0

        # Generate recommendations
        recommendations = self._generate_recommendations(results, stats)

        return DataQualityReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            statistics=stats,
            validation_results=results,
            passed_checks=passed,
            failed_checks=failed,
            warning_count=warnings,
            overall_score=round(score, 2),
            recommendations=recommendations,
        )

    def _run_rule(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Run a single validation rule."""
        try:
            if rule.rule_type == "null_check":
                return self._check_nulls(df, rule)
            elif rule.rule_type == "uniqueness":
                return self._check_uniqueness(df, rule)
            elif rule.rule_type == "range":
                return self._check_range(df, rule)
            elif rule.rule_type == "pattern":
                return self._check_pattern(df, rule)
            elif rule.rule_type == "allowed_values":
                return self._check_allowed_values(df, rule)
            elif rule.rule_type == "row_duplicates":
                return self._check_row_duplicates(df, rule)
            else:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    status=ValidationStatus.SKIPPED,
                    severity=rule.severity,
                    column=rule.column,
                    message=f"Unknown rule type: {rule.rule_type}",
                )
        except Exception as e:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.ERROR,
                column=rule.column,
                message=f"Error running rule: {str(e)}",
            )

    def _check_nulls(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Check for null values in a column."""
        column = rule.column
        max_pct = rule.parameters.get("max_null_percentage", 0.0)

        if column and column not in df.columns:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                column=column,
                message=f"Column '{column}' not found in dataset",
            )

        if column:
            null_count = df[column].isnull().sum()
            total = len(df)
        else:
            null_count = df.isnull().sum().sum()
            total = df.size

        null_pct = (null_count / total * 100) if total > 0 else 0
        passed = null_pct <= max_pct

        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=rule.severity,
            column=column,
            message=f"Null percentage: {null_pct:.2f}% (max allowed: {max_pct}%)",
            actual_value=round(null_pct, 2),
            expected_value=max_pct,
            failed_rows=int(null_count),
            failed_percentage=round(null_pct, 2),
        )

    def _check_uniqueness(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Check for unique values in a column."""
        column = rule.column

        if not column or column not in df.columns:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                column=column,
                message=f"Column '{column}' not found or not specified",
            )

        total = len(df)
        unique_count = df[column].nunique()
        duplicate_count = total - unique_count
        is_unique = duplicate_count == 0

        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASSED if is_unique else ValidationStatus.FAILED,
            severity=rule.severity,
            column=column,
            message=f"Unique values: {unique_count}/{total}",
            actual_value=unique_count,
            expected_value=total,
            failed_rows=duplicate_count,
            failed_percentage=round(duplicate_count / total * 100, 2) if total > 0 else 0,
        )

    def _check_range(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Check if numeric values are within range."""
        column = rule.column
        min_val = rule.parameters.get("min_value")
        max_val = rule.parameters.get("max_value")

        if not column or column not in df.columns:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                column=column,
                message=f"Column '{column}' not found or not specified",
            )

        col_data = df[column].dropna()
        failed_count = 0

        if min_val is not None:
            failed_count += (col_data < min_val).sum()
        if max_val is not None:
            failed_count += (col_data > max_val).sum()

        total = len(col_data)
        passed = failed_count == 0

        range_str = f"[{min_val if min_val is not None else '-inf'}, {max_val if max_val is not None else 'inf'}]"

        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=rule.severity,
            column=column,
            message=f"Values outside range {range_str}: {failed_count}",
            actual_value=(
                f"min={col_data.min()}, max={col_data.max()}" if len(col_data) > 0 else "N/A"
            ),
            expected_value=range_str,
            failed_rows=int(failed_count),
            failed_percentage=round(failed_count / total * 100, 2) if total > 0 else 0,
        )

    def _check_pattern(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Check if string values match a pattern."""
        column = rule.column
        pattern = rule.parameters.get("pattern", ".*")

        if not column or column not in df.columns:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                column=column,
                message=f"Column '{column}' not found or not specified",
            )

        col_data = df[column].dropna().astype(str)
        regex = re.compile(pattern)
        matches = col_data.apply(lambda x: bool(regex.match(x)))
        failed_count = (~matches).sum()
        total = len(col_data)
        passed = failed_count == 0

        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=rule.severity,
            column=column,
            message=f"Values not matching pattern '{pattern}': {failed_count}",
            actual_value=failed_count,
            expected_value=0,
            failed_rows=int(failed_count),
            failed_percentage=round(failed_count / total * 100, 2) if total > 0 else 0,
        )

    def _check_allowed_values(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Check if values are in allowed set."""
        column = rule.column
        allowed = rule.parameters.get("allowed_values", [])

        if not column or column not in df.columns:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                column=column,
                message=f"Column '{column}' not found or not specified",
            )

        col_data = df[column].dropna()
        invalid = ~col_data.isin(allowed)
        failed_count = invalid.sum()
        total = len(col_data)
        passed = failed_count == 0

        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=rule.severity,
            column=column,
            message=f"Values not in allowed set: {failed_count}",
            actual_value=col_data[invalid].unique().tolist()[:5] if failed_count > 0 else [],
            expected_value=allowed,
            failed_rows=int(failed_count),
            failed_percentage=round(failed_count / total * 100, 2) if total > 0 else 0,
        )

    def _check_row_duplicates(self, df: Any, rule: ValidationRule) -> ValidationResult:
        """Check for duplicate rows."""
        max_pct = rule.parameters.get("max_duplicate_percentage", 0.0)

        duplicated = df.duplicated()
        duplicate_count = duplicated.sum()
        total = len(df)
        dup_pct = (duplicate_count / total * 100) if total > 0 else 0
        passed = dup_pct <= max_pct

        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=rule.severity,
            column=None,
            message=f"Duplicate rows: {duplicate_count} ({dup_pct:.2f}%)",
            actual_value=round(dup_pct, 2),
            expected_value=max_pct,
            failed_rows=int(duplicate_count),
            failed_percentage=round(dup_pct, 2),
        )

    def _calculate_statistics(self, df: Any) -> DatasetStatistics:
        """Calculate basic statistics for the dataset."""
        from .profiler import DataProfiler

        profiler = DataProfiler()
        return profiler.get_basic_statistics(df)

    def _generate_recommendations(
        self, results: list[ValidationResult], stats: DatasetStatistics
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check for high null percentages
        high_null_cols = [
            r.column
            for r in results
            if r.rule_name.startswith("Null Check")
            and r.actual_value
            and isinstance(r.actual_value, (int, float))
            and r.actual_value > 20
        ]
        if high_null_cols:
            recommendations.append(
                f"Consider imputation or removal for columns with high null rates: {', '.join(filter(None, high_null_cols))}"
            )

        # Check for duplicates
        if stats.duplicate_percentage > 5:
            recommendations.append(
                f"Dataset has {stats.duplicate_percentage:.1f}% duplicate rows. Consider deduplication."
            )

        # Check for failed validations
        failed_errors = [
            r
            for r in results
            if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.ERROR
        ]
        if failed_errors:
            recommendations.append(
                f"{len(failed_errors)} critical validation failures require attention before model training."
            )

        return recommendations

    def validate_schema(
        self, df: Any, schema: DataSchema, dataset_name: str = "dataset"
    ) -> SchemaValidationResult:
        """
        Validate a DataFrame against a schema definition.

        Args:
            df: pandas DataFrame to validate
            schema: DataSchema defining expected structure
            dataset_name: Name for the dataset

        Returns:
            SchemaValidationResult with validation details
        """
        missing_columns: list[str] = []
        extra_columns: list[str] = []
        type_mismatches: list[dict[str, Any]] = []
        constraint_violations: list[ValidationResult] = []

        schema_columns = {f.name for f in schema.fields}
        df_columns = set(df.columns)

        # Check for missing columns
        missing_columns = list(schema_columns - df_columns)

        # Check for extra columns
        extra_columns = list(df_columns - schema_columns)

        # Validate each field
        for field in schema.fields:
            if field.name not in df.columns:
                continue

            col_data = df[field.name]

            # Check data type
            detected_type = self._detect_dtype(col_data)
            if detected_type != field.data_type and field.data_type != DataType.UNKNOWN:
                type_mismatches.append(
                    {
                        "column": field.name,
                        "expected": field.data_type.value,
                        "actual": detected_type.value,
                    }
                )

            # Check nullable constraint
            if not field.nullable and col_data.isnull().any():
                constraint_violations.append(
                    ValidationResult(
                        rule_id=f"nullable_{field.name}",
                        rule_name=f"Non-null: {field.name}",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.ERROR,
                        column=field.name,
                        message=f"Column '{field.name}' contains null values but is marked non-nullable",
                        actual_value=int(col_data.isnull().sum()),
                        expected_value=0,
                    )
                )

            # Check unique constraint
            if field.unique and col_data.duplicated().any():
                constraint_violations.append(
                    ValidationResult(
                        rule_id=f"unique_{field.name}",
                        rule_name=f"Unique: {field.name}",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.ERROR,
                        column=field.name,
                        message=f"Column '{field.name}' contains duplicate values but is marked unique",
                        actual_value=int(col_data.duplicated().sum()),
                        expected_value=0,
                    )
                )

            # Check range constraints for numeric
            if field.data_type == DataType.NUMERIC:
                numeric_data = col_data.dropna()
                if field.min_value is not None and (numeric_data < field.min_value).any():
                    constraint_violations.append(
                        ValidationResult(
                            rule_id=f"min_{field.name}",
                            rule_name=f"Min Value: {field.name}",
                            status=ValidationStatus.FAILED,
                            severity=ValidationSeverity.ERROR,
                            column=field.name,
                            message=f"Values below minimum {field.min_value}",
                            actual_value=float(numeric_data.min()),
                            expected_value=field.min_value,
                        )
                    )
                if field.max_value is not None and (numeric_data > field.max_value).any():
                    constraint_violations.append(
                        ValidationResult(
                            rule_id=f"max_{field.name}",
                            rule_name=f"Max Value: {field.name}",
                            status=ValidationStatus.FAILED,
                            severity=ValidationSeverity.ERROR,
                            column=field.name,
                            message=f"Values above maximum {field.max_value}",
                            actual_value=float(numeric_data.max()),
                            expected_value=field.max_value,
                        )
                    )

            # Check allowed values
            if field.allowed_values is not None:
                invalid = ~col_data.dropna().isin(field.allowed_values)
                if invalid.any():
                    constraint_violations.append(
                        ValidationResult(
                            rule_id=f"allowed_{field.name}",
                            rule_name=f"Allowed Values: {field.name}",
                            status=ValidationStatus.FAILED,
                            severity=ValidationSeverity.ERROR,
                            column=field.name,
                            message="Invalid values found",
                            actual_value=col_data.dropna()[invalid].unique().tolist()[:5],
                            expected_value=field.allowed_values,
                        )
                    )

            # Check pattern
            if field.pattern is not None and field.data_type in (
                DataType.TEXT,
                DataType.CATEGORICAL,
            ):
                str_data = col_data.dropna().astype(str)
                regex = re.compile(field.pattern)
                invalid = ~str_data.apply(lambda x: bool(regex.match(x)))
                if invalid.any():
                    constraint_violations.append(
                        ValidationResult(
                            rule_id=f"pattern_{field.name}",
                            rule_name=f"Pattern: {field.name}",
                            status=ValidationStatus.FAILED,
                            severity=ValidationSeverity.ERROR,
                            column=field.name,
                            message=f"Values not matching pattern '{field.pattern}'",
                            actual_value=int(invalid.sum()),
                            expected_value=0,
                        )
                    )

        is_valid = (
            len(missing_columns) == 0
            and (len(extra_columns) == 0 or not schema.strict)
            and len(type_mismatches) == 0
            and len(constraint_violations) == 0
        )

        return SchemaValidationResult(
            schema_name=schema.schema_name,
            dataset_name=dataset_name,
            is_valid=is_valid,
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            type_mismatches=type_mismatches,
            constraint_violations=constraint_violations,
        )

    def _detect_dtype(self, series: Any) -> DataType:
        """Detect the data type of a pandas Series."""
        import pandas as pd

        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN
        elif pd.api.types.is_numeric_dtype(dtype):
            return DataType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return DataType.DATETIME
        elif isinstance(dtype, pd.CategoricalDtype):
            return DataType.CATEGORICAL
        elif pd.api.types.is_object_dtype(dtype):
            # Check if it's categorical (low cardinality)
            if series.nunique() / len(series) < 0.05 and series.nunique() < 50:
                return DataType.CATEGORICAL
            return DataType.TEXT
        return DataType.UNKNOWN

    def validate_file(
        self,
        file_path: str | Path,
        dataset_name: str | None = None,
    ) -> DataQualityReport:
        """
        Load and validate a data file.

        Supports CSV, Parquet, and JSON files.

        Args:
            file_path: Path to the data file
            dataset_name: Optional name for the dataset

        Returns:
            DataQualityReport with validation results
        """
        import pandas as pd

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        name = dataset_name or path.name

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return self.validate_dataframe(df, dataset_name=name)
