"""
Great Expectations Validator Integration

Provides integration with Great Expectations for advanced data validation
in ML pipelines. Wraps GE's validation capabilities with the existing
data quality models for seamless integration.
"""

import uuid
from pathlib import Path
from typing import Any

from .models import (
    DataQualityReport,
    DataSchema,
    DatasetStatistics,
    DataType,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


class ExpectationConfig:
    """Configuration for a Great Expectations expectation."""

    def __init__(
        self,
        expectation_type: str,
        column: str | None = None,
        kwargs: dict[str, Any] | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        description: str | None = None,
    ):
        """
        Initialize an expectation configuration.

        Args:
            expectation_type: The GE expectation type (e.g., "expect_column_values_to_not_be_null")
            column: Column to apply the expectation to (if applicable)
            kwargs: Additional keyword arguments for the expectation
            severity: Severity level if expectation fails
            description: Optional human-readable description
        """
        self.expectation_type = expectation_type
        self.column = column
        self.kwargs = kwargs or {}
        self.severity = severity
        self.description = description or expectation_type


class GreatExpectationsValidator:
    """
    Validator using Great Expectations for data quality checks.

    Features:
    - Run GE expectations on pandas DataFrames
    - Convert GE validation results to internal models
    - Support for common data quality checks
    - Integration with existing DataQualityReport format

    Example usage:
        from data_quality.validator import GreatExpectationsValidator

        validator = GreatExpectationsValidator()
        validator.add_expectation("expect_column_values_to_not_be_null", column="id")
        validator.add_expectation(
            "expect_column_values_to_be_between",
            column="price",
            kwargs={"min_value": 0, "max_value": 1000}
        )
        report = validator.validate(df)
    """

    def __init__(self):
        """Initialize the Great Expectations validator."""
        self._expectations: list[ExpectationConfig] = []
        self._ge_available = self._check_ge_available()

    def _check_ge_available(self) -> bool:
        """Check if Great Expectations is available."""
        try:
            import great_expectations as ge  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def ge_available(self) -> bool:
        """Return whether Great Expectations is available."""
        return self._ge_available

    def add_expectation(
        self,
        expectation_type: str,
        column: str | None = None,
        kwargs: dict[str, Any] | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        description: str | None = None,
    ) -> None:
        """
        Add a Great Expectations expectation.

        Args:
            expectation_type: The GE expectation type
            column: Column to apply expectation to
            kwargs: Additional arguments for the expectation
            severity: Severity if expectation fails
            description: Human-readable description
        """
        config = ExpectationConfig(
            expectation_type=expectation_type,
            column=column,
            kwargs=kwargs,
            severity=severity,
            description=description,
        )
        self._expectations.append(config)

    def add_not_null_expectation(
        self,
        column: str,
        mostly: float = 1.0,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that column values should not be null.

        Args:
            column: Column name
            mostly: Fraction of values that must be non-null (0.0-1.0)
            severity: Severity level if check fails
        """
        self.add_expectation(
            "expect_column_values_to_not_be_null",
            column=column,
            kwargs={"mostly": mostly} if mostly < 1.0 else None,
            severity=severity,
            description=f"Column '{column}' should not have null values",
        )

    def add_unique_expectation(
        self,
        column: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that column values should be unique.

        Args:
            column: Column name
            severity: Severity level if check fails
        """
        self.add_expectation(
            "expect_column_values_to_be_unique",
            column=column,
            severity=severity,
            description=f"Column '{column}' should have unique values",
        )

    def add_range_expectation(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
        strict_min: bool = False,
        strict_max: bool = False,
        mostly: float = 1.0,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that column values should be within a range.

        Args:
            column: Column name
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            strict_min: If True, values must be > min_value (not >=)
            strict_max: If True, values must be < max_value (not <=)
            mostly: Fraction of values that must satisfy condition
            severity: Severity level if check fails
        """
        kwargs: dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        if strict_min:
            kwargs["strict_min"] = True
        if strict_max:
            kwargs["strict_max"] = True
        if mostly < 1.0:
            kwargs["mostly"] = mostly

        range_str = f"[{min_value if min_value is not None else '-∞'}, {max_value if max_value is not None else '∞'}]"
        self.add_expectation(
            "expect_column_values_to_be_between",
            column=column,
            kwargs=kwargs,
            severity=severity,
            description=f"Column '{column}' values should be in range {range_str}",
        )

    def add_in_set_expectation(
        self,
        column: str,
        value_set: list[Any],
        mostly: float = 1.0,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that column values should be in a set of allowed values.

        Args:
            column: Column name
            value_set: List of allowed values
            mostly: Fraction of values that must be in set
            severity: Severity level if check fails
        """
        kwargs: dict[str, Any] = {"value_set": value_set}
        if mostly < 1.0:
            kwargs["mostly"] = mostly

        self.add_expectation(
            "expect_column_values_to_be_in_set",
            column=column,
            kwargs=kwargs,
            severity=severity,
            description=f"Column '{column}' values should be in {value_set}",
        )

    def add_regex_expectation(
        self,
        column: str,
        regex: str,
        mostly: float = 1.0,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that column values should match a regex pattern.

        Args:
            column: Column name
            regex: Regular expression pattern
            mostly: Fraction of values that must match
            severity: Severity level if check fails
        """
        kwargs: dict[str, Any] = {"regex": regex}
        if mostly < 1.0:
            kwargs["mostly"] = mostly

        self.add_expectation(
            "expect_column_values_to_match_regex",
            column=column,
            kwargs=kwargs,
            severity=severity,
            description=f"Column '{column}' values should match pattern '{regex}'",
        )

    def add_column_type_expectation(
        self,
        column: str,
        type_list: list[str],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that column should have specific data types.

        Args:
            column: Column name
            type_list: List of acceptable type names (e.g., ["int64", "int32"])
            severity: Severity level if check fails
        """
        self.add_expectation(
            "expect_column_values_to_be_in_type_list",
            column=column,
            kwargs={"type_list": type_list},
            severity=severity,
            description=f"Column '{column}' should have type in {type_list}",
        )

    def add_table_row_count_expectation(
        self,
        min_value: int | None = None,
        max_value: int | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation for table row count.

        Args:
            min_value: Minimum expected row count
            max_value: Maximum expected row count
            severity: Severity level if check fails
        """
        kwargs: dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value

        self.add_expectation(
            "expect_table_row_count_to_be_between",
            kwargs=kwargs,
            severity=severity,
            description=f"Row count should be between {min_value} and {max_value}",
        )

    def add_column_exists_expectation(
        self,
        column: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """
        Add an expectation that a column should exist.

        Args:
            column: Column name
            severity: Severity level if check fails
        """
        self.add_expectation(
            "expect_column_to_exist",
            column=column,
            severity=severity,
            description=f"Column '{column}' should exist",
        )

    def add_no_duplicates_expectation(
        self,
        column_list: list[str] | None = None,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
    ) -> None:
        """
        Add an expectation that there should be no duplicate rows.

        Args:
            column_list: Columns to check for duplicates (None = all columns)
            severity: Severity level if check fails
        """
        kwargs = {}
        if column_list:
            kwargs["column_list"] = column_list

        self.add_expectation(
            "expect_compound_columns_to_be_unique",
            kwargs=kwargs,
            severity=severity,
            description="Dataset should not have duplicate rows",
        )

    def clear_expectations(self) -> None:
        """Remove all configured expectations."""
        self._expectations.clear()

    def validate(
        self,
        df: Any,
        dataset_name: str = "dataset",
        include_statistics: bool = True,
    ) -> DataQualityReport:
        """
        Validate a DataFrame using Great Expectations.

        Args:
            df: pandas DataFrame to validate
            dataset_name: Name for the dataset in the report
            include_statistics: Whether to include dataset statistics

        Returns:
            DataQualityReport with validation results
        """
        if not self._ge_available:
            return self._validate_fallback(df, dataset_name, include_statistics)

        return self._validate_with_ge(df, dataset_name, include_statistics)

    def _validate_with_ge(
        self,
        df: Any,
        dataset_name: str,
        include_statistics: bool,
    ) -> DataQualityReport:
        """Validate using Great Expectations."""
        import great_expectations as ge

        # Create GE dataset
        ge_df = ge.from_pandas(df)

        results: list[ValidationResult] = []
        validation_results_raw = []

        # Run each expectation
        for exp in self._expectations:
            try:
                # Build expectation method call
                exp_method = getattr(ge_df, exp.expectation_type, None)
                if exp_method is None:
                    results.append(
                        ValidationResult(
                            rule_id=f"ge_{exp.expectation_type}_{uuid.uuid4().hex[:8]}",
                            rule_name=exp.description,
                            status=ValidationStatus.SKIPPED,
                            severity=exp.severity,
                            column=exp.column,
                            message=f"Unknown expectation type: {exp.expectation_type}",
                        )
                    )
                    continue

                # Call expectation with appropriate args
                if exp.column:
                    result = exp_method(exp.column, **exp.kwargs)
                else:
                    result = exp_method(**exp.kwargs)

                validation_results_raw.append(result)

                # Convert to ValidationResult
                validation_result = self._convert_ge_result(result, exp)
                results.append(validation_result)

            except Exception as e:
                results.append(
                    ValidationResult(
                        rule_id=f"ge_{exp.expectation_type}_{uuid.uuid4().hex[:8]}",
                        rule_name=exp.description,
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.ERROR,
                        column=exp.column,
                        message=f"Error running expectation: {str(e)}",
                    )
                )

        # Calculate statistics
        stats = self._calculate_statistics(df) if include_statistics else self._empty_statistics()

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
        score = self._calculate_score(results)

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

    def _validate_fallback(
        self,
        df: Any,
        dataset_name: str,
        include_statistics: bool,
    ) -> DataQualityReport:
        """
        Fallback validation when Great Expectations is not available.

        Uses basic pandas operations to simulate common expectations.
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        results: list[ValidationResult] = []

        for exp in self._expectations:
            result = self._run_fallback_expectation(df, exp)
            results.append(result)

        # Calculate statistics
        stats = self._calculate_statistics(df) if include_statistics else self._empty_statistics()

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

        score = self._calculate_score(results)
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

    def _run_fallback_expectation(self, df: Any, exp: ExpectationConfig) -> ValidationResult:
        """Run a single expectation using pandas fallback logic."""
        rule_id = f"ge_{exp.expectation_type}_{uuid.uuid4().hex[:8]}"
        column = exp.column

        try:
            if exp.expectation_type == "expect_column_values_to_not_be_null":
                return self._fallback_not_null(df, exp, rule_id)
            elif exp.expectation_type == "expect_column_values_to_be_unique":
                return self._fallback_unique(df, exp, rule_id)
            elif exp.expectation_type == "expect_column_values_to_be_between":
                return self._fallback_between(df, exp, rule_id)
            elif exp.expectation_type == "expect_column_values_to_be_in_set":
                return self._fallback_in_set(df, exp, rule_id)
            elif exp.expectation_type == "expect_column_values_to_match_regex":
                return self._fallback_regex(df, exp, rule_id)
            elif exp.expectation_type == "expect_table_row_count_to_be_between":
                return self._fallback_row_count(df, exp, rule_id)
            elif exp.expectation_type == "expect_column_to_exist":
                return self._fallback_column_exists(df, exp, rule_id)
            elif exp.expectation_type == "expect_compound_columns_to_be_unique":
                return self._fallback_compound_unique(df, exp, rule_id)
            else:
                return ValidationResult(
                    rule_id=rule_id,
                    rule_name=exp.description,
                    status=ValidationStatus.SKIPPED,
                    severity=exp.severity,
                    column=column,
                    message=f"Expectation type '{exp.expectation_type}' not supported in fallback mode",
                )
        except Exception as e:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.ERROR,
                column=column,
                message=f"Error: {str(e)}",
            )

    def _fallback_not_null(self, df: Any, exp: ExpectationConfig, rule_id: str) -> ValidationResult:
        """Fallback for not null expectation."""
        column = exp.column
        if column not in df.columns:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.SKIPPED,
                severity=exp.severity,
                column=column,
                message=f"Column '{column}' not found",
            )

        mostly = exp.kwargs.get("mostly", 1.0)
        null_count = df[column].isnull().sum()
        total = len(df)
        non_null_ratio = (total - null_count) / total if total > 0 else 1.0
        passed = non_null_ratio >= mostly

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=column,
            message=f"Non-null ratio: {non_null_ratio:.2%} (required: {mostly:.2%})",
            actual_value=round(non_null_ratio, 4),
            expected_value=mostly,
            failed_rows=int(null_count),
            failed_percentage=round((1 - non_null_ratio) * 100, 2),
        )

    def _fallback_unique(self, df: Any, exp: ExpectationConfig, rule_id: str) -> ValidationResult:
        """Fallback for unique expectation."""
        column = exp.column
        if column not in df.columns:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.SKIPPED,
                severity=exp.severity,
                column=column,
                message=f"Column '{column}' not found",
            )

        duplicates = df[column].duplicated().sum()
        total = len(df)
        passed = duplicates == 0

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=column,
            message=f"Duplicate values: {duplicates}",
            actual_value=int(duplicates),
            expected_value=0,
            failed_rows=int(duplicates),
            failed_percentage=round(duplicates / total * 100, 2) if total > 0 else 0,
        )

    def _fallback_between(self, df: Any, exp: ExpectationConfig, rule_id: str) -> ValidationResult:
        """Fallback for between expectation."""
        column = exp.column
        if column not in df.columns:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.SKIPPED,
                severity=exp.severity,
                column=column,
                message=f"Column '{column}' not found",
            )

        min_val = exp.kwargs.get("min_value")
        max_val = exp.kwargs.get("max_value")
        strict_min = exp.kwargs.get("strict_min", False)
        strict_max = exp.kwargs.get("strict_max", False)
        mostly = exp.kwargs.get("mostly", 1.0)

        col_data = df[column].dropna()
        total = len(col_data)
        if total == 0:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.PASSED,
                severity=exp.severity,
                column=column,
                message="No non-null values to check",
            )

        failed_mask = False
        if min_val is not None:
            if strict_min:
                failed_mask = failed_mask | (col_data <= min_val)
            else:
                failed_mask = failed_mask | (col_data < min_val)
        if max_val is not None:
            if strict_max:
                failed_mask = failed_mask | (col_data >= max_val)
            else:
                failed_mask = failed_mask | (col_data > max_val)

        failed_count = failed_mask.sum() if hasattr(failed_mask, "sum") else 0
        success_ratio = (total - failed_count) / total if total > 0 else 1.0
        passed = success_ratio >= mostly

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=column,
            message=f"Success ratio: {success_ratio:.2%} (required: {mostly:.2%})",
            actual_value=round(success_ratio, 4),
            expected_value=mostly,
            failed_rows=int(failed_count),
            failed_percentage=round((1 - success_ratio) * 100, 2),
        )

    def _fallback_in_set(self, df: Any, exp: ExpectationConfig, rule_id: str) -> ValidationResult:
        """Fallback for in set expectation."""
        column = exp.column
        if column not in df.columns:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.SKIPPED,
                severity=exp.severity,
                column=column,
                message=f"Column '{column}' not found",
            )

        value_set = exp.kwargs.get("value_set", [])
        mostly = exp.kwargs.get("mostly", 1.0)

        col_data = df[column].dropna()
        total = len(col_data)
        if total == 0:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.PASSED,
                severity=exp.severity,
                column=column,
                message="No non-null values to check",
            )

        in_set = col_data.isin(value_set).sum()
        success_ratio = in_set / total if total > 0 else 1.0
        passed = success_ratio >= mostly
        failed_count = total - in_set

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=column,
            message=f"Values in set: {success_ratio:.2%} (required: {mostly:.2%})",
            actual_value=round(success_ratio, 4),
            expected_value=mostly,
            failed_rows=int(failed_count),
            failed_percentage=round((1 - success_ratio) * 100, 2),
        )

    def _fallback_regex(self, df: Any, exp: ExpectationConfig, rule_id: str) -> ValidationResult:
        """Fallback for regex expectation."""
        import re

        column = exp.column
        if column not in df.columns:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.SKIPPED,
                severity=exp.severity,
                column=column,
                message=f"Column '{column}' not found",
            )

        regex_pattern = exp.kwargs.get("regex", ".*")
        mostly = exp.kwargs.get("mostly", 1.0)

        col_data = df[column].dropna().astype(str)
        total = len(col_data)
        if total == 0:
            return ValidationResult(
                rule_id=rule_id,
                rule_name=exp.description,
                status=ValidationStatus.PASSED,
                severity=exp.severity,
                column=column,
                message="No non-null values to check",
            )

        regex = re.compile(regex_pattern)
        matches = col_data.apply(lambda x: bool(regex.match(x))).sum()
        success_ratio = matches / total if total > 0 else 1.0
        passed = success_ratio >= mostly
        failed_count = total - matches

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=column,
            message=f"Matching values: {success_ratio:.2%} (required: {mostly:.2%})",
            actual_value=round(success_ratio, 4),
            expected_value=mostly,
            failed_rows=int(failed_count),
            failed_percentage=round((1 - success_ratio) * 100, 2),
        )

    def _fallback_row_count(
        self, df: Any, exp: ExpectationConfig, rule_id: str
    ) -> ValidationResult:
        """Fallback for row count expectation."""
        min_value = exp.kwargs.get("min_value")
        max_value = exp.kwargs.get("max_value")
        row_count = len(df)

        passed = True
        if min_value is not None and row_count < min_value:
            passed = False
        if max_value is not None and row_count > max_value:
            passed = False

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=None,
            message=f"Row count: {row_count} (expected: [{min_value}, {max_value}])",
            actual_value=row_count,
            expected_value=f"[{min_value}, {max_value}]",
        )

    def _fallback_column_exists(
        self, df: Any, exp: ExpectationConfig, rule_id: str
    ) -> ValidationResult:
        """Fallback for column exists expectation."""
        column = exp.column
        exists = column in df.columns

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if exists else ValidationStatus.FAILED,
            severity=exp.severity,
            column=column,
            message=f"Column '{column}' {'exists' if exists else 'does not exist'}",
            actual_value=exists,
            expected_value=True,
        )

    def _fallback_compound_unique(
        self, df: Any, exp: ExpectationConfig, rule_id: str
    ) -> ValidationResult:
        """Fallback for compound unique expectation."""
        column_list = exp.kwargs.get("column_list")

        if column_list:
            # Check if all columns exist
            missing = [c for c in column_list if c not in df.columns]
            if missing:
                return ValidationResult(
                    rule_id=rule_id,
                    rule_name=exp.description,
                    status=ValidationStatus.SKIPPED,
                    severity=exp.severity,
                    column=None,
                    message=f"Missing columns: {missing}",
                )
            duplicates = df.duplicated(subset=column_list).sum()
        else:
            duplicates = df.duplicated().sum()

        total = len(df)
        passed = duplicates == 0

        return ValidationResult(
            rule_id=rule_id,
            rule_name=exp.description,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=exp.severity,
            column=None,
            message=f"Duplicate rows: {duplicates}",
            actual_value=int(duplicates),
            expected_value=0,
            failed_rows=int(duplicates),
            failed_percentage=round(duplicates / total * 100, 2) if total > 0 else 0,
        )

    def _convert_ge_result(self, ge_result: Any, exp: ExpectationConfig) -> ValidationResult:
        """Convert a Great Expectations result to ValidationResult."""
        success = ge_result.success
        result_dict = ge_result.result if hasattr(ge_result, "result") else {}

        # Extract failure details
        unexpected_count = result_dict.get("unexpected_count", 0)
        element_count = result_dict.get("element_count", 0)
        unexpected_percent = result_dict.get("unexpected_percent", 0)

        # Determine actual value
        actual_value = result_dict.get("observed_value")
        if actual_value is None and element_count > 0:
            actual_value = round(1 - (unexpected_count / element_count), 4)

        return ValidationResult(
            rule_id=f"ge_{exp.expectation_type}_{uuid.uuid4().hex[:8]}",
            rule_name=exp.description,
            status=ValidationStatus.PASSED if success else ValidationStatus.FAILED,
            severity=exp.severity,
            column=exp.column,
            message=f"{'Passed' if success else 'Failed'}: {exp.description}",
            actual_value=actual_value,
            expected_value=result_dict.get("expected_value"),
            failed_rows=int(unexpected_count) if unexpected_count else 0,
            failed_percentage=round(unexpected_percent, 2) if unexpected_percent else 0,
        )

    def _calculate_statistics(self, df: Any) -> DatasetStatistics:
        """Calculate dataset statistics."""
        from .profiler import DataProfiler

        profiler = DataProfiler()
        return profiler.get_basic_statistics(df)

    def _empty_statistics(self) -> DatasetStatistics:
        """Return empty statistics."""
        return DatasetStatistics(
            row_count=0,
            column_count=0,
            total_cells=0,
            total_missing=0,
            missing_percentage=0,
            duplicate_rows=0,
            duplicate_percentage=0,
            memory_usage_bytes=0,
            columns=[],
        )

    def _calculate_score(self, results: list[ValidationResult]) -> float:
        """Calculate overall quality score."""
        total_checks = len(results)
        if total_checks == 0:
            return 100.0

        failed_errors = sum(
            1
            for r in results
            if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.ERROR
        )
        failed_warnings = sum(
            1
            for r in results
            if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.WARNING
        )

        # Weight errors more than warnings
        error_penalty = failed_errors * 10
        warning_penalty = failed_warnings * 3
        max_penalty = total_checks * 10

        score = max(0, 100 - (error_penalty + warning_penalty) / max_penalty * 100)
        return score

    def _generate_recommendations(
        self,
        results: list[ValidationResult],
        stats: DatasetStatistics,
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Analyze failed checks
        failed_nulls = [
            r.column
            for r in results
            if r.status == ValidationStatus.FAILED and "null" in r.rule_name.lower() and r.column
        ]
        if failed_nulls:
            recommendations.append(f"Address null values in columns: {', '.join(failed_nulls)}")

        failed_range = [
            r.column
            for r in results
            if r.status == ValidationStatus.FAILED and "range" in r.rule_name.lower() and r.column
        ]
        if failed_range:
            recommendations.append(f"Review value ranges in columns: {', '.join(failed_range)}")

        failed_unique = [
            r.column
            for r in results
            if r.status == ValidationStatus.FAILED and "unique" in r.rule_name.lower() and r.column
        ]
        if failed_unique:
            recommendations.append(
                f"Check for duplicate values in columns: {', '.join(failed_unique)}"
            )

        # High failure rate
        total = len(results)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        if total > 0 and failed / total > 0.3:
            recommendations.append(
                "High failure rate detected. Consider reviewing data pipeline or source."
            )

        # Duplicate rows
        if stats.duplicate_percentage > 5:
            recommendations.append(
                f"Dataset has {stats.duplicate_percentage:.1f}% duplicate rows. Consider deduplication."
            )

        return recommendations

    def from_schema(self, schema: DataSchema) -> "GreatExpectationsValidator":
        """
        Configure validator from a DataSchema.

        Args:
            schema: DataSchema defining expected structure

        Returns:
            Self for method chaining
        """
        for field in schema.fields:
            # Column existence
            self.add_column_exists_expectation(field.name)

            # Nullable
            if not field.nullable:
                self.add_not_null_expectation(field.name)

            # Unique
            if field.unique:
                self.add_unique_expectation(field.name)

            # Range for numeric
            if field.data_type == DataType.NUMERIC:
                if field.min_value is not None or field.max_value is not None:
                    self.add_range_expectation(
                        field.name,
                        min_value=field.min_value,
                        max_value=field.max_value,
                    )

            # Allowed values
            if field.allowed_values:
                self.add_in_set_expectation(field.name, field.allowed_values)

            # Pattern
            if field.pattern:
                self.add_regex_expectation(field.name, field.pattern)

        return self

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

        return self.validate(df, dataset_name=name)
