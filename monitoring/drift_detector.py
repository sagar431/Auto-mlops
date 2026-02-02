"""
Drift Detector with Evidently Integration.

Provides data drift and model drift detection capabilities using Evidently AI
library, with fallback implementations for basic drift detection when
Evidently is not available.
"""

import uuid
from datetime import datetime
from typing import Any

from .models import (
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
)


class DriftDetector:
    """
    Data and model drift detector using Evidently AI.

    Features:
    - Detect data drift between reference and current datasets
    - Support for multiple statistical tests (KS, Chi-squared, PSI, etc.)
    - Feature-level and dataset-level drift reporting
    - Configurable thresholds and severity levels
    - Fallback to basic statistics when Evidently is not available

    Example usage:
        from monitoring import DriftDetector

        detector = DriftDetector()

        # Compare reference and current data
        report = detector.detect_drift(
            reference_data=training_df,
            current_data=production_df,
            feature_columns=["age", "income", "category"],
        )

        print(f"Drift detected: {report.overall_drift_detected}")
        print(f"Severity: {report.severity}")
    """

    def __init__(
        self,
        drift_threshold: float = 0.1,
        stattest: str = "ks",
        per_feature_stattest: dict[str, str] | None = None,
    ):
        """
        Initialize the drift detector.

        Args:
            drift_threshold: Threshold for drift detection (0-1, lower = stricter)
            stattest: Default statistical test to use
                Options: "ks" (Kolmogorov-Smirnov), "chisquare", "psi", "wasserstein",
                        "kl_div", "jensenshannon", "anderson", "cramer_von_mises"
            per_feature_stattest: Override stattest for specific features
        """
        self.drift_threshold = drift_threshold
        self.stattest = stattest
        self.per_feature_stattest = per_feature_stattest or {}
        self._evidently_available = self._check_evidently_available()

    def _check_evidently_available(self) -> bool:
        """Check if Evidently is available."""
        try:
            import evidently  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def evidently_available(self) -> bool:
        """Return whether Evidently is available."""
        return self._evidently_available

    def detect_drift(
        self,
        reference_data: Any,
        current_data: Any,
        feature_columns: list[str] | None = None,
        dataset_name: str = "dataset",
        categorical_columns: list[str] | None = None,
        numerical_columns: list[str] | None = None,
    ) -> DriftReport:
        """
        Detect data drift between reference and current datasets.

        Args:
            reference_data: Reference/baseline pandas DataFrame
            current_data: Current/production pandas DataFrame
            feature_columns: Columns to check for drift (None = all)
            dataset_name: Name for the dataset in the report
            categorical_columns: Explicitly specify categorical columns
            numerical_columns: Explicitly specify numerical columns

        Returns:
            DriftReport with detailed drift analysis
        """
        if self._evidently_available:
            return self._detect_drift_evidently(
                reference_data,
                current_data,
                feature_columns,
                dataset_name,
                categorical_columns,
                numerical_columns,
            )
        else:
            return self._detect_drift_fallback(
                reference_data,
                current_data,
                feature_columns,
                dataset_name,
            )

    def _detect_drift_evidently(
        self,
        reference_data: Any,
        current_data: Any,
        feature_columns: list[str] | None,
        dataset_name: str,
        categorical_columns: list[str] | None,
        numerical_columns: list[str] | None,
    ) -> DriftReport:
        """Detect drift using Evidently."""
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        # Determine columns to analyze
        if feature_columns:
            ref_df = reference_data[feature_columns].copy()
            cur_df = current_data[feature_columns].copy()
        else:
            ref_df = reference_data.copy()
            cur_df = current_data.copy()
            feature_columns = list(ref_df.columns)

        # Set up column mapping
        column_mapping = ColumnMapping()
        if categorical_columns:
            column_mapping.categorical_features = [
                c for c in categorical_columns if c in feature_columns
            ]
        if numerical_columns:
            column_mapping.numerical_features = [
                c for c in numerical_columns if c in feature_columns
            ]

        # Create and run Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_df,
            current_data=cur_df,
            column_mapping=column_mapping if (categorical_columns or numerical_columns) else None,
        )

        # Extract results from Evidently report
        report_dict = report.as_dict()
        metrics = report_dict.get("metrics", [])

        # Find the dataset drift metric
        dataset_drift_result = None
        for metric in metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                dataset_drift_result = metric.get("result", {})
                break

        if not dataset_drift_result:
            return self._detect_drift_fallback(
                reference_data, current_data, feature_columns, dataset_name
            )

        # Extract overall drift info
        drift_share = dataset_drift_result.get("share_of_drifted_columns", 0.0)
        overall_drift = dataset_drift_result.get("dataset_drift", False)
        drift_by_columns = dataset_drift_result.get("drift_by_columns", {})

        # Build feature results
        feature_results = []
        for col_name, col_result in drift_by_columns.items():
            feature_results.append(
                FeatureDriftResult(
                    feature_name=col_name,
                    drift_detected=col_result.get("drift_detected", False),
                    drift_score=col_result.get("drift_score", 0.0),
                    stattest_name=col_result.get("stattest_name", self.stattest),
                    stattest_threshold=col_result.get("stattest_threshold", self.drift_threshold),
                    p_value=col_result.get("p_value"),
                    reference_distribution=col_result.get("reference", {}).get("distribution", {}),
                    current_distribution=col_result.get("current", {}).get("distribution", {}),
                )
            )

        # Calculate severity
        severity = self._calculate_severity(drift_share, overall_drift)

        # Generate recommendations
        recommendations = self._generate_recommendations(feature_results, drift_share, severity)

        return DriftReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            timestamp=datetime.utcnow(),
            drift_type=DriftType.DATA,
            overall_drift_detected=overall_drift,
            drift_share=drift_share,
            severity=severity,
            feature_results=feature_results,
            reference_rows=len(reference_data),
            current_rows=len(current_data),
            recommendations=recommendations,
        )

    def _detect_drift_fallback(
        self,
        reference_data: Any,
        current_data: Any,
        feature_columns: list[str] | None,
        dataset_name: str,
    ) -> DriftReport:
        """
        Fallback drift detection using basic statistics.

        Uses Kolmogorov-Smirnov test for numerical columns and
        chi-squared test for categorical columns.
        """
        import pandas as pd
        from scipy import stats

        if not isinstance(reference_data, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        # Determine columns to analyze
        if feature_columns:
            columns = [c for c in feature_columns if c in reference_data.columns]
        else:
            columns = list(reference_data.columns)

        feature_results = []
        drifted_count = 0

        for col in columns:
            if col not in current_data.columns:
                continue

            ref_col = reference_data[col].dropna()
            cur_col = current_data[col].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                continue

            # Determine if categorical or numerical
            is_numeric = pd.api.types.is_numeric_dtype(ref_col)

            if is_numeric:
                # Kolmogorov-Smirnov test for numerical columns
                statistic, p_value = stats.ks_2samp(ref_col, cur_col)
                test_name = "ks"
            else:
                # Chi-squared test for categorical columns
                ref_counts = ref_col.value_counts()
                cur_counts = cur_col.value_counts()

                # Align categories
                all_cats = set(ref_counts.index) | set(cur_counts.index)
                ref_freq = [ref_counts.get(c, 0) for c in all_cats]
                cur_freq = [cur_counts.get(c, 0) for c in all_cats]

                # Normalize to same total
                ref_total = sum(ref_freq)
                cur_total = sum(cur_freq)
                if ref_total > 0 and cur_total > 0:
                    ref_freq = [f * cur_total / ref_total for f in ref_freq]

                try:
                    statistic, p_value = stats.chisquare(cur_freq, f_exp=ref_freq)
                    # Normalize statistic to 0-1 range approximately
                    statistic = min(1.0, statistic / (len(all_cats) * 10))
                except Exception:
                    statistic, p_value = 0.0, 1.0
                test_name = "chisquare"

            # Determine if drift detected based on threshold
            drift_detected = p_value < self.drift_threshold
            if drift_detected:
                drifted_count += 1

            # Compute basic distribution stats
            ref_dist = None
            cur_dist = None
            if is_numeric:
                ref_dist = {
                    "mean": float(ref_col.mean()),
                    "std": float(ref_col.std()),
                    "min": float(ref_col.min()),
                    "max": float(ref_col.max()),
                }
                cur_dist = {
                    "mean": float(cur_col.mean()),
                    "std": float(cur_col.std()),
                    "min": float(cur_col.min()),
                    "max": float(cur_col.max()),
                }

            feature_results.append(
                FeatureDriftResult(
                    feature_name=col,
                    drift_detected=drift_detected,
                    drift_score=float(statistic),
                    stattest_name=test_name,
                    stattest_threshold=self.drift_threshold,
                    p_value=float(p_value) if p_value is not None else None,
                    reference_distribution=ref_dist,
                    current_distribution=cur_dist,
                )
            )

        # Calculate overall drift
        total_features = len(feature_results)
        drift_share = drifted_count / total_features if total_features > 0 else 0.0
        overall_drift = drift_share > 0.5

        # Calculate severity
        severity = self._calculate_severity(drift_share, overall_drift)

        # Generate recommendations
        recommendations = self._generate_recommendations(feature_results, drift_share, severity)

        return DriftReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            timestamp=datetime.utcnow(),
            drift_type=DriftType.DATA,
            overall_drift_detected=overall_drift,
            drift_share=drift_share,
            severity=severity,
            feature_results=feature_results,
            reference_rows=len(reference_data),
            current_rows=len(current_data),
            recommendations=recommendations,
        )

    def _calculate_severity(self, drift_share: float, overall_drift: bool) -> DriftSeverity:
        """Calculate drift severity based on drift share."""
        if drift_share == 0:
            return DriftSeverity.NONE
        elif drift_share < 0.1:
            return DriftSeverity.LOW
        elif drift_share < 0.3:
            return DriftSeverity.MEDIUM
        elif drift_share < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _generate_recommendations(
        self,
        feature_results: list[FeatureDriftResult],
        drift_share: float,
        severity: DriftSeverity,
    ) -> list[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []

        # Get drifted features
        drifted = [f for f in feature_results if f.drift_detected]

        if not drifted:
            return ["No significant drift detected. Continue monitoring."]

        # High drift features
        high_drift = [f for f in drifted if f.drift_score > 0.5]
        if high_drift:
            names = ", ".join(f.feature_name for f in high_drift[:5])
            recommendations.append(
                f"High drift detected in features: {names}. Investigate root cause."
            )

        # Severity-based recommendations
        if severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            recommendations.append(
                "Consider retraining the model with recent data to address drift."
            )
            recommendations.append("Review data pipeline for potential data quality issues.")

        if severity == DriftSeverity.CRITICAL:
            recommendations.append(
                "Critical drift level: Consider immediate model refresh or rollback."
            )

        # General recommendations
        if drift_share > 0.3:
            recommendations.append(
                "Multiple features affected. Check for systemic changes in data source."
            )

        if len(drifted) > 0 and len(drifted) <= 3:
            recommendations.append(
                f"Focused drift in {len(drifted)} feature(s). "
                "May indicate targeted data collection changes."
            )

        return recommendations

    def detect_prediction_drift(
        self,
        reference_predictions: Any,
        current_predictions: Any,
        prediction_column: str = "prediction",
        dataset_name: str = "predictions",
    ) -> DriftReport:
        """
        Detect drift in model predictions.

        Args:
            reference_predictions: Reference predictions (DataFrame or array)
            current_predictions: Current predictions (DataFrame or array)
            prediction_column: Column name for predictions
            dataset_name: Name for the dataset in the report

        Returns:
            DriftReport for prediction drift
        """
        import pandas as pd

        # Convert to DataFrame if needed
        if not isinstance(reference_predictions, pd.DataFrame):
            reference_predictions = pd.DataFrame({prediction_column: reference_predictions})
        if not isinstance(current_predictions, pd.DataFrame):
            current_predictions = pd.DataFrame({prediction_column: current_predictions})

        # Detect drift on predictions
        report = self.detect_drift(
            reference_data=reference_predictions,
            current_data=current_predictions,
            feature_columns=[prediction_column],
            dataset_name=dataset_name,
        )

        # Update drift type
        report.drift_type = DriftType.PREDICTION

        return report


class ConceptDriftDetector:
    """
    Concept drift detector for monitoring model-target relationship changes.

    Concept drift occurs when the relationship between input features and
    the target variable changes over time, even if the input distribution
    remains stable.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the concept drift detector.

        Args:
            significance_level: Statistical significance level for drift detection
        """
        self.significance_level = significance_level

    def detect_concept_drift(
        self,
        reference_data: Any,
        current_data: Any,
        target_column: str,
        prediction_column: str | None = None,
        feature_columns: list[str] | None = None,
        dataset_name: str = "dataset",
    ) -> DriftReport:
        """
        Detect concept drift by comparing error distributions.

        Args:
            reference_data: Reference dataset with features and target
            current_data: Current dataset with features and target
            target_column: Name of the target column
            prediction_column: Name of the prediction column (if available)
            feature_columns: Feature columns to analyze
            dataset_name: Name for the report

        Returns:
            DriftReport with concept drift analysis
        """
        import pandas as pd
        from scipy import stats

        if target_column not in reference_data.columns:
            raise ValueError(f"Target column '{target_column}' not in reference data")
        if target_column not in current_data.columns:
            raise ValueError(f"Target column '{target_column}' not in current data")

        feature_results = []
        drifted_count = 0

        # If predictions are available, compare error distributions
        if prediction_column and prediction_column in reference_data.columns:
            ref_errors = (reference_data[target_column] - reference_data[prediction_column]).abs()
            cur_errors = (current_data[target_column] - current_data[prediction_column]).abs()

            statistic, p_value = stats.ks_2samp(ref_errors, cur_errors)
            drift_detected = p_value < self.significance_level
            if drift_detected:
                drifted_count += 1

            feature_results.append(
                FeatureDriftResult(
                    feature_name="prediction_error",
                    drift_detected=drift_detected,
                    drift_score=float(statistic),
                    stattest_name="ks",
                    stattest_threshold=self.significance_level,
                    p_value=float(p_value),
                    reference_distribution={
                        "mean_error": float(ref_errors.mean()),
                        "std_error": float(ref_errors.std()),
                    },
                    current_distribution={
                        "mean_error": float(cur_errors.mean()),
                        "std_error": float(cur_errors.std()),
                    },
                )
            )

        # Compare target distribution
        ref_target = reference_data[target_column].dropna()
        cur_target = current_data[target_column].dropna()

        is_numeric = pd.api.types.is_numeric_dtype(ref_target)

        if is_numeric:
            statistic, p_value = stats.ks_2samp(ref_target, cur_target)
        else:
            # For categorical targets, use chi-squared
            ref_counts = ref_target.value_counts()
            cur_counts = cur_target.value_counts()
            all_cats = set(ref_counts.index) | set(cur_counts.index)
            ref_freq = [ref_counts.get(c, 0) for c in all_cats]
            cur_freq = [cur_counts.get(c, 0) for c in all_cats]

            ref_total = sum(ref_freq)
            cur_total = sum(cur_freq)
            if ref_total > 0 and cur_total > 0:
                ref_freq = [f * cur_total / ref_total for f in ref_freq]

            try:
                statistic, p_value = stats.chisquare(cur_freq, f_exp=ref_freq)
                statistic = min(1.0, statistic / (len(all_cats) * 10))
            except Exception:
                statistic, p_value = 0.0, 1.0

        drift_detected = p_value < self.significance_level
        if drift_detected:
            drifted_count += 1

        feature_results.append(
            FeatureDriftResult(
                feature_name=target_column,
                drift_detected=drift_detected,
                drift_score=float(statistic),
                stattest_name="ks" if is_numeric else "chisquare",
                stattest_threshold=self.significance_level,
                p_value=float(p_value),
                reference_distribution=None,
                current_distribution=None,
            )
        )

        # Calculate overall drift
        total_checks = len(feature_results)
        drift_share = drifted_count / total_checks if total_checks > 0 else 0.0
        overall_drift = drift_share > 0.5

        # Calculate severity
        if drift_share == 0:
            severity = DriftSeverity.NONE
        elif drift_share < 0.3:
            severity = DriftSeverity.LOW
        elif drift_share < 0.6:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        recommendations = []
        if overall_drift:
            recommendations.append(
                "Concept drift detected. The relationship between features and target may have changed."
            )
            recommendations.append("Consider retraining the model with recent labeled data.")

        return DriftReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            timestamp=datetime.utcnow(),
            drift_type=DriftType.CONCEPT,
            overall_drift_detected=overall_drift,
            drift_share=drift_share,
            severity=severity,
            feature_results=feature_results,
            reference_rows=len(reference_data),
            current_rows=len(current_data),
            recommendations=recommendations,
        )
