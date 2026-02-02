"""
Data Profiler

Provides statistical profiling and anomaly detection for ML datasets.
"""

from typing import Any

import numpy as np

from .models import (
    AnomalyDetectionResult,
    AnomalyRecord,
    AnomalyType,
    ColumnStatistics,
    DatasetStatistics,
    DataType,
    ValidationSeverity,
)


class DataProfiler:
    """
    Profiles datasets to provide statistical summaries and detect anomalies.

    Features:
    - Column-level statistics (mean, median, std, quartiles)
    - Missing value analysis
    - Duplicate detection
    - Outlier detection using IQR and Z-score methods
    - Distribution analysis
    """

    def __init__(self, outlier_threshold: float = 1.5, zscore_threshold: float = 3.0):
        """
        Initialize the profiler.

        Args:
            outlier_threshold: IQR multiplier for outlier detection (default 1.5)
            zscore_threshold: Z-score threshold for outlier detection (default 3.0)
        """
        self.outlier_threshold = outlier_threshold
        self.zscore_threshold = zscore_threshold

    def profile(self, df: Any, dataset_name: str = "dataset") -> DatasetStatistics:
        """
        Generate a complete statistical profile of a DataFrame.

        Args:
            df: pandas DataFrame to profile
            dataset_name: Name for the dataset

        Returns:
            DatasetStatistics with complete profile
        """
        return self.get_basic_statistics(df)

    def get_basic_statistics(self, df: Any) -> DatasetStatistics:
        """
        Calculate basic statistics for a DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            DatasetStatistics with column-level and dataset-level stats
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        row_count = len(df)
        column_count = len(df.columns)
        total_cells = row_count * column_count
        total_missing = int(df.isnull().sum().sum())
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        # Count duplicate rows
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = (duplicate_rows / row_count * 100) if row_count > 0 else 0

        # Memory usage
        memory_bytes = int(df.memory_usage(deep=True).sum())

        # Profile each column
        columns: list[ColumnStatistics] = []
        for col in df.columns:
            col_stats = self._profile_column(df[col], col)
            columns.append(col_stats)

        return DatasetStatistics(
            row_count=row_count,
            column_count=column_count,
            total_cells=total_cells,
            total_missing=total_missing,
            missing_percentage=round(missing_pct, 2),
            duplicate_rows=duplicate_rows,
            duplicate_percentage=round(duplicate_pct, 2),
            memory_usage_bytes=memory_bytes,
            columns=columns,
        )

    def _profile_column(self, series: Any, col_name: str) -> ColumnStatistics:
        """Profile a single column."""
        import pandas as pd

        total = len(series)
        null_count = int(series.isnull().sum())
        null_pct = (null_count / total * 100) if total > 0 else 0
        unique_count = int(series.nunique())
        unique_pct = (unique_count / total * 100) if total > 0 else 0

        # Detect data type
        data_type = self._detect_dtype(series)

        # Initialize numeric stats as None
        mean = median = std = min_val = max_val = q1 = q3 = None
        top_values: list[dict[str, Any]] = []

        if data_type == DataType.NUMERIC:
            numeric_data = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_data) > 0:
                mean = float(numeric_data.mean())
                median = float(numeric_data.median())
                std = float(numeric_data.std())
                min_val = float(numeric_data.min())
                max_val = float(numeric_data.max())
                q1 = float(numeric_data.quantile(0.25))
                q3 = float(numeric_data.quantile(0.75))

        elif data_type in (DataType.CATEGORICAL, DataType.TEXT, DataType.BOOLEAN):
            # Get top values
            value_counts = series.value_counts().head(10)
            top_values = [
                {
                    "value": str(val),
                    "count": int(count),
                    "percentage": round(count / total * 100, 2),
                }
                for val, count in value_counts.items()
            ]

        return ColumnStatistics(
            column_name=col_name,
            data_type=data_type,
            total_count=total,
            null_count=null_count,
            null_percentage=round(null_pct, 2),
            unique_count=unique_count,
            unique_percentage=round(unique_pct, 2),
            mean=round(mean, 4) if mean is not None else None,
            median=round(median, 4) if median is not None else None,
            std=round(std, 4) if std is not None else None,
            min_value=round(min_val, 4) if min_val is not None else None,
            max_value=round(max_val, 4) if max_val is not None else None,
            q1=round(q1, 4) if q1 is not None else None,
            q3=round(q3, 4) if q3 is not None else None,
            top_values=top_values,
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
            non_null = series.dropna()
            if len(non_null) == 0:
                return DataType.UNKNOWN
            # Check if it's categorical (low cardinality)
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.05 and series.nunique() < 50:
                return DataType.CATEGORICAL
            return DataType.TEXT
        return DataType.UNKNOWN

    def detect_anomalies(
        self,
        df: Any,
        dataset_name: str = "dataset",
        methods: list[str] | None = None,
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in the dataset.

        Args:
            df: pandas DataFrame to analyze
            dataset_name: Name for the dataset
            methods: List of detection methods to use. Options:
                - "iqr": Interquartile range method for numeric outliers
                - "zscore": Z-score method for numeric outliers
                - "missing": Detect missing value patterns
                - "duplicates": Detect duplicate rows
            If None, uses all methods.

        Returns:
            AnomalyDetectionResult with detected anomalies
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")

        if methods is None:
            methods = ["iqr", "zscore", "missing", "duplicates"]

        anomalies: list[AnomalyRecord] = []
        affected_rows: set[int] = set()

        # Detect numeric outliers
        if "iqr" in methods or "zscore" in methods:
            for col in df.select_dtypes(include=[np.number]).columns:
                col_anomalies = self._detect_numeric_outliers(
                    df, col, use_iqr="iqr" in methods, use_zscore="zscore" in methods
                )
                for anomaly in col_anomalies:
                    anomalies.append(anomaly)
                    affected_rows.update(anomaly.row_indices)

        # Detect missing value patterns
        if "missing" in methods:
            missing_anomalies = self._detect_missing_patterns(df)
            for anomaly in missing_anomalies:
                anomalies.append(anomaly)
                affected_rows.update(anomaly.row_indices)

        # Detect duplicates
        if "duplicates" in methods:
            dup_anomalies = self._detect_duplicates(df)
            for anomaly in dup_anomalies:
                anomalies.append(anomaly)
                affected_rows.update(anomaly.row_indices)

        # Count anomalies by type
        anomalies_by_type: dict[str, int] = {}
        for a in anomalies:
            key = a.anomaly_type.value
            anomalies_by_type[key] = anomalies_by_type.get(key, 0) + 1

        total_rows = len(df)
        affected_count = len(affected_rows)
        affected_pct = (affected_count / total_rows * 100) if total_rows > 0 else 0

        return AnomalyDetectionResult(
            dataset_name=dataset_name,
            total_anomalies=len(anomalies),
            anomalies_by_type=anomalies_by_type,
            anomalies=anomalies,
            affected_rows=affected_count,
            affected_percentage=round(affected_pct, 2),
        )

    def _detect_numeric_outliers(
        self,
        df: Any,
        column: str,
        use_iqr: bool = True,
        use_zscore: bool = True,
    ) -> list[AnomalyRecord]:
        """Detect outliers in a numeric column."""
        import pandas as pd

        anomalies: list[AnomalyRecord] = []
        col_data = pd.to_numeric(df[column], errors="coerce")
        valid_data = col_data.dropna()

        if len(valid_data) < 4:
            return anomalies

        # IQR method
        if use_iqr:
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr

            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_indices = col_data[outlier_mask].index.tolist()

            if outlier_indices:
                outlier_values = col_data[outlier_mask].tolist()
                anomalies.append(
                    AnomalyRecord(
                        anomaly_type=AnomalyType.OUTLIER,
                        column=column,
                        row_indices=outlier_indices[:100],  # Limit to first 100
                        description=f"IQR outliers in '{column}': {len(outlier_indices)} values outside [{lower_bound:.2f}, {upper_bound:.2f}]",
                        severity=ValidationSeverity.WARNING,
                        value=outlier_values[:10],  # Show first 10 values
                        expected_range=f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                    )
                )

        # Z-score method
        if use_zscore:
            mean = valid_data.mean()
            std = valid_data.std()

            if std > 0:
                z_scores = (col_data - mean) / std
                outlier_mask = z_scores.abs() > self.zscore_threshold
                outlier_indices = col_data[outlier_mask].index.tolist()

                # Only add if different from IQR outliers
                if outlier_indices and (not use_iqr or len(outlier_indices) > 0):
                    # Filter to only new outliers not caught by IQR
                    if use_iqr and anomalies:
                        existing = set(anomalies[-1].row_indices)
                        new_indices = [i for i in outlier_indices if i not in existing]
                        if new_indices:
                            anomalies.append(
                                AnomalyRecord(
                                    anomaly_type=AnomalyType.OUTLIER,
                                    column=column,
                                    row_indices=new_indices[:100],
                                    description=f"Z-score outliers in '{column}': {len(new_indices)} values with |z| > {self.zscore_threshold}",
                                    severity=ValidationSeverity.WARNING,
                                    value=col_data[new_indices[:10]].tolist(),
                                    expected_range=f"z-score within [-{self.zscore_threshold}, {self.zscore_threshold}]",
                                )
                            )
                    else:
                        anomalies.append(
                            AnomalyRecord(
                                anomaly_type=AnomalyType.OUTLIER,
                                column=column,
                                row_indices=outlier_indices[:100],
                                description=f"Z-score outliers in '{column}': {len(outlier_indices)} values with |z| > {self.zscore_threshold}",
                                severity=ValidationSeverity.WARNING,
                                value=col_data[outlier_indices[:10]].tolist(),
                                expected_range=f"z-score within [-{self.zscore_threshold}, {self.zscore_threshold}]",
                            )
                        )

        return anomalies

    def _detect_missing_patterns(self, df: Any) -> list[AnomalyRecord]:
        """Detect unusual patterns in missing values."""
        anomalies: list[AnomalyRecord] = []

        # Find rows with many missing values
        missing_per_row = df.isnull().sum(axis=1)
        threshold = len(df.columns) * 0.5  # More than 50% missing

        high_missing_rows = missing_per_row[missing_per_row > threshold]
        if len(high_missing_rows) > 0:
            anomalies.append(
                AnomalyRecord(
                    anomaly_type=AnomalyType.MISSING_PATTERN,
                    column=None,
                    row_indices=high_missing_rows.index.tolist()[:100],
                    description=f"{len(high_missing_rows)} rows have >50% missing values",
                    severity=ValidationSeverity.WARNING,
                    value=None,
                    expected_range="<50% missing per row",
                )
            )

        # Find columns with high missing rates
        missing_per_col = df.isnull().sum()
        col_threshold = len(df) * 0.3  # More than 30% missing

        high_missing_cols = missing_per_col[missing_per_col > col_threshold]
        for col, count in high_missing_cols.items():
            pct = count / len(df) * 100
            anomalies.append(
                AnomalyRecord(
                    anomaly_type=AnomalyType.MISSING_PATTERN,
                    column=str(col),
                    row_indices=[],
                    description=f"Column '{col}' has {pct:.1f}% missing values ({count} rows)",
                    severity=ValidationSeverity.WARNING if pct < 50 else ValidationSeverity.ERROR,
                    value=count,
                    expected_range="<30% missing",
                )
            )

        return anomalies

    def _detect_duplicates(self, df: Any) -> list[AnomalyRecord]:
        """Detect duplicate rows."""
        anomalies: list[AnomalyRecord] = []

        duplicated = df.duplicated(keep="first")
        dup_indices = df[duplicated].index.tolist()

        if dup_indices:
            anomalies.append(
                AnomalyRecord(
                    anomaly_type=AnomalyType.DUPLICATE,
                    column=None,
                    row_indices=dup_indices[:100],
                    description=f"{len(dup_indices)} duplicate rows found",
                    severity=ValidationSeverity.WARNING,
                    value=len(dup_indices),
                    expected_range="0 duplicates",
                )
            )

        return anomalies

    def compare_distributions(
        self,
        df_reference: Any,
        df_current: Any,
        columns: list[str] | None = None,
    ) -> list[AnomalyRecord]:
        """
        Compare distributions between reference and current datasets.

        Useful for detecting data drift.

        Args:
            df_reference: Reference DataFrame (e.g., training data)
            df_current: Current DataFrame to compare
            columns: Columns to compare (default: all numeric columns)

        Returns:
            List of AnomalyRecords for distribution shifts
        """
        import pandas as pd
        from scipy import stats

        anomalies: list[AnomalyRecord] = []

        if columns is None:
            # Use common numeric columns
            ref_numeric = set(df_reference.select_dtypes(include=[np.number]).columns)
            cur_numeric = set(df_current.select_dtypes(include=[np.number]).columns)
            columns = list(ref_numeric & cur_numeric)

        for col in columns:
            ref_data = pd.to_numeric(df_reference[col], errors="coerce").dropna()
            cur_data = pd.to_numeric(df_current[col], errors="coerce").dropna()

            if len(ref_data) < 10 or len(cur_data) < 10:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)

            if p_value < 0.05:  # Significant distribution difference
                anomalies.append(
                    AnomalyRecord(
                        anomaly_type=AnomalyType.DISTRIBUTION_SHIFT,
                        column=col,
                        row_indices=[],
                        description=f"Distribution shift detected in '{col}' (KS statistic: {ks_stat:.3f}, p-value: {p_value:.4f})",
                        severity=(
                            ValidationSeverity.WARNING
                            if p_value > 0.01
                            else ValidationSeverity.ERROR
                        ),
                        value={
                            "ks_statistic": round(ks_stat, 4),
                            "p_value": round(p_value, 4),
                            "ref_mean": round(ref_data.mean(), 4),
                            "cur_mean": round(cur_data.mean(), 4),
                        },
                        expected_range="p-value > 0.05",
                    )
                )

        return anomalies
