"""
Data preprocessing module for Supply Chain Explorer.

This module provides the DataPreprocessor class responsible for cleaning,
transforming, and preparing raw supply chain data for feature engineering
and machine learning. It handles missing values, outliers, data type
conversions, and categorical standardization.

Preprocessing is critical because:
1. ML models cannot handle missing values (NaN)
2. Outliers can skew model training and reduce accuracy
3. Incorrect data types cause calculation errors
4. Inconsistent categories lead to encoding problems

Author: Luca Gozzi 
Date: November 2025
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    DATE_COLUMNS,
    REQUIRED_COLUMNS,
)


# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingReport:
    """
    Container for preprocessing operation results.
    
    Tracks all transformations applied to the data for transparency
    and debugging. This is important for reproducibility and for
    explaining data transformations in your technical report.
    
    Attributes:
        initial_rows: Number of rows before preprocessing.
        final_rows: Number of rows after preprocessing.
        rows_removed: Number of rows removed during preprocessing.
        missing_values_handled: Dict mapping column to number of imputed values.
        outliers_handled: Dict mapping column to number of capped values.
        duplicates_removed: Number of duplicate rows removed.
        transformations: List of transformation descriptions.
    """
    initial_rows: int = 0
    final_rows: int = 0
    rows_removed: int = 0
    missing_values_handled: Dict[str, int] = field(default_factory=dict)
    outliers_handled: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    transformations: List[str] = field(default_factory=list)
    
    def add_transformation(self, description: str) -> None:
        """Record a transformation that was applied."""
        self.transformations.append(description)
        logger.info(f"Transformation: {description}")
    
    def summary(self) -> str:
        """Generate a human-readable summary of preprocessing."""
        lines = [
            "=" * 60,
            "PREPROCESSING REPORT",
            "=" * 60,
            f"Initial rows: {self.initial_rows:,}",
            f"Final rows: {self.final_rows:,}",
            f"Rows removed: {self.rows_removed:,} ({self.rows_removed/max(self.initial_rows,1)*100:.2f}%)",
            f"Duplicates removed: {self.duplicates_removed:,}",
            "-" * 60,
        ]
        
        if self.missing_values_handled:
            lines.append("MISSING VALUES HANDLED:")
            for col, count in self.missing_values_handled.items():
                lines.append(f"  {col}: {count:,} values imputed")
        
        if self.outliers_handled:
            lines.append("OUTLIERS HANDLED:")
            for col, count in self.outliers_handled.items():
                lines.append(f"  {col}: {count:,} values capped")
        
        if self.transformations:
            lines.append("-" * 60)
            lines.append("TRANSFORMATIONS APPLIED:")
            for i, t in enumerate(self.transformations, 1):
                lines.append(f"  {i}. {t}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class DataPreprocessor:
    """
    Preprocesses supply chain data for machine learning.
    
    This class encapsulates all data cleaning and transformation logic.
    It follows the principle of immutability - the original DataFrame
    is not modified; instead, a cleaned copy is returned.
    
    The preprocessing pipeline includes:
    1. Duplicate removal
    2. Missing value imputation
    3. Outlier detection and capping
    4. Data type standardization
    5. Categorical value cleaning
    
    Attributes:
        df: The DataFrame to preprocess.
        report: PreprocessingReport tracking all changes.
        
    Example:
        >>> preprocessor = DataPreprocessor(raw_df)
        >>> clean_df = preprocessor.preprocess()
        >>> print(preprocessor.report.summary())
    """
    
    # Columns where missing values should cause row removal
    # (too important to impute)
    CRITICAL_COLUMNS: List[str] = [
        TARGET_COLUMN,  # Cannot impute the target variable
        "Order Id",  # Need order identifier
    ]
    
    # Numerical columns with their imputation strategies
    # Options: 'median', 'mean', 'zero', 'mode'
    NUMERICAL_IMPUTATION: Dict[str, str] = {
        "Days for shipping (real)": "median",
        "Days for shipment (scheduled)": "median",
        "Order Item Discount Rate": "zero",  # Missing discount = no discount
        "Order Item Quantity": "median",
        "Order Item Product Price": "median",
        "Sales": "median",
        "Product Price": "median",
    }
    
    # Categorical columns with their imputation strategies
    CATEGORICAL_IMPUTATION: Dict[str, str] = {
        "Shipping Mode": "mode",  # Most common shipping mode
        "Category Name": "Unknown",  # Explicit unknown category
        "Market": "mode",
        "Order Region": "mode",
        "Order Country": "mode",
        "Order City": "mode",
        "Customer Segment": "mode",
    }
    
    # Columns to check for outliers (using IQR method)
    # Format: column_name: (lower_percentile, upper_percentile)
    # Add this constant at the top of the class (after OUTLIER_COLUMNS)
    
    # Columns to drop (not useful for analysis)
    COLUMNS_TO_DROP: List[str] = [
        "Product Description",  # 100% missing
        "Order Zipcode",  # 86% missing, not useful
        "Customer Lname",  # PII, not useful for ML
        "Customer Fname",  # PII, not useful for ML  
        "Customer Email",  # PII, not useful for ML
        "Customer Password",  # PII, should never use
        "Customer Street",  # PII, not useful
        "Customer Zipcode",  # Mostly missing
        "Order Item Cardprod Id",  # Internal ID
        "Order Item Id",  # Internal ID
        "Product Card Id",  # Internal ID
        "Product Category Id",  # Use Category Name instead
        "Department Id",  # Use Department Name instead
        "Customer Id",  # Internal ID
        "Order Customer Id",  # Internal ID
        "Product Image",  # Not useful
        "Order Profit Per Order",  # Could cause data leakage (calculated after delivery)
        "Benefit per order",  # Could cause data leakage
    ]
    OUTLIER_COLUMNS: Dict[str, Tuple[float, float]] = {
        "Days for shipping (real)": (0.01, 0.99),
        "Days for shipment (scheduled)": (0.01, 0.99),
        "Order Item Quantity": (0.0, 0.99),  # No lower bound (1 is valid)
        "Order Item Product Price": (0.0, 0.99),
        "Sales": (0.0, 0.99),
    }
    
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the preprocessor with a DataFrame.
        
        Args:
            df: The raw DataFrame to preprocess.
        """
        self.df = df.copy()  # Work on a copy to preserve original
        self.report = PreprocessingReport(initial_rows=len(df))
        
        logger.info(f"DataPreprocessor initialized with {len(df):,} rows")
    
    def preprocess(
        self,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        standardize_categories: bool = True,
        drop_unused: bool = True
    ) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.
        
        This is the main entry point that orchestrates all preprocessing
        steps in the correct order. Each step can be toggled on/off.
        
        Args:
            remove_duplicates: Whether to remove duplicate rows.
            handle_missing: Whether to handle missing values.
            handle_outliers: Whether to cap outliers.
            standardize_categories: Whether to clean categorical values.
            
        Returns:
            pd.DataFrame: The preprocessed DataFrame.
            
        Example:
            >>> preprocessor = DataPreprocessor(raw_df)
            >>> clean_df = preprocessor.preprocess()
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 0: Drop unused columns
        if drop_unused:
            self._drop_unused_columns()
            
        # Step 1: Remove duplicates
        if remove_duplicates:
            self._remove_duplicates()
        
        # Step 2: Handle critical missing values (remove rows)
        self._remove_critical_missing()
        
        # Step 3: Handle remaining missing values (impute)
        if handle_missing:
            self._impute_missing_numerical()
            self._impute_missing_categorical()
        
        # Step 4: Handle outliers
        if handle_outliers:
            self._cap_outliers()
        
        # Step 5: Standardize categorical values
        if standardize_categories:
            self._standardize_categories()
        
        # Step 6: Ensure correct data types
        self._fix_data_types()
        
        # Update report
        self.report.final_rows = len(self.df)
        self.report.rows_removed = self.report.initial_rows - self.report.final_rows
        
        logger.info(f"Preprocessing complete. {len(self.df):,} rows remaining.")
        
        return self.df
    
    def _remove_duplicates(self) -> None:
        """
        Remove exact duplicate rows.
        
        Duplicates can arise from data export issues or system errors.
        Keeping them would give those observations more weight in training.
        """
        initial_count = len(self.df)
        
        self.df = self.df.drop_duplicates()
        
        removed = initial_count - len(self.df)
        self.report.duplicates_removed = removed
        
        if removed > 0:
            self.report.add_transformation(
                f"Removed {removed:,} duplicate rows"
            )
    
    def _remove_critical_missing(self) -> None:
        """
        Remove rows with missing values in critical columns.
        
        Some columns are too important to impute - if the target variable
        or order ID is missing, we must remove the row entirely.
        """
        initial_count = len(self.df)
        
        for col in self.CRITICAL_COLUMNS:
            if col in self.df.columns:
                null_count = self.df[col].isna().sum()
                if null_count > 0:
                    self.df = self.df.dropna(subset=[col])
                    self.report.add_transformation(
                        f"Removed {null_count:,} rows with missing '{col}'"
                    )
        
        removed = initial_count - len(self.df)
        if removed > 0:
            logger.info(f"Removed {removed:,} rows with critical missing values")
    
    def _impute_missing_numerical(self) -> None:
        """
        Impute missing values in numerical columns.
        
        Different strategies are used based on the column:
        - median: Robust to outliers (used for most numerical columns)
        - mean: When distribution is approximately normal
        - zero: When missing means absence (e.g., no discount)
        
        Why median over mean? Consider shipping days:
        Values: [2, 3, 3, 4, 100]  # 100 is an outlier
        Mean = 22.4 (skewed by outlier)
        Median = 3 (robust, represents typical value)
        """
        for col, strategy in self.NUMERICAL_IMPUTATION.items():
            if col not in self.df.columns:
                continue
            
            null_count = self.df[col].isna().sum()
            if null_count == 0:
                continue
            
            if strategy == "median":
                fill_value = self.df[col].median()
            elif strategy == "mean":
                fill_value = self.df[col].mean()
            elif strategy == "zero":
                fill_value = 0
            elif strategy == "mode":
                fill_value = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else 0
            else:
                fill_value = self.df[col].median()  # Default to median
            
            self.df[col] = self.df[col].fillna(fill_value)
            
            self.report.missing_values_handled[col] = null_count
            self.report.add_transformation(
                f"Imputed {null_count:,} missing values in '{col}' with {strategy} ({fill_value:.2f})"
            )
    
    def _impute_missing_categorical(self) -> None:
        """
        Impute missing values in categorical columns.
        
        Strategies:
        - mode: Fill with most common category
        - literal string: Fill with explicit value like 'Unknown'
        
        Using mode is appropriate when we believe missing values
        follow the same distribution as non-missing values.
        Using 'Unknown' is better when missingness is informative.
        """
        for col, strategy in self.CATEGORICAL_IMPUTATION.items():
            if col not in self.df.columns:
                continue
            
            null_count = self.df[col].isna().sum()
            if null_count == 0:
                continue
            
            if strategy == "mode":
                mode_values = self.df[col].mode()
                if len(mode_values) > 0:
                    fill_value = mode_values.iloc[0]
                else:
                    fill_value = "Unknown"
            else:
                # Use the strategy string as the fill value
                fill_value = strategy
            
            self.df[col] = self.df[col].fillna(fill_value)
            
            self.report.missing_values_handled[col] = null_count
            self.report.add_transformation(
                f"Imputed {null_count:,} missing values in '{col}' with '{fill_value}'"
            )
    
    def _cap_outliers(self) -> None:
        """
        Cap outliers using percentile-based winsorization.
        
        Winsorization replaces extreme values with percentile boundaries:
        - Values below the lower percentile are set to that percentile
        - Values above the upper percentile are set to that percentile
        
        Why cap instead of remove?
        1. Removing outliers reduces sample size
        2. Extreme values may be valid (very large orders exist)
        3. Capping preserves the "extreme" signal without distorting the model
        
        Example:
        Data: [1, 2, 3, 4, 5, 100]  # 100 is outlier
        99th percentile: ~5
        After capping: [1, 2, 3, 4, 5, 5]
        """
        for col, (lower_pct, upper_pct) in self.OUTLIER_COLUMNS.items():
            if col not in self.df.columns:
                continue
            
            # Calculate percentile bounds
            lower_bound = self.df[col].quantile(lower_pct)
            upper_bound = self.df[col].quantile(upper_pct)
            
            # Count outliers
            below_count = (self.df[col] < lower_bound).sum()
            above_count = (self.df[col] > upper_bound).sum()
            total_outliers = below_count + above_count
            
            if total_outliers == 0:
                continue
            
            # Cap values
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.report.outliers_handled[col] = total_outliers
            self.report.add_transformation(
                f"Capped {total_outliers:,} outliers in '{col}' "
                f"to [{lower_bound:.2f}, {upper_bound:.2f}]"
            )
    
    def _standardize_categories(self) -> None:
        """
        Standardize categorical values for consistency.
        
        Common issues in categorical data:
        1. Extra whitespace: "Standard Class " vs "Standard Class"
        2. Case inconsistency: "STANDARD CLASS" vs "Standard Class"
        3. Typos: "Standrad Class" vs "Standard Class"
        
        This method handles the first two issues. Typos would require
        fuzzy matching which is beyond our scope.
        """
        for col in CATEGORICAL_COLUMNS:
            if col not in self.df.columns:
                continue
            
            # Skip if not string type
            if self.df[col].dtype not in ['object', 'category', 'string']:
                continue
            
            # Convert to string, strip whitespace, and title case
            original_unique = self.df[col].nunique()
            
            # Handle category dtype
            if self.df[col].dtype.name == 'category':
                self.df[col] = self.df[col].astype(str)
            
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.title()
            )
            
            new_unique = self.df[col].nunique()
            
            if new_unique < original_unique:
                self.report.add_transformation(
                    f"Standardized '{col}': {original_unique} -> {new_unique} unique values"
                )
    
    def _fix_data_types(self) -> None:
        """
        Ensure all columns have correct data types.
        
        Proper data types are important for:
        1. Memory efficiency (int8 vs int64)
        2. Correct operations (can't do math on strings)
        3. Model compatibility (sklearn expects specific types)
        """
        # Ensure target is integer (0 or 1)
        if TARGET_COLUMN in self.df.columns:
            self.df[TARGET_COLUMN] = self.df[TARGET_COLUMN].astype(int)
        
        # Ensure numerical columns are float
        for col in NUMERICAL_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Ensure date columns are datetime
        for col in DATE_COLUMNS:
            if col in self.df.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        self.report.add_transformation(
                            f"Converted '{col}' to datetime"
                        )
                    except Exception as e:
                        logger.warning(f"Could not convert '{col}' to datetime: {e}")
    
    def get_missing_value_summary(self) -> pd.DataFrame:
        """
        Get a summary of missing values in the current DataFrame.
        
        Returns:
            pd.DataFrame: Summary with column names and missing counts.
        """
        missing_data = []
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            if null_count > 0:
                missing_data.append({
                    'column': col,
                    'missing_count': null_count,
                    'missing_percent': round(null_count / len(self.df) * 100, 2),
                })
        
        if not missing_data:
            return pd.DataFrame(columns=['column', 'missing_count', 'missing_percent'])
        
        return pd.DataFrame(missing_data).sort_values(
            'missing_count', ascending=False
        )
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get a summary of potential outliers using IQR method.
        
        The IQR (Interquartile Range) method defines outliers as:
        - Below: Q1 - 1.5 * IQR
        - Above: Q3 + 1.5 * IQR
        
        Returns:
            pd.DataFrame: Summary of outlier counts by column.
        """
        outlier_data = []
        
        for col in NUMERICAL_COLUMNS:
            if col not in self.df.columns:
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            below = (self.df[col] < lower_bound).sum()
            above = (self.df[col] > upper_bound).sum()
            
            if below > 0 or above > 0:
                outlier_data.append({
                    'column': col,
                    'below_lower': below,
                    'above_upper': above,
                    'total_outliers': below + above,
                    'outlier_percent': round((below + above) / len(self.df) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                })
        
        if not outlier_data:
            return pd.DataFrame()
        
        return pd.DataFrame(outlier_data).sort_values(
            'total_outliers', ascending=False
        )

    def _drop_unused_columns(self) -> None:
        """
        Drop columns that are not useful for analysis.
        
        Reasons for dropping:
        1. Too many missing values (>50%)
        2. Personally Identifiable Information (PII)
        3. Internal IDs with no predictive value
        4. Columns that could cause data leakage
        """
        columns_to_drop = [col for col in self.COLUMNS_TO_DROP if col in self.df.columns]
        
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            self.report.add_transformation(
                f"Dropped {len(columns_to_drop)} unused columns: {columns_to_drop[:5]}..."
            )

def preprocess_dataframe(
    df: pd.DataFrame,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Convenience function for preprocessing a DataFrame.
    
    Args:
        df: DataFrame to preprocess.
        **kwargs: Additional arguments passed to preprocess().
        
    Returns:
        Tuple of (preprocessed DataFrame, preprocessing report).
        
    Example:
        >>> clean_df, report = preprocess_dataframe(raw_df)
        >>> print(report.summary())
    """
    preprocessor = DataPreprocessor(df)
    clean_df = preprocessor.preprocess(**kwargs)
    return clean_df, preprocessor.report
