"""
Data validation module for Supply Chain Explorer.

This module provides the DataValidator class responsible for verifying
that loaded data conforms to expected schemas, checking data quality,
and identifying potential issues before they affect downstream processing.

Validation is critical in supply chain analytics because data often comes
from multiple heterogeneous sources with inconsistent formats.

Author: Luca Gozzi 
Date: November 2025
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any

import pandas as pd
import numpy as np

from src.config import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    DATE_COLUMNS,
)


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Container for validation results.
    
    Using a dataclass provides a clean, typed structure for reporting
    validation outcomes. This is better than returning a dictionary
    because it provides IDE autocompletion and type checking.
    
    Attributes:
        is_valid: True if all critical validations passed.
        errors: List of critical issues that must be fixed.
        warnings: List of non-critical issues to be aware of.
        stats: Dictionary of data statistics.
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"Validation error: {message}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")
    
    def summary(self) -> str:
        """Generate a human-readable summary of validation results."""
        lines = [
            "=" * 60,
            "VALIDATION SUMMARY",
            "=" * 60,
            f"Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            "-" * 60,
        ]
        
        if self.errors:
            lines.append("ERRORS:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
        
        if self.stats:
            lines.append("-" * 60)
            lines.append("STATISTICS:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DataValidator:
    """
    Validates supply chain data against expected schema and quality rules.
    
    This class performs multiple types of validation:
    1. Schema validation: Required columns exist with correct types
    2. Completeness validation: Missing value analysis
    3. Range validation: Values within expected bounds
    4. Consistency validation: Cross-column logical checks
    
    Attributes:
        df: DataFrame to validate.
        result: ValidationResult containing all findings.
        
    Example:
        >>> validator = DataValidator(df)
        >>> result = validator.validate_all()
        >>> print(result.summary())
        >>> if not result.is_valid:
        ...     raise ValueError("Data validation failed")
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize validator with a DataFrame.
        
        Args:
            df: The DataFrame to validate.
        """
        self.df = df
        self.result = ValidationResult()
        
        logger.info(f"DataValidator initialized with {len(df):,} rows")
    
    def validate_all(self) -> ValidationResult:
        """
        Run all validation checks.
        
        This is the main entry point that orchestrates all validation
        steps in the correct order.
        
        Returns:
            ValidationResult: Complete validation results.
        """
        logger.info("Starting comprehensive data validation...")
        
        # Reset result for fresh validation
        self.result = ValidationResult()
        
        # Run validations in order of importance
        self._validate_not_empty()
        self._validate_required_columns()
        self._validate_target_column()
        self._validate_no_duplicates()
        self._validate_missing_values()
        self._validate_numerical_ranges()
        self._validate_categorical_values()
        self._validate_date_columns()
        self._validate_logical_consistency()
        
        # Compute summary statistics
        self._compute_statistics()
        
        logger.info(
            f"Validation complete. Valid: {self.result.is_valid}, "
            f"Errors: {len(self.result.errors)}, "
            f"Warnings: {len(self.result.warnings)}"
        )
        
        return self.result
    
    def _validate_not_empty(self) -> None:
        """Check that DataFrame is not empty."""
        if len(self.df) == 0:
            self.result.add_error("DataFrame is empty (0 rows)")
        elif len(self.df) < 100:
            self.result.add_warning(
                f"DataFrame has only {len(self.df)} rows. "
                "This may not be enough for reliable ML training."
            )
    
    def _validate_required_columns(self) -> None:
        """Check that all required columns are present."""
        existing_columns = set(self.df.columns)
        required_columns = set(REQUIRED_COLUMNS)
        
        missing_columns = required_columns - existing_columns
        
        if missing_columns:
            self.result.add_error(
                f"Missing required columns: {sorted(missing_columns)}"
            )
        
        # Also note any extra columns (informational)
        extra_columns = existing_columns - required_columns
        if extra_columns:
            logger.debug(f"Extra columns present (not required): {len(extra_columns)}")
    
    def _validate_target_column(self) -> None:
        """Validate the target column for ML."""
        if TARGET_COLUMN not in self.df.columns:
            self.result.add_error(f"Target column '{TARGET_COLUMN}' not found")
            return
        
        target = self.df[TARGET_COLUMN]
        
        # Check for missing values in target
        null_count = target.isna().sum()
        if null_count > 0:
            self.result.add_error(
                f"Target column has {null_count:,} missing values "
                f"({null_count / len(target) * 100:.2f}%)"
            )
        
        # Check that target is binary (0 or 1)
        unique_values = set(target.dropna().unique())
        expected_values = {0, 1}
        
        if not unique_values.issubset(expected_values):
            unexpected = unique_values - expected_values
            self.result.add_error(
                f"Target column contains unexpected values: {unexpected}. "
                f"Expected only 0 and 1."
            )
        
        # Check class balance
        if len(unique_values) == 2:
            class_counts = target.value_counts()
            minority_ratio = class_counts.min() / class_counts.sum()
            
            if minority_ratio < 0.1:
                self.result.add_warning(
                    f"Severe class imbalance: minority class is only "
                    f"{minority_ratio * 100:.1f}% of data"
                )
            elif minority_ratio < 0.3:
                self.result.add_warning(
                    f"Moderate class imbalance: minority class is "
                    f"{minority_ratio * 100:.1f}% of data"
                )
    
    def _validate_no_duplicates(self) -> None:
        """Check for duplicate rows."""
        # Check for completely duplicate rows
        duplicate_count = self.df.duplicated().sum()
        
        if duplicate_count > 0:
            duplicate_pct = duplicate_count / len(self.df) * 100
            if duplicate_pct > 5:
                self.result.add_warning(
                    f"Found {duplicate_count:,} duplicate rows "
                    f"({duplicate_pct:.2f}%)"
                )
            else:
                logger.info(f"Found {duplicate_count} duplicate rows ({duplicate_pct:.2f}%)")
        
        # Check for duplicate Order IDs (should be unique)
        if "Order Id" in self.df.columns:
            order_duplicates = self.df["Order Id"].duplicated().sum()
            if order_duplicates > 0:
                # Note: In DataCo, an order can have multiple items, so duplicates are expected
                logger.debug(
                    f"Found {order_duplicates} duplicate Order IDs "
                    "(expected for multi-item orders)"
                )
    
    def _validate_missing_values(self) -> None:
        """Analyze missing values across all columns."""
        missing_summary = []
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            if null_count > 0:
                null_pct = null_count / len(self.df) * 100
                missing_summary.append({
                    "column": col,
                    "missing_count": null_count,
                    "missing_pct": null_pct,
                })
                
                # Critical columns with missing values
                if col in REQUIRED_COLUMNS and null_pct > 10:
                    self.result.add_warning(
                        f"Column '{col}' has {null_pct:.1f}% missing values"
                    )
        
        if missing_summary:
            self.result.stats["columns_with_missing"] = len(missing_summary)
            logger.info(f"Found {len(missing_summary)} columns with missing values")
    
    def _validate_numerical_ranges(self) -> None:
        """Validate that numerical columns have sensible values."""
        range_checks = {
            "Days for shipping (real)": (0, 365),  # 0 to 1 year
            "Days for shipment (scheduled)": (0, 365),
            "Order Item Discount Rate": (0, 1),  # 0% to 100%
            "Order Item Quantity": (1, 10000),  # At least 1 item
            "Order Item Product Price": (0, 1e9),  # Non-negative
            "Sales": (0, 1e9),  # Non-negative
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col not in self.df.columns:
                continue
            
            series = self.df[col].dropna()
            
            # Check minimum
            below_min = (series < min_val).sum()
            if below_min > 0:
                self.result.add_warning(
                    f"Column '{col}' has {below_min} values below {min_val}"
                )
            
            # Check maximum
            above_max = (series > max_val).sum()
            if above_max > 0:
                self.result.add_warning(
                    f"Column '{col}' has {above_max} values above {max_val}"
                )
            
            # Check for negative values in columns that should be positive
            if min_val >= 0:
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    self.result.add_warning(
                        f"Column '{col}' has {negative_count} negative values"
                    )
    
    def _validate_categorical_values(self) -> None:
        """Validate categorical columns have expected values."""
        # Define expected values for key categorical columns
        expected_values = {
            "Shipping Mode": {"Standard Class", "First Class", "Second Class", "Same Day"},
            "Customer Segment": {"Consumer", "Corporate", "Home Office"},
            "Market": {"LATAM", "Europe", "Pacific Asia", "USCA", "Africa"},
        }
        
        for col, expected in expected_values.items():
            if col not in self.df.columns:
                continue
            
            actual = set(self.df[col].dropna().unique())
            unexpected = actual - expected
            
            if unexpected:
                # This is a warning, not error, as data may have legitimate new values
                self.result.add_warning(
                    f"Column '{col}' has unexpected values: {unexpected}"
                )
            
            missing_expected = expected - actual
            if missing_expected:
                logger.debug(f"Column '{col}' missing expected values: {missing_expected}")
    
    def _validate_date_columns(self) -> None:
        """Validate date columns are properly formatted."""
        for col in DATE_COLUMNS:
            if col not in self.df.columns:
                continue
            
            # Check if already datetime type
            if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.result.add_warning(
                    f"Column '{col}' is not datetime type (is {self.df[col].dtype})"
                )
                continue
            
            # Check for dates outside reasonable range
            series = self.df[col].dropna()
            
            min_date = pd.Timestamp("2010-01-01")
            max_date = pd.Timestamp("2030-01-01")
            
            before_min = (series < min_date).sum()
            after_max = (series > max_date).sum()
            
            if before_min > 0:
                self.result.add_warning(
                    f"Column '{col}' has {before_min} dates before {min_date.date()}"
                )
            
            if after_max > 0:
                self.result.add_warning(
                    f"Column '{col}' has {after_max} dates after {max_date.date()}"
                )
    
    def _validate_logical_consistency(self) -> None:
        """Check for logical inconsistencies between columns."""
        # Check: shipping date should be >= order date
        order_date_col = "order date (DateOrders)"
        ship_date_col = "shipping date (DateOrders)"
        
        if order_date_col in self.df.columns and ship_date_col in self.df.columns:
            # Ensure both are datetime
            if (pd.api.types.is_datetime64_any_dtype(self.df[order_date_col]) and
                pd.api.types.is_datetime64_any_dtype(self.df[ship_date_col])):
                
                invalid_dates = (self.df[ship_date_col] < self.df[order_date_col]).sum()
                
                if invalid_dates > 0:
                    self.result.add_warning(
                        f"Found {invalid_dates} rows where shipping date "
                        f"is before order date"
                    )
        
        # Check: Late_delivery_risk should align with actual vs scheduled days
        if all(col in self.df.columns for col in 
               ["Late_delivery_risk", "Days for shipping (real)", "Days for shipment (scheduled)"]):
            
            # Late should mean real > scheduled
            calculated_late = (
                self.df["Days for shipping (real)"] > self.df["Days for shipment (scheduled)"]
            ).astype(int)
            
            mismatch = (calculated_late != self.df["Late_delivery_risk"]).sum()
            mismatch_pct = mismatch / len(self.df) * 100
            
            if mismatch_pct > 5:
                self.result.add_warning(
                    f"Late_delivery_risk mismatches calculated lateness in "
                    f"{mismatch:,} rows ({mismatch_pct:.1f}%)"
                )
    
    def _compute_statistics(self) -> None:
        """Compute summary statistics for the validation report."""
        self.result.stats["total_rows"] = len(self.df)
        self.result.stats["total_columns"] = len(self.df.columns)
        self.result.stats["memory_mb"] = round(
            self.df.memory_usage(deep=True).sum() / 1e6, 2
        )
        
        # Target distribution
        if TARGET_COLUMN in self.df.columns:
            target_dist = self.df[TARGET_COLUMN].value_counts(normalize=True)
            self.result.stats["target_distribution"] = target_dist.to_dict()
        
        # Date range
        order_date_col = "order date (DateOrders)"
        if order_date_col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[order_date_col]):
                self.result.stats["date_range_start"] = str(
                    self.df[order_date_col].min().date()
                )
                self.result.stats["date_range_end"] = str(
                    self.df[order_date_col].max().date()
                )


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Convenience function for validating a DataFrame.
    
    Args:
        df: DataFrame to validate.
        
    Returns:
        ValidationResult: Validation results.
        
    Example:
        >>> result = validate_dataframe(df)
        >>> if not result.is_valid:
        ...     print(result.summary())
    """
    validator = DataValidator(df)
    return validator.validate_all()