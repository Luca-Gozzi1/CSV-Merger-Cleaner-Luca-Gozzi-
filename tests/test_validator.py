"""
Unit tests for the DataValidator class.

These tests verify that data validation correctly identifies issues
and produces accurate validation reports.

Run with: pytest tests/test_validator.py -v

Author: Luca Gozzi 
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np

from src.data.validator import (
    DataValidator,
    ValidationResult,
    validate_dataframe,
)


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""
    
    def test_default_is_valid(self):
        """New ValidationResult should be valid by default."""
        result = ValidationResult()
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error_marks_invalid(self):
        """Adding an error should mark result as invalid."""
        result = ValidationResult()
        
        result.add_error("Test error")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Test error" in result.errors
    
    def test_add_warning_keeps_valid(self):
        """Adding a warning should not affect validity."""
        result = ValidationResult()
        
        result.add_warning("Test warning")
        
        assert result.is_valid is True
        assert len(result.warnings) == 1
    
    def test_summary_includes_status(self):
        """Summary should include validation status."""
        result = ValidationResult()
        result.add_error("An error")
        
        summary = result.summary()
        
        assert "FAILED" in summary
        assert "An error" in summary
    
    def test_summary_passed_status(self):
        """Summary should show PASSED when valid."""
        result = ValidationResult()
        result.add_warning("Just a warning")
        
        summary = result.summary()
        
        assert "PASSED" in summary


class TestDataValidatorInit:
    """Tests for DataValidator initialization."""
    
    def test_init_stores_dataframe(self, sample_valid_df: pd.DataFrame):
        """Validator should store reference to DataFrame."""
        validator = DataValidator(sample_valid_df)
        
        assert validator.df is sample_valid_df
    
    def test_init_creates_result(self, sample_valid_df: pd.DataFrame):
        """Validator should create ValidationResult on init."""
        validator = DataValidator(sample_valid_df)
        
        assert isinstance(validator.result, ValidationResult)


class TestValidateAll:
    """Tests for the validate_all() method."""
    
    def test_valid_data_passes(self, sample_valid_df: pd.DataFrame):
        """Valid data should pass validation."""
        validator = DataValidator(sample_valid_df)
        result = validator.validate_all()
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_empty_dataframe_fails(self, empty_df: pd.DataFrame):
        """Empty DataFrame should fail validation."""
        validator = DataValidator(empty_df)
        result = validator.validate_all()
        
        assert result.is_valid is False
        assert any("empty" in error.lower() for error in result.errors)
    
    def test_missing_columns_fails(self, sample_df_missing_columns: pd.DataFrame):
        """DataFrame missing required columns should fail."""
        validator = DataValidator(sample_df_missing_columns)
        result = validator.validate_all()
        
        assert result.is_valid is False
        assert any("missing" in error.lower() for error in result.errors)
    
    def test_invalid_target_values_fails(self, sample_df_invalid_target: pd.DataFrame):
        """Invalid target values should fail validation."""
        validator = DataValidator(sample_df_invalid_target)
        result = validator.validate_all()
        
        assert result.is_valid is False
        assert any("unexpected values" in error.lower() for error in result.errors)
    
    def test_statistics_computed(self, sample_valid_df: pd.DataFrame):
        """validate_all should compute statistics."""
        validator = DataValidator(sample_valid_df)
        result = validator.validate_all()
        
        assert "total_rows" in result.stats
        assert "total_columns" in result.stats
        assert result.stats["total_rows"] == len(sample_valid_df)


class TestTargetValidation:
    """Tests specifically for target column validation."""
    
    def test_missing_target_values_error(self):
        """Target column with missing values should produce error."""
        df = pd.DataFrame({
            "Late_delivery_risk": [0, 1, None, 0, 1],
            "Order Id": ["1", "2", "3", "4", "5"],
        })
        
        validator = DataValidator(df)
        result = validator.validate_all()
        
        assert any("missing values" in error.lower() for error in result.errors)
    
    def test_class_imbalance_warning(self):
        """Severe class imbalance should produce warning."""
        # Create severely imbalanced data (95% class 0, 5% class 1)
        df = pd.DataFrame({
            "Late_delivery_risk": [0] * 95 + [1] * 5,
            "Order Id": [f"ORD-{i}" for i in range(100)],
        })
        
        validator = DataValidator(df)
        result = validator.validate_all()
        
        assert any("imbalance" in warning.lower() for warning in result.warnings)


class TestNumericalValidation:
    """Tests for numerical range validation."""
    
    def test_negative_shipping_days_warning(self, sample_valid_df: pd.DataFrame):
        """Negative shipping days should produce warning."""
        df = sample_valid_df.copy()
        df.loc[0, "Days for shipping (real)"] = -5
        
        validator = DataValidator(df)
        result = validator.validate_all()
        
        assert any("negative" in warning.lower() for warning in result.warnings)
    
    def test_excessive_discount_warning(self, sample_valid_df: pd.DataFrame):
        """Discount rate > 1 should produce warning."""
        df = sample_valid_df.copy()
        df.loc[0, "Order Item Discount Rate"] = 1.5  # 150% discount
        
        validator = DataValidator(df)
        result = validator.validate_all()
        
        assert any("above" in warning.lower() for warning in result.warnings)


class TestCategoricalValidation:
    """Tests for categorical value validation."""
    
    def test_unexpected_shipping_mode_warning(self, sample_valid_df: pd.DataFrame):
        """Unexpected shipping mode should produce warning."""
        df = sample_valid_df.copy()
        df.loc[0, "Shipping Mode"] = "Super Fast Shipping"  # Not in expected list
        
        validator = DataValidator(df)
        result = validator.validate_all()
        
        assert any("unexpected values" in warning.lower() for warning in result.warnings)


class TestLogicalConsistency:
    """Tests for logical consistency validation."""
    
    def test_shipping_before_order_warning(self, sample_valid_df: pd.DataFrame):
        """Shipping date before order date should produce warning."""
        df = sample_valid_df.copy()
        
        # Ensure datetime types
        df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
        df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"])
        
        # Set shipping date before order date
        df.loc[0, "shipping date (DateOrders)"] = df.loc[0, "order date (DateOrders)"] - pd.Timedelta(days=5)
        
        validator = DataValidator(df)
        result = validator.validate_all()
        
        assert any("before order date" in warning.lower() for warning in result.warnings)


class TestValidateDataframeFunction:
    """Tests for the validate_dataframe convenience function."""
    
    def test_convenience_function_works(self, sample_valid_df: pd.DataFrame):
        """validate_dataframe function should work correctly."""
        result = validate_dataframe(sample_valid_df)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True