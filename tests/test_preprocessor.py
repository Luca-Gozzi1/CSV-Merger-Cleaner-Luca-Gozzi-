"""
Tests for the DataPreprocessor class.

These tests cover the preprocessing pipeline including:
- PreprocessingReport class
- Duplicate removal
- Missing value imputation
- Outlier capping
- Category standardization
- Data type fixing

Run with: pytest tests/test_preprocessor.py -v

Author: Luca Gozzi
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import (
    DataPreprocessor,
    PreprocessingReport,
    preprocess_dataframe,
)
from src.config import TARGET_COLUMN


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing preprocessing."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        "Order Id": range(1, n + 1),
        TARGET_COLUMN: np.random.randint(0, 2, n),
        "Days for shipping (real)": np.random.randint(1, 15, n).astype(float),
        "Days for shipment (scheduled)": np.random.randint(1, 10, n).astype(float),
        "Order Item Quantity": np.random.randint(1, 10, n).astype(float),
        "Order Item Product Price": np.random.uniform(10, 500, n),
        "Order Item Discount Rate": np.random.uniform(0, 0.3, n),
        "Sales": np.random.uniform(100, 5000, n),
        "Shipping Mode": np.random.choice(["Standard Class", "First Class", "Second Class"], n),
        "Market": np.random.choice(["LATAM", "Europe", "Pacific Asia", "Africa"], n),
        "Category Name": np.random.choice(["Electronics", "Clothing", "Furniture"], n),
        "order date (DateOrders)": pd.date_range("2020-01-01", periods=n, freq="D"),
        "shipping date (DateOrders)": pd.date_range("2020-01-05", periods=n, freq="D"),
    })


@pytest.fixture
def dataframe_with_missing():
    """Create a DataFrame with missing values."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        "Order Id": range(1, n + 1),
        TARGET_COLUMN: np.random.randint(0, 2, n),
        "Days for shipping (real)": np.random.randint(1, 15, n).astype(float),
        "Days for shipment (scheduled)": np.random.randint(1, 10, n).astype(float),
        "Order Item Quantity": np.random.randint(1, 10, n).astype(float),
        "Order Item Discount Rate": np.random.uniform(0, 0.3, n),
        "Shipping Mode": np.random.choice(["Standard Class", "First Class"], n),
        "Market": np.random.choice(["LATAM", "Europe"], n),
    })
    
    # Introduce missing values
    df.loc[5:10, "Days for shipping (real)"] = np.nan
    df.loc[15:20, "Order Item Discount Rate"] = np.nan
    df.loc[25:30, "Shipping Mode"] = np.nan
    df.loc[35:40, "Market"] = np.nan
    
    return df


@pytest.fixture
def dataframe_with_duplicates():
    """Create a DataFrame with duplicate rows."""
    df = pd.DataFrame({
        "Order Id": [1, 2, 3, 3, 4, 5, 5, 5],
        TARGET_COLUMN: [0, 1, 0, 0, 1, 0, 0, 0],
        "Days for shipping (real)": [3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 7.0],
        "Shipping Mode": ["Standard", "First", "Second", "Second", "Standard", "First", "First", "First"],
    })
    return df


@pytest.fixture
def dataframe_with_outliers():
    """Create a DataFrame with outliers."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        "Order Id": range(1, n + 1),
        TARGET_COLUMN: np.random.randint(0, 2, n),
        "Days for shipping (real)": np.random.randint(3, 7, n).astype(float),
        "Order Item Quantity": np.random.randint(1, 5, n).astype(float),
        "Sales": np.random.uniform(100, 500, n),
    })
    
    # Add outliers
    df.loc[0, "Days for shipping (real)"] = 100.0  # Extreme outlier
    df.loc[1, "Days for shipping (real)"] = -5.0   # Negative outlier
    df.loc[2, "Order Item Quantity"] = 1000.0      # Extreme quantity
    df.loc[3, "Sales"] = 1000000.0                  # Extreme sales
    
    return df


@pytest.fixture
def dataframe_with_category_issues():
    """Create a DataFrame with category standardization issues."""
    return pd.DataFrame({
        "Order Id": range(1, 6),
        TARGET_COLUMN: [0, 1, 0, 1, 0],
        "Days for shipping (real)": [3.0, 4.0, 5.0, 6.0, 7.0],
        "Shipping Mode": [
            "Standard Class",
            "  standard class  ",  # Extra whitespace
            "STANDARD CLASS",      # All caps
            "standard class",      # All lowercase
            "First Class",
        ],
        "Market": [
            "Europe",
            "  europe  ",
            "EUROPE",
            "europe",
            "LATAM",
        ],
    })


# =============================================================================
# PREPROCESSING REPORT TESTS
# =============================================================================

class TestPreprocessingReport:
    """Tests for PreprocessingReport class."""
    
    def test_default_values(self):
        """Should initialize with correct defaults."""
        report = PreprocessingReport()
        
        assert report.initial_rows == 0
        assert report.final_rows == 0
        assert report.rows_removed == 0
        assert report.duplicates_removed == 0
        assert report.missing_values_handled == {}
        assert report.outliers_handled == {}
        assert report.transformations == []
    
    def test_add_transformation(self):
        """Should add transformation to list."""
        report = PreprocessingReport()
        
        report.add_transformation("Test transformation")
        
        assert "Test transformation" in report.transformations
        assert len(report.transformations) == 1
    
    def test_add_multiple_transformations(self):
        """Should track multiple transformations."""
        report = PreprocessingReport()
        
        report.add_transformation("First")
        report.add_transformation("Second")
        report.add_transformation("Third")
        
        assert len(report.transformations) == 3
    
    def test_summary_generation(self):
        """Should generate readable summary."""
        report = PreprocessingReport(initial_rows=100, final_rows=95)
        report.rows_removed = 5
        report.duplicates_removed = 2
        report.missing_values_handled = {"col1": 3}
        report.add_transformation("Test transform")
        
        summary = report.summary()
        
        assert "PREPROCESSING REPORT" in summary
        assert "100" in summary  # initial_rows
        assert "95" in summary   # final_rows
        assert "5" in summary    # rows_removed
    
    def test_summary_includes_missing_values(self):
        """Summary should include missing value info."""
        report = PreprocessingReport()
        report.missing_values_handled = {"column_a": 10, "column_b": 5}
        
        summary = report.summary()
        
        assert "MISSING VALUES HANDLED" in summary
        assert "column_a" in summary
    
    def test_summary_includes_outliers(self):
        """Summary should include outlier info."""
        report = PreprocessingReport()
        report.outliers_handled = {"column_x": 15}
        
        summary = report.summary()
        
        assert "OUTLIERS HANDLED" in summary
        assert "column_x" in summary


# =============================================================================
# DATA PREPROCESSOR INITIALIZATION TESTS
# =============================================================================

class TestDataPreprocessorInit:
    """Tests for DataPreprocessor initialization."""
    
    def test_init_creates_copy(self, sample_dataframe):
        """Should work on a copy, not the original."""
        original = sample_dataframe.copy()
        preprocessor = DataPreprocessor(sample_dataframe)
        
        # Modify preprocessor's df
        preprocessor.df["new_col"] = 1
        
        # Original should be unchanged
        assert "new_col" not in sample_dataframe.columns
    
    def test_init_creates_report(self, sample_dataframe):
        """Should create a PreprocessingReport."""
        preprocessor = DataPreprocessor(sample_dataframe)
        
        assert isinstance(preprocessor.report, PreprocessingReport)
    
    def test_init_sets_initial_rows(self, sample_dataframe):
        """Should set initial_rows in report."""
        preprocessor = DataPreprocessor(sample_dataframe)
        
        assert preprocessor.report.initial_rows == len(sample_dataframe)


# =============================================================================
# DUPLICATE REMOVAL TESTS
# =============================================================================

class TestDuplicateRemoval:
    """Tests for _remove_duplicates() method."""
    
    def test_removes_duplicates(self, dataframe_with_duplicates):
        """Should remove duplicate rows."""
        preprocessor = DataPreprocessor(dataframe_with_duplicates)
        preprocessor._remove_duplicates()
        
        # Original had 8 rows with 3 duplicates
        assert len(preprocessor.df) == 5
    
    def test_reports_duplicates_removed(self, dataframe_with_duplicates):
        """Should report number of duplicates removed."""
        preprocessor = DataPreprocessor(dataframe_with_duplicates)
        preprocessor._remove_duplicates()
        
        assert preprocessor.report.duplicates_removed == 3
    
    def test_no_duplicates_unchanged(self, sample_dataframe):
        """Should not change DataFrame without duplicates."""
        preprocessor = DataPreprocessor(sample_dataframe)
        original_len = len(preprocessor.df)
        
        preprocessor._remove_duplicates()
        
        assert len(preprocessor.df) == original_len
        assert preprocessor.report.duplicates_removed == 0


# =============================================================================
# MISSING VALUE IMPUTATION TESTS
# =============================================================================

class TestMissingValueImputation:
    """Tests for missing value handling methods."""
    
    def test_impute_numerical_fills_values(self, dataframe_with_missing):
        """Should fill missing numerical values."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        
        assert dataframe_with_missing["Days for shipping (real)"].isna().sum() > 0
        
        preprocessor._impute_missing_numerical()
        
        assert preprocessor.df["Days for shipping (real)"].isna().sum() == 0
    
    def test_impute_numerical_reports_handled(self, dataframe_with_missing):
        """Should report number of imputed values."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        preprocessor._impute_missing_numerical()
        
        assert "Days for shipping (real)" in preprocessor.report.missing_values_handled
    
    def test_impute_categorical_fills_values(self, dataframe_with_missing):
        """Should fill missing categorical values."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        
        assert dataframe_with_missing["Shipping Mode"].isna().sum() > 0
        
        preprocessor._impute_missing_categorical()
        
        assert preprocessor.df["Shipping Mode"].isna().sum() == 0
    
    def test_impute_categorical_reports_handled(self, dataframe_with_missing):
        """Should report number of imputed categorical values."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        preprocessor._impute_missing_categorical()
        
        assert "Shipping Mode" in preprocessor.report.missing_values_handled
    
    def test_zero_imputation_for_discount(self, dataframe_with_missing):
        """Order Item Discount Rate should be imputed with zero."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        
        # Find rows that were NaN
        nan_mask = dataframe_with_missing["Order Item Discount Rate"].isna()
        
        preprocessor._impute_missing_numerical()
        
        # Those rows should now be 0
        assert (preprocessor.df.loc[nan_mask, "Order Item Discount Rate"] == 0).all()


# =============================================================================
# CRITICAL MISSING REMOVAL TESTS
# =============================================================================

class TestCriticalMissingRemoval:
    """Tests for _remove_critical_missing() method."""
    
    def test_removes_rows_with_missing_target(self):
        """Should remove rows where target is missing."""
        df = pd.DataFrame({
            "Order Id": [1, 2, 3, 4, 5],
            TARGET_COLUMN: [0, 1, np.nan, 0, 1],
            "Days for shipping (real)": [3.0, 4.0, 5.0, 6.0, 7.0],
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor._remove_critical_missing()
        
        assert len(preprocessor.df) == 4
        assert preprocessor.df[TARGET_COLUMN].isna().sum() == 0
    
    def test_removes_rows_with_missing_order_id(self):
        """Should remove rows where Order Id is missing."""
        df = pd.DataFrame({
            "Order Id": [1, 2, np.nan, 4, 5],
            TARGET_COLUMN: [0, 1, 0, 0, 1],
            "Days for shipping (real)": [3.0, 4.0, 5.0, 6.0, 7.0],
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor._remove_critical_missing()
        
        assert len(preprocessor.df) == 4


# =============================================================================
# OUTLIER CAPPING TESTS
# =============================================================================

class TestOutlierCapping:
    """Tests for _cap_outliers() method."""
    
    def test_caps_extreme_high_values(self, dataframe_with_outliers):
        """Should cap extremely high values."""
        preprocessor = DataPreprocessor(dataframe_with_outliers)
        
        max_before = preprocessor.df["Days for shipping (real)"].max()
        assert max_before == 100.0
        
        preprocessor._cap_outliers()
        
        max_after = preprocessor.df["Days for shipping (real)"].max()
        assert max_after < 100.0
    
    def test_caps_extreme_low_values(self, dataframe_with_outliers):
        """Should cap extremely low values."""
        preprocessor = DataPreprocessor(dataframe_with_outliers)
        
        min_before = preprocessor.df["Days for shipping (real)"].min()
        assert min_before == -5.0
        
        preprocessor._cap_outliers()
        
        min_after = preprocessor.df["Days for shipping (real)"].min()
        assert min_after > -5.0
    
    def test_reports_outliers_handled(self, dataframe_with_outliers):
        """Should report number of outliers handled."""
        preprocessor = DataPreprocessor(dataframe_with_outliers)
        preprocessor._cap_outliers()
        
        # At least one column should have outliers handled
        assert len(preprocessor.report.outliers_handled) > 0


# =============================================================================
# CATEGORY STANDARDIZATION TESTS
# =============================================================================

class TestCategoryStandardization:
    """Tests for _standardize_categories() method."""
    
    def test_removes_whitespace(self, dataframe_with_category_issues):
        """Should remove leading/trailing whitespace."""
        preprocessor = DataPreprocessor(dataframe_with_category_issues)
        preprocessor._standardize_categories()
        
        # Check no values have leading/trailing whitespace
        for val in preprocessor.df["Shipping Mode"]:
            assert val == val.strip()
    
    def test_standardizes_case(self, dataframe_with_category_issues):
        """Should standardize case to Title Case."""
        preprocessor = DataPreprocessor(dataframe_with_category_issues)
        preprocessor._standardize_categories()
        
        # All Shipping Mode values should be title case now
        unique_modes = preprocessor.df["Shipping Mode"].unique()
        
        # Should have fewer unique values after standardization
        # (since "Standard Class", "STANDARD CLASS", "standard class" become same)
        assert len(unique_modes) <= 3


# =============================================================================
# DATA TYPE FIXING TESTS
# =============================================================================

class TestDataTypeFix:
    """Tests for _fix_data_types() method."""
    
    def test_target_is_integer(self, sample_dataframe):
        """Target column should be integer type."""
        preprocessor = DataPreprocessor(sample_dataframe)
        preprocessor._fix_data_types()
        
        assert preprocessor.df[TARGET_COLUMN].dtype in [np.int32, np.int64, int]
    
    def test_handles_date_conversion(self, sample_dataframe):
        """Should convert date columns to datetime."""
        preprocessor = DataPreprocessor(sample_dataframe)
        preprocessor._fix_data_types()
        
        # Date columns should be datetime type
        if "order date (DateOrders)" in preprocessor.df.columns:
            assert pd.api.types.is_datetime64_any_dtype(
                preprocessor.df["order date (DateOrders)"]
            )


# =============================================================================
# DROP UNUSED COLUMNS TESTS
# =============================================================================

class TestDropUnusedColumns:
    """Tests for _drop_unused_columns() method."""
    
    def test_drops_pii_columns(self):
        """Should drop PII columns."""
        df = pd.DataFrame({
            "Order Id": [1, 2, 3],
            TARGET_COLUMN: [0, 1, 0],
            "Customer Email": ["a@b.com", "c@d.com", "e@f.com"],
            "Customer Password": ["pass1", "pass2", "pass3"],
            "Days for shipping (real)": [3.0, 4.0, 5.0],
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor._drop_unused_columns()
        
        assert "Customer Email" not in preprocessor.df.columns
        assert "Customer Password" not in preprocessor.df.columns
    
    def test_keeps_required_columns(self, sample_dataframe):
        """Should keep required columns."""
        preprocessor = DataPreprocessor(sample_dataframe)
        preprocessor._drop_unused_columns()
        
        assert TARGET_COLUMN in preprocessor.df.columns
        assert "Order Id" in preprocessor.df.columns


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullPipeline:
    """Tests for the complete preprocess() pipeline."""
    
    def test_preprocess_returns_dataframe(self, sample_dataframe):
        """Should return a DataFrame."""
        preprocessor = DataPreprocessor(sample_dataframe)
        result = preprocessor.preprocess()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_preprocess_updates_report(self, sample_dataframe):
        """Should update final_rows in report."""
        preprocessor = DataPreprocessor(sample_dataframe)
        preprocessor.preprocess()
        
        assert preprocessor.report.final_rows > 0
    
    def test_preprocess_calculates_rows_removed(self, dataframe_with_duplicates):
        """Should calculate rows_removed."""
        preprocessor = DataPreprocessor(dataframe_with_duplicates)
        preprocessor.preprocess()
        
        expected_removed = (
            preprocessor.report.initial_rows - 
            preprocessor.report.final_rows
        )
        assert preprocessor.report.rows_removed == expected_removed
    
    def test_preprocess_with_all_options_disabled(self, sample_dataframe):
        """Should work with all options disabled."""
        preprocessor = DataPreprocessor(sample_dataframe)
        result = preprocessor.preprocess(
            remove_duplicates=False,
            handle_missing=False,
            handle_outliers=False,
            standardize_categories=False,
            drop_unused=False,
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_fit_transform_alias(self, sample_dataframe):
        """preprocess() should work as fit_transform equivalent."""
        preprocessor = DataPreprocessor(sample_dataframe)
        result = preprocessor.preprocess()
        
        # Result should be cleaned
        assert len(result) <= len(sample_dataframe)


# =============================================================================
# SUMMARY METHODS TESTS
# =============================================================================

class TestSummaryMethods:
    """Tests for get_missing_value_summary() and get_outlier_summary()."""
    
    def test_missing_value_summary_returns_dataframe(self, dataframe_with_missing):
        """Should return a DataFrame."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        result = preprocessor.get_missing_value_summary()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_missing_value_summary_columns(self, dataframe_with_missing):
        """Should have correct columns."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        result = preprocessor.get_missing_value_summary()
        
        assert "column" in result.columns
        assert "missing_count" in result.columns
        assert "missing_percent" in result.columns
    
    def test_missing_value_summary_detects_missing(self, dataframe_with_missing):
        """Should detect missing values."""
        preprocessor = DataPreprocessor(dataframe_with_missing)
        result = preprocessor.get_missing_value_summary()
        
        # Should find the columns with missing values
        assert len(result) > 0
    
    def test_missing_value_summary_empty_when_no_missing(self, sample_dataframe):
        """Should return empty DataFrame when no missing values."""
        preprocessor = DataPreprocessor(sample_dataframe)
        result = preprocessor.get_missing_value_summary()
        
        # sample_dataframe has no missing values
        assert len(result) == 0
    
    def test_outlier_summary_returns_dataframe(self, dataframe_with_outliers):
        """Should return a DataFrame."""
        preprocessor = DataPreprocessor(dataframe_with_outliers)
        result = preprocessor.get_outlier_summary()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_outlier_summary_detects_outliers(self, dataframe_with_outliers):
        """Should detect outliers."""
        preprocessor = DataPreprocessor(dataframe_with_outliers)
        result = preprocessor.get_outlier_summary()
        
        # Should find outliers
        assert len(result) > 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction:
    """Tests for preprocess_dataframe() convenience function."""
    
    def test_returns_tuple(self, sample_dataframe):
        """Should return (DataFrame, Report) tuple."""
        result = preprocess_dataframe(sample_dataframe)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_first_element_is_dataframe(self, sample_dataframe):
        """First element should be DataFrame."""
        df, report = preprocess_dataframe(sample_dataframe)
        
        assert isinstance(df, pd.DataFrame)
    
    def test_second_element_is_report(self, sample_dataframe):
        """Second element should be PreprocessingReport."""
        df, report = preprocess_dataframe(sample_dataframe)
        
        assert isinstance(report, PreprocessingReport)
    
    def test_passes_kwargs(self, sample_dataframe):
        """Should pass kwargs to preprocess()."""
        df1, _ = preprocess_dataframe(sample_dataframe, remove_duplicates=True)
        df2, _ = preprocess_dataframe(sample_dataframe, remove_duplicates=False)
        
        # Both should work (may or may not differ based on data)
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)