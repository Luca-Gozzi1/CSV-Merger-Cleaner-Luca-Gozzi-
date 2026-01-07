"""
Pytest configuration and shared fixtures for Supply Chain Explorer tests.

Fixtures defined here are automatically available to all test files.
Using fixtures promotes DRY (Don't Repeat Yourself) principle and
ensures consistent test data across the test suite.

Author: Luca Gozzi 
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# PATH FIXTURES
# =============================================================================

@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_csv_path(test_data_dir: Path, sample_valid_df: pd.DataFrame) -> Path:
    """
    Create a temporary CSV file with valid sample data.
    
    This fixture creates the file, yields its path for the test,
    and cleans up afterwards.
    """
    # Ensure test data directory exists
    test_data_dir.mkdir(exist_ok=True)
    
    filepath = test_data_dir / "sample_valid.csv"
    sample_valid_df.to_csv(filepath, index=False)
    
    yield filepath
    
    # Cleanup after test
    if filepath.exists():
        filepath.unlink()


# =============================================================================
# DATAFRAME FIXTURES
# =============================================================================

@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """
    Create a small, valid DataFrame mimicking DataCo structure.
    
    This fixture provides realistic test data with all required columns
    and valid values. Use this for tests that expect valid input.
    """
    np.random.seed(42)
    n_rows = 100
    
    # Generate dates
    base_date = pd.Timestamp("2017-01-01")
    order_dates = [base_date + pd.Timedelta(days=i) for i in range(n_rows)]
    ship_dates = [d + pd.Timedelta(days=np.random.randint(1, 5)) for d in order_dates]
    
    # Generate realistic data
    data = {
        "Order Id": [f"ORD-{i:05d}" for i in range(n_rows)],
        "order date (DateOrders)": order_dates,
        "shipping date (DateOrders)": ship_dates,
        "Days for shipping (real)": np.random.randint(1, 10, n_rows),
        "Days for shipment (scheduled)": np.random.randint(2, 7, n_rows),
        "Late_delivery_risk": np.random.choice([0, 1], n_rows, p=[0.45, 0.55]),
        "Shipping Mode": np.random.choice(
            ["Standard Class", "First Class", "Second Class", "Same Day"],
            n_rows,
            p=[0.6, 0.15, 0.2, 0.05]
        ),
        "Order Item Discount Rate": np.random.uniform(0, 0.3, n_rows),
        "Order Item Quantity": np.random.randint(1, 10, n_rows),
        "Order Item Product Price": np.random.uniform(10, 500, n_rows),
        "Sales": np.random.uniform(50, 1000, n_rows),
        "Category Name": np.random.choice(
            ["Electronics", "Furniture", "Office Supplies", "Technology"],
            n_rows
        ),
        "Product Price": np.random.uniform(10, 500, n_rows),
        "Market": np.random.choice(
            ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"],
            n_rows
        ),
        "Order Region": np.random.choice(
            ["Central America", "Western Europe", "Southeast Asia", "Central US"],
            n_rows
        ),
        "Order Country": np.random.choice(
            ["Mexico", "France", "Thailand", "United States"],
            n_rows
        ),
        "Order City": np.random.choice(
            ["Mexico City", "Paris", "Bangkok", "New York"],
            n_rows
        ),
        "Customer Segment": np.random.choice(
            ["Consumer", "Corporate", "Home Office"],
            n_rows,
            p=[0.5, 0.3, 0.2]
        ),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_missing_columns() -> pd.DataFrame:
    """
    Create a DataFrame missing some required columns.
    
    Use this to test validation error handling.
    """
    return pd.DataFrame({
        "Order Id": ["ORD-001", "ORD-002"],
        "Sales": [100.0, 200.0],
        # Missing most required columns
    })


@pytest.fixture
def sample_df_with_nulls() -> pd.DataFrame:
    """
    Create a DataFrame with missing values.
    
    Use this to test null handling.
    """
    return pd.DataFrame({
        "Order Id": ["ORD-001", "ORD-002", "ORD-003", None, "ORD-005"],
        "Late_delivery_risk": [0, 1, None, 1, 0],
        "Days for shipping (real)": [3, None, 5, 2, None],
        "Shipping Mode": ["Standard Class", None, "First Class", "Same Day", "Standard Class"],
    })


@pytest.fixture
def sample_df_invalid_target() -> pd.DataFrame:
    """
    Create a DataFrame with invalid target values.
    
    Use this to test target validation.
    """
    return pd.DataFrame({
        "Order Id": ["ORD-001", "ORD-002", "ORD-003"],
        "Late_delivery_risk": [0, 2, -1],  # Invalid: should only be 0 or 1
        "Days for shipping (real)": [3, 5, 2],
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def mock_config(tmp_path: Path) -> dict:
    """
    Create a mock configuration for testing.
    
    Uses pytest's tmp_path for temporary file operations.
    """
    return {
        "raw_data_path": tmp_path / "raw" / "test_data.csv",
        "processed_data_path": tmp_path / "processed",
        "models_path": tmp_path / "models",
    }