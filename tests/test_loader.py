"""
Unit tests for the DataLoader class.

These tests verify that data loading works correctly for various scenarios
including successful loads, encoding issues, and edge cases.

Run with: pytest tests/test_loader.py -v

Author: Luca Gozzi 
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.loader import DataLoader, load_data


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""
    
    def test_init_with_default_path(self):
        """DataLoader should use default path from config when none provided."""
        loader = DataLoader()
        
        # Should have a filepath set (from config)
        assert loader.filepath is not None
        assert isinstance(loader.filepath, Path)
    
    def test_init_with_custom_path(self, tmp_path: Path):
        """DataLoader should accept custom file path."""
        custom_path = tmp_path / "custom_data.csv"
        loader = DataLoader(filepath=custom_path)
        
        assert loader.filepath == custom_path
    
    def test_init_df_is_none(self):
        """DataFrame should be None before load() is called."""
        loader = DataLoader()
        
        assert loader.df is None


class TestDataLoaderLoad:
    """Tests for DataLoader.load() method."""
    
    def test_load_valid_csv(self, sample_csv_path: Path):
        """Should successfully load a valid CSV file."""
        loader = DataLoader(filepath=sample_csv_path)
        df = loader.load(parse_dates=False, optimize_memory=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert loader.df is not None
    
    def test_load_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError for non-existent file."""
        fake_path = tmp_path / "nonexistent.csv"
        loader = DataLoader(filepath=fake_path)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_load_returns_correct_shape(self, sample_csv_path: Path):
        """Loaded DataFrame should have expected shape."""
        loader = DataLoader(filepath=sample_csv_path)
        df = loader.load(parse_dates=False, optimize_memory=False)
        
        # Sample data has 100 rows
        assert len(df) == 100
        # And multiple columns
        assert len(df.columns) > 10
    
    def test_load_with_memory_optimization(self, sample_csv_path: Path):
        """Memory optimization should reduce DataFrame memory usage."""
        loader = DataLoader(filepath=sample_csv_path)
        
        # Load without optimization
        df_unoptimized = pd.read_csv(sample_csv_path)
        mem_before = df_unoptimized.memory_usage(deep=True).sum()
        
        # Load with optimization
        df_optimized = loader.load(
            parse_dates=False,
            optimize_memory=True
        )
        mem_after = df_optimized.memory_usage(deep=True).sum()
        
        # Memory should be reduced (or at least not increased)
        assert mem_after <= mem_before
    
    def test_load_stores_df_reference(self, sample_csv_path: Path):
        """load() should store DataFrame reference in self.df."""
        loader = DataLoader(filepath=sample_csv_path)
        
        assert loader.df is None
        
        df = loader.load(parse_dates=False, optimize_memory=False)
        
        assert loader.df is not None
        assert loader.df is df


class TestDataLoaderSample:
    """Tests for DataLoader.load_sample() method."""
    
    def test_load_sample_respects_n_rows(self, sample_csv_path: Path):
        """load_sample should return requested number of rows."""
        loader = DataLoader(filepath=sample_csv_path)
        
        sample = loader.load_sample(n_rows=10, random=False)
        
        assert len(sample) == 10
    
    def test_load_sample_random_differs_from_head(self, sample_csv_path: Path):
        """Random sample should differ from first N rows."""
        loader = DataLoader(filepath=sample_csv_path)
        
        head_sample = loader.load_sample(n_rows=50, random=False)
        random_sample = loader.load_sample(n_rows=50, random=True)
        
        # Order IDs should differ (random vs sequential)
        # Note: There's a tiny chance they could be the same, but very unlikely
        head_ids = set(head_sample["Order Id"].tolist())
        random_ids = set(random_sample["Order Id"].tolist())
        
        # At least some IDs should differ
        assert head_ids != random_ids or len(head_ids) < 50


class TestDataLoaderColumnInfo:
    """Tests for DataLoader.get_column_info() method."""
    
    def test_get_column_info_without_load_raises(self):
        """get_column_info should raise if data not loaded."""
        loader = DataLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader.get_column_info()
        
        assert "not loaded" in str(exc_info.value).lower()
    
    def test_get_column_info_returns_dataframe(self, sample_csv_path: Path):
        """get_column_info should return a DataFrame with column metadata."""
        loader = DataLoader(filepath=sample_csv_path)
        loader.load(parse_dates=False, optimize_memory=False)
        
        info = loader.get_column_info()
        
        assert isinstance(info, pd.DataFrame)
        assert "column" in info.columns
        assert "dtype" in info.columns
        assert "null_count" in info.columns


class TestLoadDataFunction:
    """Tests for the load_data convenience function."""
    
    def test_load_data_with_path(self, sample_csv_path: Path):
        """load_data function should work with explicit path."""
        df = load_data(
            filepath=sample_csv_path,
            parse_dates=False,
            optimize_memory=False
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestEncodingDetection:
    """Tests for encoding detection functionality."""
    
    def test_encodings_list_not_empty(self):
        """DataLoader should have encodings to try."""
        assert len(DataLoader.ENCODINGS_TO_TRY) > 0
    
    def test_utf8_is_first_encoding(self):
        """UTF-8 should be the first encoding attempted."""
        assert DataLoader.ENCODINGS_TO_TRY[0] == "utf-8"