"""
Data loading module for Supply Chain Explorer.

This module provides the DataLoader class responsible for reading CSV files
with proper encoding detection, date parsing, and memory optimization.
It handles the heterogeneous nature of supply chain data exports.

Author: Luca Gozzi 
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATASET_PATH,
    DATE_COLUMNS,
    REQUIRED_COLUMNS,
    RANDOM_SEED,
)


# Configure module logger
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial processing of supply chain CSV data.
    
    This class encapsulates all logic for reading CSV files, including
    encoding detection, date parsing, and basic dtype optimization.
    Using a class allows us to maintain state (like the loaded DataFrame)
    and provides a clean interface for the rest of the pipeline.
    
    Attributes:
        filepath: Path to the CSV file to load.
        df: The loaded DataFrame (None until load() is called).
        
    Example:
        >>> loader = DataLoader()
        >>> df = loader.load()
        >>> print(f"Loaded {len(df)} rows")
    """
    
    # Common encodings to try when loading CSV files
    # Listed in order of likelihood for supply chain data exports
    ENCODINGS_TO_TRY: List[str] = [
        "utf-8",
        "latin-1",
        "iso-8859-1",
        "cp1252",  # Windows encoding
    ]
    
    def __init__(self, filepath: Optional[Path] = None) -> None:
        """
        Initialize the DataLoader with a file path.
        
        Args:
            filepath: Path to the CSV file. If None, uses the default
                     path from config (RAW_DATASET_PATH).
        """
        self.filepath = filepath if filepath is not None else RAW_DATASET_PATH
        self.df: Optional[pd.DataFrame] = None
        
        logger.info(f"DataLoader initialized with path: {self.filepath}")
    
    def load(
        self,
        parse_dates: bool = True,
        optimize_memory: bool = True,
        subset_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.
        
        This method handles encoding detection automatically and can
        optionally parse date columns and optimize memory usage.
        
        Args:
            parse_dates: If True, parse columns listed in DATE_COLUMNS
                        as datetime objects.
            optimize_memory: If True, downcast numerical columns to
                           reduce memory footprint.
            subset_columns: If provided, only load these columns.
                          Useful for faster loading during development.
        
        Returns:
            pd.DataFrame: The loaded data.
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the file cannot be read with any encoding.
            
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load(parse_dates=True, optimize_memory=True)
        """
        # Verify file exists before attempting to load
        if not self.filepath.exists():
            error_msg = f"Data file not found: {self.filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading data from {self.filepath}")
        
        # Determine which columns to load
        usecols = subset_columns if subset_columns else None
        
        # Determine which columns to parse as dates
        date_cols = DATE_COLUMNS if parse_dates else None
        
        # Try different encodings until one works
        df = self._load_with_encoding_detection(
            usecols=usecols,
            parse_dates=date_cols,
        )
        
        # Optimize memory if requested
        if optimize_memory:
            df = self._optimize_dtypes(df)
        
        # Store reference and return
        self.df = df
        
        logger.info(
            f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns"
        )
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        return df
    
    def _load_with_encoding_detection(
        self,
        usecols: Optional[List[str]] = None,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Attempt to load CSV with multiple encodings.
        
        Supply chain data often comes from various systems (ERP, WMS, TMS)
        that may use different character encodings. This method tries
        common encodings until one succeeds.
        
        Args:
            usecols: Columns to load (None for all).
            parse_dates: Columns to parse as dates.
            
        Returns:
            pd.DataFrame: Successfully loaded data.
            
        Raises:
            ValueError: If no encoding works.
        """
        last_error: Optional[Exception] = None
        
        for encoding in self.ENCODINGS_TO_TRY:
            try:
                logger.debug(f"Trying encoding: {encoding}")
                
                df = pd.read_csv(
                    self.filepath,
                    encoding=encoding,
                    usecols=usecols,
                    parse_dates=parse_dates,
                    low_memory=False,  # Avoid mixed dtype warnings
                )
                
                logger.info(f"Successfully loaded with encoding: {encoding}")
                return df
                
            except UnicodeDecodeError as e:
                logger.debug(f"Encoding {encoding} failed: {e}")
                last_error = e
                continue
        
        # If we get here, no encoding worked
        error_msg = f"Failed to load {self.filepath} with any encoding"
        logger.error(error_msg)
        raise ValueError(error_msg) from last_error
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numerical types.
        
        This is important for large datasets like DataCo (180K rows).
        For example, an int64 column that only contains values 0-100
        can be stored as int8, using 8x less memory.
        
        Args:
            df: DataFrame to optimize.
            
        Returns:
            pd.DataFrame: Memory-optimized DataFrame.
        """
        logger.info("Optimizing DataFrame memory usage...")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1e6
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Downcast integers
            if col_type in ["int64", "int32"]:
                df[col] = pd.to_numeric(df[col], downcast="integer")
            
            # Downcast floats
            elif col_type in ["float64"]:
                df[col] = pd.to_numeric(df[col], downcast="float")
            
            # Convert low-cardinality strings to category
            elif col_type == "object":
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                # If fewer than 50% unique values, convert to category
                # This saves memory for columns like "Shipping Mode" (4 values)
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype("category")
        
        final_memory = df.memory_usage(deep=True).sum() / 1e6
        reduction = (1 - final_memory / initial_memory) * 100
        
        logger.info(
            f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )
        
        return df
    
    def load_sample(
        self,
        n_rows: int = 1000,
        random: bool = False,
    ) -> pd.DataFrame:
        """
        Load a sample of the data for quick exploration or testing.
        
        Args:
            n_rows: Number of rows to load.
            random: If True, sample randomly. If False, take first n rows.
            
        Returns:
            pd.DataFrame: Sample of the data.
            
        Example:
            >>> loader = DataLoader()
            >>> sample = loader.load_sample(n_rows=500, random=True)
        """
        logger.info(f"Loading sample of {n_rows} rows (random={random})")
        
        if random:
            # Load full dataset first, then sample
            # This ensures truly random sampling
            full_df = self.load(optimize_memory=False)
            sample = full_df.sample(n=n_rows, random_state=RANDOM_SEED)
        else:
            # Use nrows parameter for efficiency
            sample = pd.read_csv(
                self.filepath,
                nrows=n_rows,
                encoding="latin-1",  # DataCo uses this encoding
            )
        
        logger.info(f"Loaded sample with {len(sample)} rows")
        return sample
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get summary information about all columns in the dataset.
        
        Useful for initial data exploration and understanding the schema.
        
        Returns:
            pd.DataFrame: Summary with column name, dtype, non-null count,
                         null count, and sample values.
                         
        Raises:
            ValueError: If data hasn't been loaded yet.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        info_data = []
        
        for col in self.df.columns:
            col_data = self.df[col]
            info_data.append({
                "column": col,
                "dtype": str(col_data.dtype),
                "non_null_count": col_data.notna().sum(),
                "null_count": col_data.isna().sum(),
                "null_percent": round(col_data.isna().mean() * 100, 2),
                "unique_count": col_data.nunique(),
                "sample_values": str(col_data.dropna().head(3).tolist()),
            })
        
        return pd.DataFrame(info_data)


def load_data(
    filepath: Optional[Path] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convenience function for loading data without instantiating DataLoader.
    
    This provides a simple functional interface for quick data loading.
    
    Args:
        filepath: Path to CSV file (uses default if None).
        **kwargs: Additional arguments passed to DataLoader.load().
        
    Returns:
        pd.DataFrame: Loaded data.
        
    Example:
        >>> df = load_data()  # Uses default path
        >>> df = load_data(Path("data/raw/custom.csv"))
    """
    loader = DataLoader(filepath)
    return loader.load(**kwargs)