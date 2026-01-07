"""
Data splitting module for Supply Chain Explorer.

This module provides the DataSplitter class responsible for creating
train/validation/test splits using time-based splitting. Time-based
splitting is essential for temporal prediction problems to avoid
data leakage from future observations.

Key Principle: Train on the past, validate on the near future,
test on the far future. This simulates real-world deployment.

Author: Luca Gozzi 
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
)


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """
    Container for data split results.
    
    Holds the train, validation, and test DataFrames along with
    metadata about the split for documentation and verification.
    
    Attributes:
        train: Training DataFrame.
        validation: Validation DataFrame.
        test: Test DataFrame.
        split_info: Dictionary with split metadata.
    """
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def train_size(self) -> int:
        """Number of rows in training set."""
        return len(self.train)
    
    @property
    def validation_size(self) -> int:
        """Number of rows in validation set."""
        return len(self.validation)
    
    @property
    def test_size(self) -> int:
        """Number of rows in test set."""
        return len(self.test)
    
    @property
    def total_size(self) -> int:
        """Total number of rows across all sets."""
        return self.train_size + self.validation_size + self.test_size
    
    def summary(self) -> str:
        """Generate a human-readable summary of the split."""
        total = self.total_size
        
        lines = [
            "=" * 60,
            "DATA SPLIT SUMMARY",
            "=" * 60,
            f"Total samples: {total:,}",
            "-" * 60,
            f"Training set:   {self.train_size:,} ({self.train_size/total*100:.1f}%)",
            f"Validation set: {self.validation_size:,} ({self.validation_size/total*100:.1f}%)",
            f"Test set:       {self.test_size:,} ({self.test_size/total*100:.1f}%)",
            "-" * 60,
        ]
        
        # Add split info
        if self.split_info:
            lines.append("SPLIT DETAILS:")
            for key, value in self.split_info.items():
                lines.append(f"  {key}: {value}")
        
        # Add target distribution per set
        lines.append("-" * 60)
        lines.append("TARGET DISTRIBUTION (Late_delivery_risk=1):")
        
        for name, df in [("Train", self.train), ("Validation", self.validation), ("Test", self.test)]:
            if TARGET_COLUMN in df.columns:
                late_pct = df[TARGET_COLUMN].mean() * 100
                lines.append(f"  {name}: {late_pct:.1f}% late deliveries")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class DataSplitter:
    """
    Splits data into train/validation/test sets using time-based splitting.
    
    Time-based splitting ensures that:
    1. Training data comes from the earliest time period
    2. Validation data comes from the middle time period
    3. Test data comes from the most recent time period
    
    This prevents data leakage where the model would learn from
    future events to predict past events.
    
    Attributes:
        df: DataFrame to split.
        date_column: Column containing order dates for temporal splitting.
        train_ratio: Proportion of data for training (default 0.70).
        validation_ratio: Proportion for validation (default 0.15).
        test_ratio: Proportion for testing (default 0.15).
        
    Example:
        >>> splitter = DataSplitter(featured_df)
        >>> result = splitter.split()
        >>> print(result.summary())
        >>> X_train, y_train = result.train.drop(columns=['Late_delivery_risk']), result.train['Late_delivery_risk']
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str = "order date (DateOrders)",
        train_ratio: float = TRAIN_RATIO,
        validation_ratio: float = VALIDATION_RATIO,
        test_ratio: float = TEST_RATIO,
    ) -> None:
        """
        Initialize the DataSplitter.
        
        Args:
            df: DataFrame to split.
            date_column: Column name containing dates for temporal ordering.
            train_ratio: Proportion of data for training.
            validation_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            
        Raises:
            ValueError: If ratios don't sum to 1.0 (approximately).
        """
        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=0.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.3f}"
            )
        
        self.df = df.copy()
        self.date_column = date_column
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        
        logger.info(
            f"DataSplitter initialized with {len(df):,} rows. "
            f"Ratios: {train_ratio}/{validation_ratio}/{test_ratio}"
        )
    
    def split(
        self,
        method: str = "time",
        shuffle_within_splits: bool = False,
    ) -> SplitResult:
        """
        Split the data into train/validation/test sets.
        
        Args:
            method: Splitting method ('time' for time-based, 'random' for random).
            shuffle_within_splits: Whether to shuffle rows within each split
                                  (preserves split boundaries but randomizes order).
        
        Returns:
            SplitResult: Container with train, validation, and test DataFrames.
            
        Raises:
            ValueError: If date column not found for time-based split.
        """
        if method == "time":
            result = self._time_based_split()
        elif method == "random":
            result = self._random_split()
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        # Optionally shuffle within splits
        if shuffle_within_splits:
            result.train = result.train.sample(frac=1, random_state=RANDOM_SEED)
            result.validation = result.validation.sample(frac=1, random_state=RANDOM_SEED)
            result.test = result.test.sample(frac=1, random_state=RANDOM_SEED)
            result.split_info["shuffled_within_splits"] = True
        
        # Verify no data leakage
        self._verify_no_overlap(result)
        
        return result
    
    def _time_based_split(self) -> SplitResult:
        """
        Perform time-based splitting.
        
        Orders the data by date and takes:
        - First 70% for training
        - Next 15% for validation
        - Last 15% for testing
        
        This ensures we train on past data and evaluate on future data.
        """
        logger.info("Performing time-based split...")
        
        # Check if date column exists
        if self.date_column not in self.df.columns:
            logger.warning(
                f"Date column '{self.date_column}' not found. "
                "Falling back to index-based split (assumes data is ordered)."
            )
            # Use index-based split if no date column
            return self._index_based_split()
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_column]):
            self.df[self.date_column] = pd.to_datetime(
                self.df[self.date_column], 
                errors='coerce'
            )
        
        # Sort by date
        df_sorted = self.df.sort_values(self.date_column).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df_sorted)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.validation_ratio))
        
        # Split the data
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        # Get date ranges for each split
        split_info = {
            "method": "time-based",
            "date_column": self.date_column,
            "train_date_range": f"{train_df[self.date_column].min()} to {train_df[self.date_column].max()}",
            "validation_date_range": f"{val_df[self.date_column].min()} to {val_df[self.date_column].max()}",
            "test_date_range": f"{test_df[self.date_column].min()} to {test_df[self.date_column].max()}",
        }
        
        logger.info(f"Train: {len(train_df):,} rows, {split_info['train_date_range']}")
        logger.info(f"Validation: {len(val_df):,} rows, {split_info['validation_date_range']}")
        logger.info(f"Test: {len(test_df):,} rows, {split_info['test_date_range']}")
        
        return SplitResult(
            train=train_df,
            validation=val_df,
            test=test_df,
            split_info=split_info,
        )
    
    def _index_based_split(self) -> SplitResult:
        """
        Perform index-based splitting (assumes data is already ordered).
        
        Used as fallback when date column is not available.
        """
        logger.info("Performing index-based split...")
        
        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.validation_ratio))
        
        train_df = self.df.iloc[:train_end].copy()
        val_df = self.df.iloc[train_end:val_end].copy()
        test_df = self.df.iloc[val_end:].copy()
        
        split_info = {
            "method": "index-based",
            "note": "No date column found, split by row order",
        }
        
        return SplitResult(
            train=train_df,
            validation=val_df,
            test=test_df,
            split_info=split_info,
        )
    
    def _random_split(self) -> SplitResult:
        """
        Perform random splitting.
        
        Note: This should generally NOT be used for time-series problems
        as it causes data leakage. Provided for comparison purposes.
        """
        logger.warning(
            "Using random split - this may cause data leakage for temporal data!"
        )
        
        # Shuffle the data
        df_shuffled = self.df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        n = len(df_shuffled)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.validation_ratio))
        
        train_df = df_shuffled.iloc[:train_end].copy()
        val_df = df_shuffled.iloc[train_end:val_end].copy()
        test_df = df_shuffled.iloc[val_end:].copy()
        
        split_info = {
            "method": "random",
            "random_seed": RANDOM_SEED,
            "warning": "Random split may cause data leakage for temporal problems",
        }
        
        return SplitResult(
            train=train_df,
            validation=val_df,
            test=test_df,
            split_info=split_info,
        )
    
    def _verify_no_overlap(self, result: SplitResult) -> None:
        """
        Verify that there's no temporal overlap between splits.
        
        For time-based splitting, the maximum date in training should be
        less than the minimum date in validation, and so on.
        """
        if self.date_column not in self.df.columns:
            logger.debug("Cannot verify temporal overlap without date column")
            return
        
        # Check if date column exists in result DataFrames
        if self.date_column not in result.train.columns:
            logger.debug("Date column not in split results, skipping verification")
            return
        
        # Get date ranges
        train_max = result.train[self.date_column].max()
        val_min = result.validation[self.date_column].min()
        val_max = result.validation[self.date_column].max()
        test_min = result.test[self.date_column].min()
        
        # Check for overlap
        if train_max > val_min:
            logger.warning(
                f"Potential overlap: train max ({train_max}) > validation min ({val_min})"
            )
        
        if val_max > test_min:
            logger.warning(
                f"Potential overlap: validation max ({val_max}) > test min ({test_min})"
            )
        
        # Log verification
        logger.info("Temporal split verification complete")
    
    def save_splits(
        self,
        result: SplitResult,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Save train/validation/test splits to CSV files.
        
        Saving splits ensures reproducibility - you can reload the same
        splits later without re-running the splitting logic.
        
        Args:
            result: SplitResult containing the data splits.
            output_dir: Directory to save files (default: from config).
            
        Returns:
            Dictionary mapping split names to file paths.
        """
        output_dir = output_dir or PROCESSED_DATA_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {
            "train": output_dir / "train.csv",
            "validation": output_dir / "validation.csv",
            "test": output_dir / "test.csv",
        }
        
        # Save each split
        result.train.to_csv(paths["train"], index=False)
        logger.info(f"Saved training data to {paths['train']}")
        
        result.validation.to_csv(paths["validation"], index=False)
        logger.info(f"Saved validation data to {paths['validation']}")
        
        result.test.to_csv(paths["test"], index=False)
        logger.info(f"Saved test data to {paths['test']}")
        
        # Save split info as JSON for documentation
        split_info_path = output_dir / "split_info.txt"
        with open(split_info_path, "w") as f:
            f.write(result.summary())
        logger.info(f"Saved split info to {split_info_path}")
        
        return paths
    
    @staticmethod
    def load_splits(
        input_dir: Optional[Path] = None,
    ) -> SplitResult:
        """
        Load previously saved train/validation/test splits.
        
        Args:
            input_dir: Directory containing saved splits.
            
        Returns:
            SplitResult with loaded data.
            
        Raises:
            FileNotFoundError: If split files don't exist.
        """
        input_dir = input_dir or PROCESSED_DATA_DIR
        input_dir = Path(input_dir)
        
        train_path = input_dir / "train.csv"
        val_path = input_dir / "validation.csv"
        test_path = input_dir / "test.csv"
        
        # Check files exist
        for path in [train_path, val_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}")
        
        # Load data
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(
            f"Loaded splits: train={len(train_df):,}, "
            f"validation={len(val_df):,}, test={len(test_df):,}"
        )
        
        return SplitResult(
            train=train_df,
            validation=val_df,
            test=test_df,
            split_info={"method": "loaded from files"},
        )


def get_X_y(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_date_columns: bool = True,
    remove_leaky_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y).
    
    Args:
        df: DataFrame containing features and target.
        target_column: Name of the target column.
        drop_date_columns: Whether to drop date columns from features.
                          These are kept for splitting but not for training.
        remove_leaky_features: Whether to remove features that cause data leakage.
                              Set to True for realistic production evaluation.
                              Set to False to see maximum accuracy (with leakage).
        
    Returns:
        Tuple of (features DataFrame, target Series).
        
    Raises:
        ValueError: If target column not found.
    """
    # Import here to avoid circular imports
    from src.config import LEAKY_FEATURES, REMOVE_LEAKY_FEATURES
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Drop date columns if present (they were kept for splitting)
    if drop_date_columns:
        date_cols_to_drop = [
            "order date (DateOrders)",
            "shipping date (DateOrders)",
        ]
        cols_to_drop = [col for col in date_cols_to_drop if col in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            logger.info(f"Dropped date columns from features: {cols_to_drop}")
    
    # Remove leaky features to prevent data leakage
    # This is controlled by both the function parameter and the config setting
    if remove_leaky_features and REMOVE_LEAKY_FEATURES:
        leaky_cols_to_drop = [col for col in LEAKY_FEATURES if col in X.columns]
        if leaky_cols_to_drop:
            X = X.drop(columns=leaky_cols_to_drop)
            logger.info(f"Dropped leaky features to prevent data leakage: {leaky_cols_to_drop}")
            logger.info(f"Remaining features: {len(X.columns)}")
    
    # Also drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        logger.info(f"Dropped non-numeric columns: {non_numeric}")
    
    return X, y

def split_data(
    df: pd.DataFrame,
    method: str = "time",
    save: bool = False,
    **kwargs: Any,
) -> SplitResult:
    """
    Convenience function for splitting data.
    
    Args:
        df: DataFrame to split.
        method: Splitting method ('time' or 'random').
        save: Whether to save splits to CSV files.
        **kwargs: Additional arguments passed to DataSplitter.
        
    Returns:
        SplitResult with train, validation, and test DataFrames.
        
    Example:
        >>> result = split_data(featured_df, method='time', save=True)
        >>> print(result.summary())
        >>> X_train, y_train = get_X_y(result.train)
    """
    splitter = DataSplitter(df, **kwargs)
    result = splitter.split(method=method)
    
    if save:
        splitter.save_splits(result)
    
    return result