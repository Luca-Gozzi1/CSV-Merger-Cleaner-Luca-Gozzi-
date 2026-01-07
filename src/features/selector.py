"""
Feature selection module for Supply Chain Explorer.

This module provides utilities for selecting the most important features
for machine learning models. Feature selection helps:
1. Reduce overfitting by removing noisy features
2. Speed up training by reducing dimensionality
3. Improve interpretability by focusing on key predictors

Author: Luca Gozzi 
Date: November 2025
"""

import logging
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.ensemble import RandomForestClassifier

from src.config import TARGET_COLUMN, RANDOM_SEED


# Configure module logger
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Selects most important features for ML models.
    
    Provides multiple selection methods:
    1. Correlation-based: Remove highly correlated features
    2. Variance-based: Remove low-variance features
    3. Statistical tests: F-test, mutual information
    4. Model-based: Random Forest feature importance
    
    Attributes:
        X: Feature DataFrame.
        y: Target Series.
        
    Example:
        >>> selector = FeatureSelector(X, y)
        >>> importance_df = selector.get_feature_importance()
        >>> selected_features = selector.select_top_k(k=20)
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """
        Initialize the feature selector.
        
        Args:
            X: Feature DataFrame (no target column).
            y: Target Series.
        """
        self.X = X
        self.y = y
        
        # Store only numeric columns for analysis
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(
            f"FeatureSelector initialized with {len(X.columns)} features, "
            f"{len(self.numeric_cols)} numeric"
        )
    
    def remove_low_variance(
        self,
        threshold: float = 0.01,
    ) -> List[str]:
        """
        Identify features with very low variance.
        
        Low variance features are nearly constant and provide little
        predictive signal. For binary features, variance < 0.01 means
        less than 1% of values differ from the majority.
        
        Args:
            threshold: Minimum variance to keep a feature.
            
        Returns:
            List of feature names with variance below threshold.
        """
        low_variance = []
        
        for col in self.numeric_cols:
            variance = self.X[col].var()
            if variance < threshold:
                low_variance.append(col)
                logger.debug(f"Low variance feature: {col} (var={variance:.6f})")
        
        logger.info(f"Found {len(low_variance)} low-variance features")
        return low_variance
    
    def remove_high_correlation(
        self,
        threshold: float = 0.95,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated feature pairs.
        
        Highly correlated features are redundant - they provide similar
        information. Keeping both can lead to:
        - Multicollinearity in linear models
        - Wasted computation
        - Harder interpretation
        
        Args:
            threshold: Correlation above which to flag pairs.
            
        Returns:
            List of (feature1, feature2, correlation) tuples.
        """
        # Calculate correlation matrix for numeric columns only
        numeric_df = self.X[self.numeric_cols]
        corr_matrix = numeric_df.corr().abs()
        
        # Get upper triangle (avoid duplicate pairs)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs above threshold
        high_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                corr_value = upper_tri.loc[idx, col]
                if pd.notna(corr_value) and corr_value > threshold:
                    high_corr_pairs.append((idx, col, round(corr_value, 4)))
        
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs")
        return high_corr_pairs
    
    def get_feature_importance_rf(
        self,
        n_estimators: int = 50,
    ) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.
        
        Random Forest importance measures how much each feature
        decreases impurity (Gini) across all trees. Higher = more important.
        
        Advantages:
        - Captures non-linear relationships
        - Handles feature interactions
        - Works with mixed feature types
        
        Args:
            n_estimators: Number of trees in the forest.
            
        Returns:
            DataFrame with feature names and importance scores, sorted.
        """
        logger.info("Calculating Random Forest feature importance...")
        
        # Use only numeric columns
        X_numeric = self.X[self.numeric_cols].fillna(0)
        
        # Train a quick Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_numeric, self.y)
        
        # Extract importance
        importance_df = pd.DataFrame({
            "feature": self.numeric_cols,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["cumulative_importance"] = importance_df["importance"].cumsum()
        
        logger.info(f"Top 5 features: {importance_df.head()['feature'].tolist()}")
        
        return importance_df
    
    def get_feature_importance_ftest(self) -> pd.DataFrame:
        """
        Calculate feature importance using F-test (ANOVA).
        
        F-test measures the variance between class means relative to
        variance within classes. Higher F-score = more discriminative.
        
        Best for: Linear relationships, normally distributed features.
        Limitation: Misses non-linear relationships.
        
        Returns:
            DataFrame with feature names and F-scores, sorted.
        """
        logger.info("Calculating F-test feature importance...")
        
        X_numeric = self.X[self.numeric_cols].fillna(0)
        
        # Calculate F-scores
        f_scores, p_values = f_classif(X_numeric, self.y)
        
        importance_df = pd.DataFrame({
            "feature": self.numeric_cols,
            "f_score": f_scores,
            "p_value": p_values,
        }).sort_values("f_score", ascending=False)
        
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        # Mark statistically significant features
        importance_df["is_significant"] = importance_df["p_value"] < 0.05
        
        return importance_df
    
    def get_feature_importance_mutual_info(self) -> pd.DataFrame:
        """
        Calculate feature importance using mutual information.
        
        Mutual information measures the dependency between feature and target.
        Unlike correlation, it captures non-linear relationships.
        
        MI = 0: Feature and target are independent
        MI > 0: Feature provides information about target
        
        Returns:
            DataFrame with feature names and MI scores, sorted.
        """
        logger.info("Calculating mutual information feature importance...")
        
        X_numeric = self.X[self.numeric_cols].fillna(0)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(
            X_numeric, 
            self.y,
            random_state=RANDOM_SEED,
        )
        
        importance_df = pd.DataFrame({
            "feature": self.numeric_cols,
            "mutual_info": mi_scores,
        }).sort_values("mutual_info", ascending=False)
        
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def get_combined_importance(self) -> pd.DataFrame:
        """
        Combine multiple importance methods for robust selection.
        
        Different methods capture different aspects:
        - F-test: Linear relationships
        - Mutual info: Non-linear relationships
        - Random Forest: Complex interactions
        
        Combining them gives a more robust ranking.
        
        Returns:
            DataFrame with feature rankings from all methods.
        """
        logger.info("Calculating combined feature importance...")
        
        # Get importance from all methods
        rf_importance = self.get_feature_importance_rf()
        f_importance = self.get_feature_importance_ftest()
        mi_importance = self.get_feature_importance_mutual_info()
        
        # Merge on feature name
        combined = rf_importance[["feature", "importance"]].rename(
            columns={"importance": "rf_importance"}
        )
        combined = combined.merge(
            f_importance[["feature", "f_score"]],
            on="feature",
            how="outer"
        )
        combined = combined.merge(
            mi_importance[["feature", "mutual_info"]],
            on="feature",
            how="outer"
        )
        
        # Normalize each score to 0-1 range
        for col in ["rf_importance", "f_score", "mutual_info"]:
            if col in combined.columns:
                min_val = combined[col].min()
                max_val = combined[col].max()
                if max_val > min_val:
                    combined[f"{col}_norm"] = (
                        (combined[col] - min_val) / (max_val - min_val)
                    )
                else:
                    combined[f"{col}_norm"] = 0
        
        # Calculate average normalized score
        norm_cols = [col for col in combined.columns if col.endswith("_norm")]
        combined["avg_importance"] = combined[norm_cols].mean(axis=1)
        
        # Sort by average importance
        combined = combined.sort_values("avg_importance", ascending=False)
        combined["rank"] = range(1, len(combined) + 1)
        
        return combined
    
    def select_top_k(
        self,
        k: int = 20,
        method: str = "random_forest",
    ) -> List[str]:
        """
        Select top k most important features.
        
        Args:
            k: Number of features to select.
            method: Importance method ('random_forest', 'f_test', 
                   'mutual_info', 'combined').
        
        Returns:
            List of top k feature names.
        """
        if method == "random_forest":
            importance = self.get_feature_importance_rf()
        elif method == "f_test":
            importance = self.get_feature_importance_ftest()
            importance = importance.rename(columns={"f_score": "importance"})
        elif method == "mutual_info":
            importance = self.get_feature_importance_mutual_info()
            importance = importance.rename(columns={"mutual_info": "importance"})
        elif method == "combined":
            importance = self.get_combined_importance()
            importance = importance.rename(columns={"avg_importance": "importance"})
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top k features
        top_features = importance.head(k)["feature"].tolist()
        
        logger.info(f"Selected top {k} features using {method}")
        return top_features
    
    def get_correlation_with_target(self) -> pd.DataFrame:
        """
        Calculate correlation of each feature with target.
        
        Returns:
            DataFrame with feature correlations to target.
        """
        correlations = []
        
        for col in self.numeric_cols:
            corr = self.X[col].corr(self.y)
            correlations.append({
                "feature": col,
                "correlation": corr,
                "abs_correlation": abs(corr),
            })
        
        corr_df = pd.DataFrame(correlations).sort_values(
            "abs_correlation", ascending=False
        )
        
        return corr_df