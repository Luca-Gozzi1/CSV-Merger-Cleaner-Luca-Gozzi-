"""
Feature engineering module for Supply Chain Explorer.

This module provides the FeatureEngineer class responsible for creating
new features from raw supply chain data. Feature engineering is the process
of using domain knowledge to create variables that make ML algorithms work
better.

Key principle: Features should capture domain knowledge about what causes
shipment delays in supply chains.

Author: Luca Gozzi 
Date: November 2025
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    DATE_COLUMNS,
    RANDOM_SEED,
)


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringReport:
    """
    Container for feature engineering results.
    
    Tracks all features created and transformations applied.
    Useful for documentation and debugging.
    """
    initial_features: int = 0
    final_features: int = 0
    features_created: List[str] = field(default_factory=list)
    features_dropped: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    
    def add_feature(self, name: str, description: str = "") -> None:
        """Record a new feature creation."""
        self.features_created.append(name)
        if description:
            self.transformations.append(f"Created '{name}': {description}")
        logger.debug(f"Created feature: {name}")
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "FEATURE ENGINEERING REPORT",
            "=" * 60,
            f"Initial features: {self.initial_features}",
            f"Final features: {self.final_features}",
            f"Features created: {len(self.features_created)}",
            f"Features dropped: {len(self.features_dropped)}",
            "-" * 60,
        ]
        
        if self.features_created:
            lines.append("NEW FEATURES:")
            for feat in self.features_created:
                lines.append(f"  + {feat}")
        
        if self.transformations:
            lines.append("-" * 60)
            lines.append("TRANSFORMATIONS:")
            for i, t in enumerate(self.transformations, 1):
                lines.append(f"  {i}. {t}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class FeatureEngineer:
    """
    Creates features for shipment delay prediction.
    
    This class transforms raw supply chain data into features suitable
    for machine learning. It encapsulates domain knowledge about what
    factors influence delivery delays.
    
    Feature Categories:
    1. Temporal: Time-based patterns (day of week, month, holidays)
    2. Shipping: Delivery time characteristics
    3. Geographic: Location-based risk factors
    4. Product: Order and product characteristics
    5. Encoded: Categorical variables converted to numeric
    
    Attributes:
        df: DataFrame to engineer features for.
        report: FeatureEngineeringReport tracking all changes.
        
    Example:
        >>> engineer = FeatureEngineer(preprocessed_df)
        >>> featured_df = engineer.engineer_all()
        >>> print(engineer.report.summary())
    """
    
    # Shipping mode risk scores based on domain research
    # Higher score = higher risk of delay
    SHIPPING_MODE_RISK: Dict[str, float] = {
        "Same Day": 0.2,       # Fastest, most reliable
        "First Class": 0.8,   # Highest delay risk (research finding)
        "Second Class": 0.5,  # Medium risk
        "Standard Class": 0.6, # Common but moderate risk
    }
    
    # Market region risk scores based on geographic complexity
    MARKET_RISK: Dict[str, float] = {
        "USCA": 0.3,          # Domestic US/Canada - lower risk
        "Europe": 0.5,        # Western Europe - medium
        "LATAM": 0.7,         # Latin America - higher complexity
        "Pacific Asia": 0.6,  # Asia Pacific - medium-high
        "Africa": 0.8,        # Africa - highest complexity
    }
    
    # Customer segment ordering behavior patterns
    SEGMENT_ORDER_FREQUENCY: Dict[str, float] = {
        "Consumer": 0.5,      # Individual buyers - varied
        "Corporate": 0.7,     # Business buyers - regular, larger
        "Home Office": 0.4,   # Small office - smaller, irregular
    }
    
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the feature engineer with a DataFrame.
        
        Args:
            df: Preprocessed DataFrame to create features from.
        """
        self.df = df.copy()
        self.report = FeatureEngineeringReport(
            initial_features=len(df.columns)
        )
        
        logger.info(
            "FeatureEngineer initialized with %d rows, %d columns",
            len(df),
            len(df.columns),
        )

    def engineer_all(
        self,
        create_temporal: bool = True,
        create_shipping: bool = True,
        create_geographic: bool = True,
        create_product: bool = True,
        encode_categorical: bool = True,
        drop_original_categoricals: bool = True,
        keep_date_columns: bool = True,
        skip_leaky_features: bool = False,  # NEW: Skip creating leaky features
    ) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.
        
        This is the main entry point that orchestrates all feature
        creation steps. Each step can be toggled on/off.
        
        Args:
            create_temporal: Create time-based features.
            create_shipping: Create shipping-related features.
            create_geographic: Create location-based features.
            create_product: Create product/order features.
            encode_categorical: Encode categorical variables.
            drop_original_categoricals: Drop original categorical columns
                                       after encoding.
            keep_date_columns: Keep date columns for time-based splitting.
                              Set to False only after splitting is done.
            skip_leaky_features: If True, skip creating features that use
                                post-delivery information (prevents data leakage).
        
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Store the flag for use in feature creation methods
        self._skip_leaky_features = skip_leaky_features
        
        if skip_leaky_features:
            logger.info("Skipping leaky features to prevent data leakage")
        
        # Step 1: Temporal features
        if create_temporal:
            self._create_temporal_features()
        
        # Step 2: Shipping features
        if create_shipping:
            self._create_shipping_features()
        
        # Step 3: Geographic features
        if create_geographic:
            self._create_geographic_features()
        
        # Step 4: Product/Order features
        if create_product:
            self._create_product_features()

        # Step 5: Interaction features (NEW)
        self._create_interaction_features()    
        
        # Step 5: Encode categorical variables
        if encode_categorical:
            self._encode_categoricals(drop_original=drop_original_categoricals)
        
        # Step 6: Drop columns not needed for ML (but keep dates if requested)
        self._drop_non_feature_columns(keep_date_columns=keep_date_columns)
        
        # Update report
        self.report.final_features = len(self.df.columns)
        
        logger.info(
            "Feature engineering complete. %d features total.",
            self.report.final_features,
        )
        
        return self.df

    def _create_temporal_features(self) -> None:
        """
        Create time-based features from date columns.
        
        Temporal patterns are important because:
        - Weekends may have slower processing
        - Holiday seasons have higher volume and delays
        - Month-end/quarter-end may have shipping surges
        
        Features created:
        - order_day_of_week: 0=Monday, 6=Sunday
        - order_month: 1-12
        - order_quarter: 1-4
        - is_weekend: Order placed on weekend
        - is_month_start: First 5 days of month
        - is_month_end: Last 5 days of month
        - is_holiday_season: November-December (high volume)
        """
        order_date_col = "order date (DateOrders)"
        
        if order_date_col not in self.df.columns:
            logger.warning(f"Column '{order_date_col}' not found, skipping temporal features")
            return
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df[order_date_col]):
            self.df[order_date_col] = pd.to_datetime(self.df[order_date_col], errors='coerce')
        
        date_col = self.df[order_date_col]
        
        # Day of week (0=Monday, 6=Sunday)
        self.df["order_day_of_week"] = date_col.dt.dayofweek
        self.report.add_feature(
            "order_day_of_week",
            "Day of week (0=Mon, 6=Sun)"
        )
        
        # Month (1-12)
        self.df["order_month"] = date_col.dt.month
        self.report.add_feature("order_month", "Month of year (1-12)")
        
        # Quarter (1-4)
        self.df["order_quarter"] = date_col.dt.quarter
        self.report.add_feature("order_quarter", "Quarter of year (1-4)")
        
        # Is weekend (Saturday=5, Sunday=6)
        self.df["is_weekend"] = (date_col.dt.dayofweek >= 5).astype(int)
        self.report.add_feature("is_weekend", "1 if order on weekend")
        
        # Is month start (first 5 days)
        self.df["is_month_start"] = (date_col.dt.day <= 5).astype(int)
        self.report.add_feature("is_month_start", "1 if order in first 5 days")
        
        # Is month end (last 5 days)
        self.df["is_month_end"] = (date_col.dt.day >= 26).astype(int)
        self.report.add_feature("is_month_end", "1 if order in last 5 days")
        
        # Is holiday season (November-December)
        self.df["is_holiday_season"] = date_col.dt.month.isin([11, 12]).astype(int)
        self.report.add_feature(
            "is_holiday_season",
            "1 if Nov-Dec (high shipping volume)"
        )
        
        # Day of year (1-365) - captures seasonal patterns
        self.df["order_day_of_year"] = date_col.dt.dayofyear
        self.report.add_feature("order_day_of_year", "Day of year (1-365)")
        
        self.report.transformations.append(
            f"Created 8 temporal features from '{order_date_col}'"
        )
        
    def _create_shipping_features(self) -> None:
        """
        Create shipping-related features.
        
        These features relate to shipping and delivery patterns.
        Some features use post-delivery information and are marked as "leaky".
        
        Non-leaky features (safe for production):
        - shipping_mode_risk: Risk score based on shipping method
        - is_expedited: 1 if Same Day or First Class shipping
        - scheduled_shipping_days: The planned delivery days
        
        Leaky features (only for analysis, not production):
        - shipping_lead_time_variance: real - scheduled days
        - shipping_time_ratio: real / scheduled days
        - order_processing_time: shipping date - order date
        """
        # Check if we should skip leaky features
        skip_leaky = getattr(self, '_skip_leaky_features', False)
        
        real_col = "Days for shipping (real)"
        scheduled_col = "Days for shipment (scheduled)"
        
        # === NON-LEAKY FEATURES (always create) ===
        
        # Scheduled shipping days (this IS known at order time)
        if scheduled_col in self.df.columns:
            self.df["scheduled_shipping_days"] = self.df[scheduled_col]
            self.report.add_feature(
                "scheduled_shipping_days",
                "Planned delivery days (known at order time)"
            )
        
        # Shipping mode risk score (known at order time)
        shipping_mode_col = "Shipping Mode"
        if shipping_mode_col in self.df.columns:
            # Normalize the shipping mode values for matching
            shipping_normalized = self.df[shipping_mode_col].astype(str).str.strip().str.title()
            
            self.df["shipping_mode_risk"] = shipping_normalized.map(
                self.SHIPPING_MODE_RISK
            ).fillna(0.5)  # Default to medium risk
            
            self.report.add_feature(
                "shipping_mode_risk",
                "Risk score by shipping method (0-1) - known at order time"
            )
            
            # Is expedited shipping (Same Day or First Class)
            self.df["is_expedited"] = shipping_normalized.isin(
                ["Same Day", "First Class"]
            ).astype(int)
            
            self.report.add_feature(
                "is_expedited",
                "1 if Same Day or First Class shipping - known at order time"
            )
        
        # === LEAKY FEATURES (only create if not skipping) ===
        
        if not skip_leaky:
            # Shipping lead time variance (real - scheduled)
            # WARNING: This is a LEAKY feature - uses post-delivery information
            if real_col in self.df.columns and scheduled_col in self.df.columns:
                self.df["shipping_lead_time_variance"] = (
                    self.df[real_col] - self.df[scheduled_col]
                )
                self.report.add_feature(
                    "shipping_lead_time_variance",
                    "Actual - Scheduled days (LEAKY: uses post-delivery info)"
                )
            
            # Shipping time ratio
            # WARNING: This is a LEAKY feature - uses post-delivery information
            if real_col in self.df.columns and scheduled_col in self.df.columns:
                scheduled_safe = self.df[scheduled_col].replace(0, 1)
                self.df["shipping_time_ratio"] = self.df[real_col] / scheduled_safe
                
                self.report.add_feature(
                    "shipping_time_ratio",
                    "Actual / Scheduled days (LEAKY: uses post-delivery info)"
                )
            
            # Order processing time
            # WARNING: This may be leaky as shipping date may not be known at order time
            order_date_col = "order date (DateOrders)"
            ship_date_col = "shipping date (DateOrders)"
            
            if order_date_col in self.df.columns and ship_date_col in self.df.columns:
                # Ensure datetime
                for col in [order_date_col, ship_date_col]:
                    if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                
                self.df["order_processing_time"] = (
                    self.df[ship_date_col] - self.df[order_date_col]
                ).dt.days
                
                self.df["order_processing_time"] = self.df["order_processing_time"].clip(lower=0)
                
                self.report.add_feature(
                    "order_processing_time",
                    "Days from order to shipping (LEAKY: shipping date may not be known)"
                )
            
            self.report.transformations.append(
                "Created shipping features INCLUDING leaky features (for analysis)"
            )
        else:
            self.report.transformations.append(
                "Created shipping features EXCLUDING leaky features (production-safe)"
            )

    def _create_geographic_features(self) -> None:
        """
        Create geographic/location-based features.
        
        Geography affects delays through:
        - Distance and shipping complexity
        - Customs and international borders
        - Regional infrastructure quality
        
        Features created:
        - market_risk_score: Risk by market region
        - is_international: Inferred from market (non-USCA)
        """
        # Market risk score
        market_col = "Market"
        if market_col in self.df.columns:
            # Normalize market values
            market_normalized = self.df[market_col].astype(str).str.strip().str.upper()
            
            self.df["market_risk_score"] = market_normalized.map(
                {k.upper(): v for k, v in self.MARKET_RISK.items()}
            ).fillna(0.5)
            
            self.report.add_feature(
                "market_risk_score",
                "Risk score by market region (0-1)"
            )
            
            # Is international (non-USCA)
            self.df["is_international"] = (
                ~market_normalized.isin(["USCA"])
            ).astype(int)
            
            self.report.add_feature(
                "is_international",
                "1 if shipping outside US/Canada"
            )
        
        # Region complexity (number of unique countries in region)
        # This could be pre-calculated from domain knowledge
        region_col = "Order Region"
        if region_col in self.df.columns:
            # Count orders per region as a proxy for infrastructure
            region_counts = self.df[region_col].value_counts()
            total_orders = len(self.df)
            
            # Higher proportion = better infrastructure (more common route)
            self.df["region_order_frequency"] = self.df[region_col].map(
                region_counts / total_orders
            )
            
            self.report.add_feature(
                "region_order_frequency",
                "Proportion of orders from this region"
            )
        
        self.report.transformations.append(
            "Created geographic features (market risk, international flag)"
        )
    
    def _create_product_features(self) -> None:
        """
        Create product and order-related features.
        
        Order characteristics affect delivery:
        - Larger orders may require special handling
        - High-value orders may get priority
        - Heavy discounts may indicate clearance items
        
        Features created:
        - order_total_value: Price * Quantity
        - discount_intensity: Categorized discount level
        - is_high_value_order: Above median order value
        - items_per_order: Order quantity indicator
        """
        # Order total value (price * quantity)
        price_col = "Order Item Product Price"
        quantity_col = "Order Item Quantity"
        
        if price_col in self.df.columns and quantity_col in self.df.columns:
            self.df["order_item_value"] = (
                self.df[price_col] * self.df[quantity_col]
            )
            self.report.add_feature(
                "order_item_value",
                "Product price × quantity"
            )
            
            # Is high value order (above median)
            median_value = self.df["order_item_value"].median()
            self.df["is_high_value_order"] = (
                self.df["order_item_value"] > median_value
            ).astype(int)
            
            self.report.add_feature(
                "is_high_value_order",
                f"1 if order value > median ({median_value:.2f})"
            )
        
        # Discount intensity categories
        discount_col = "Order Item Discount Rate"
        if discount_col in self.df.columns:
            # Categorize discounts: none, low, medium, high
            self.df["discount_category"] = pd.cut(
                self.df[discount_col],
                bins=[-0.01, 0.0, 0.1, 0.2, 1.0],
                labels=["none", "low", "medium", "high"]
            )
            
            # Also keep a binary indicator for any discount
            self.df["has_discount"] = (self.df[discount_col] > 0).astype(int)
            
            self.report.add_feature("discount_category", "Discount level category")
            self.report.add_feature("has_discount", "1 if any discount applied")
        
        # Customer segment features
        segment_col = "Customer Segment"
        if segment_col in self.df.columns:
            segment_normalized = self.df[segment_col].astype(str).str.strip().str.title()
            
            self.df["segment_order_frequency"] = segment_normalized.map(
                self.SEGMENT_ORDER_FREQUENCY
            ).fillna(0.5)
            
            self.report.add_feature(
                "segment_order_frequency",
                "Expected order frequency by segment"
            )
        
        # Quantity buckets
        if quantity_col in self.df.columns:
            self.df["is_bulk_order"] = (self.df[quantity_col] > 5).astype(int)
            self.report.add_feature(
                "is_bulk_order",
                "1 if quantity > 5 items"
            )
        
        self.report.transformations.append(
            "Created product/order features (value, discount, bulk)"
        )
    
    def _create_interaction_features(self) -> None:
        """
        Create interaction features that capture combined effects.
        Interaction features can capture patterns that individual
        features miss, often providing 1-3% accuracy boost.
        """
        logger.info("Creating interaction features...")
        
        # 1. Shipping Mode + Market Risk Interaction
        if 'shipping_mode_risk' in self.df.columns and 'market_risk_score' in self.df.columns:
            self.df['shipping_market_interaction'] = (
                self.df['shipping_mode_risk'] * self.df['market_risk_score']
            )
            self.report.add_feature(
                "shipping_market_interaction",
                "Shipping risk × Market risk"
            )
        
        # 2. High-risk shipping in international markets
        if 'is_expedited' in self.df.columns and 'is_international' in self.df.columns:
            self.df['expedited_international'] = (
                self.df['is_expedited'] * self.df['is_international']
            )
            self.report.add_feature(
                "expedited_international",
                "Expedited shipping to international destination"
            )
        
        # 3. Weekend + Holiday Season (high delay risk periods)
        if 'is_weekend' in self.df.columns and 'is_holiday_season' in self.df.columns:
            self.df['high_risk_period'] = (
                (self.df['is_weekend'] == 1) | (self.df['is_holiday_season'] == 1)
            ).astype(int)
            self.report.add_feature(
                "high_risk_period",
                "Weekend or holiday season order"
            )
        
        # 4. Scheduled days categories (binned)
        scheduled_col = 'Days for shipment (scheduled)'
        if scheduled_col in self.df.columns:
            self.df['scheduled_days_cat'] = pd.cut(
                self.df[scheduled_col],
                bins=[-1, 2, 4, 6, 100],
                labels=[0, 1, 2, 3]
            ).astype(float).fillna(1)
            self.report.add_feature(
                "scheduled_days_cat",
                "Scheduled days category (0=fast, 3=slow)"
            )
        
        # 5. Order value relative to quantity (price per unit)
        if 'Sales' in self.df.columns and 'Order Item Quantity' in self.df.columns:
            qty_safe = self.df['Order Item Quantity'].replace(0, 1)
            self.df['value_per_unit'] = self.df['Sales'] / qty_safe
            
            # Normalize to 0-1 range
            max_val = self.df['value_per_unit'].quantile(0.99)
            self.df['value_per_unit_norm'] = (
                self.df['value_per_unit'].clip(upper=max_val) / max_val
            )
            self.report.add_feature(
                "value_per_unit_norm",
                "Normalized value per unit"
            )
        
        # 6. Complex order indicator (high value + bulk + expedited)
        complex_conditions = []
        if 'is_high_value_order' in self.df.columns:
            complex_conditions.append(self.df['is_high_value_order'])
        if 'is_bulk_order' in self.df.columns:
            complex_conditions.append(self.df['is_bulk_order'])
        if 'is_expedited' in self.df.columns:
            complex_conditions.append(self.df['is_expedited'])
        
        if complex_conditions:
            self.df['order_complexity'] = sum(complex_conditions)
            self.report.add_feature(
                "order_complexity",
                "Sum of: high value + bulk + expedited (0-3)"
            )
        
        # 7. Market-Segment interaction
        if 'market_risk_score' in self.df.columns and 'segment_order_frequency' in self.df.columns:
            self.df['market_segment_risk'] = (
                self.df['market_risk_score'] * (1 - self.df['segment_order_frequency'])
            )
            self.report.add_feature(
                "market_segment_risk",
                "Market risk adjusted by segment frequency"
            )
        
        # 8. Day of week risk score (Monday/Friday higher risk)
        if 'order_day_of_week' in self.df.columns:
            day_risk = {
                0: 0.7,  # Monday - high risk (backlog)
                1: 0.4,  # Tuesday
                2: 0.3,  # Wednesday
                3: 0.4,  # Thursday
                4: 0.6,  # Friday - medium-high (weekend coming)
                5: 0.5,  # Saturday
                6: 0.5,  # Sunday
            }
            self.df['day_risk_score'] = self.df['order_day_of_week'].map(day_risk).fillna(0.5)
            self.report.add_feature(
                "day_risk_score",
                "Day of week risk (Mon/Fri higher)"
            )
        
        # 9. Combined temporal risk
        if 'day_risk_score' in self.df.columns and 'high_risk_period' in self.df.columns:
            self.df['temporal_risk'] = (
                self.df['day_risk_score'] * 0.5 + 
                self.df['high_risk_period'] * 0.5
            )
            self.report.add_feature(
                "temporal_risk",
                "Combined temporal risk score"
            )
        
        # 10. Overall risk score (combines multiple risk factors)
        risk_features = ['shipping_mode_risk', 'market_risk_score', 'temporal_risk']
        available_risks = [f for f in risk_features if f in self.df.columns]
        
        if available_risks:
            self.df['overall_risk_score'] = self.df[available_risks].mean(axis=1)
            self.report.add_feature(
                "overall_risk_score",
                f"Average of {len(available_risks)} risk scores"
            )
        
        self.report.transformations.append(
            f"Created interaction and combined risk features"
        )

    def _encode_categoricals(self, drop_original: bool = True) -> None:
        """
        Encode categorical variables for ML models.
        
        Two encoding strategies:
        1. One-Hot Encoding: For low-cardinality categories (< 10 unique)
           - Creates binary columns for each category
           - Best for: Shipping Mode, Customer Segment
        
        2. Label Encoding: For high-cardinality categories
           - Creates single numeric column
           - Best for: Order Region, Category Name (many unique values)
        
        Why encode?
        - ML models (except tree-based) require numeric input
        - One-hot prevents ordinal assumptions
        - Label encoding is memory efficient for high cardinality
        
        Args:
            drop_original: Whether to drop original categorical columns.
        """
        # Define encoding strategy per column
        # One-hot for low cardinality, label for high cardinality
        one_hot_columns = ["Shipping Mode", "Customer Segment"]
        label_encode_columns = ["Market", "Order Region", "Category Name"]
        
        # One-hot encoding
        for col in one_hot_columns:
            if col not in self.df.columns:
                continue
            
            # Get dummies with prefix
            prefix = col.lower().replace(" ", "_")
            dummies = pd.get_dummies(
                self.df[col].astype(str),
                prefix=prefix,
                dtype=int
            )
            
            # Add to dataframe
            self.df = pd.concat([self.df, dummies], axis=1)
            
            for dummy_col in dummies.columns:
                self.report.add_feature(
                    dummy_col,
                    f"One-hot encoded from '{col}'"
                )
            
            # Track for dropping
            if drop_original:
                self.report.features_dropped.append(col)
        
        # Label encoding for high-cardinality columns
        for col in label_encode_columns:
            if col not in self.df.columns:
                continue
            
            # Create label encoded column
            new_col = f"{col.lower().replace(' ', '_')}_encoded"
            
            # Use pandas factorize for label encoding
            self.df[new_col], _ = pd.factorize(self.df[col].astype(str))
            
            self.report.add_feature(
                new_col,
                f"Label encoded from '{col}' ({self.df[col].nunique()} categories)"
            )
            
            if drop_original:
                self.report.features_dropped.append(col)
        
        # Handle discount_category created earlier
        if "discount_category" in self.df.columns:
            dummies = pd.get_dummies(
                self.df["discount_category"].astype(str),
                prefix="discount",
                dtype=int
            )
            self.df = pd.concat([self.df, dummies], axis=1)
            
            for dummy_col in dummies.columns:
                self.report.add_feature(dummy_col, "One-hot from discount_category")
            
            if drop_original:
                self.report.features_dropped.append("discount_category")
        
        # Drop original categorical columns if requested
        if drop_original:
            cols_to_drop = [
                col for col in self.report.features_dropped 
                if col in self.df.columns
            ]
            self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
            
            self.report.transformations.append(
                f"Dropped {len(cols_to_drop)} original categorical columns after encoding"
            )
        
        self.report.transformations.append(
            f"Encoded categorical variables (one-hot: {len(one_hot_columns)}, "
            f"label: {len(label_encode_columns)})"
        )
    
    def _drop_non_feature_columns(self, keep_date_columns: bool = True) -> None:
        """
        Drop columns that should not be used as ML features.
        
        Columns to drop:
        - Identifiers (Order Id) - no predictive value
        - Date columns - already extracted temporal features (optional)
        - Leaky columns - contain future information
        - Redundant columns - superseded by engineered features
        
        Args:
            keep_date_columns: If True, keep date columns for time-based splitting.
                              The splitter will drop them later.
        """
        # Columns that are not useful as features
        columns_to_drop = [
            # Identifiers
            "Order Id",
            # Potentially leaky (known after delivery)
            "Delivery Status",  # This directly encodes the outcome
            "Order Status",     # May indicate delivery status
            # Redundant with encoded versions
            "Shipping Mode",
            "Customer Segment",
            "Market",
            "Order Region",
            "Category Name",
            # Location details (too granular, use region instead)
            "Order Country",
            "Order City",
            "Order State",
            "Customer City",
            "Customer State",
            "Customer Country",
            # Other non-predictive columns
            "Department Name",
            "Product Name",
            "Product Status",
            "Type",
        ]
        
        # IMPORTANT: Only drop date columns if not keeping them
        if not keep_date_columns:
            columns_to_drop.extend([
                "order date (DateOrders)",
                "shipping date (DateOrders)",
            ])
        
        # Only drop columns that exist
        cols_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
            
            self.report.transformations.append(
                f"Dropped {len(cols_to_drop)} non-feature columns"
            )
            
            for col in cols_to_drop:
                if col not in self.report.features_dropped:
                    self.report.features_dropped.append(col)
        
        # Log what was kept
        if keep_date_columns:
            date_cols_kept = [col for col in DATE_COLUMNS if col in self.df.columns]
            if date_cols_kept:
                logger.info(f"Kept date columns for splitting: {date_cols_kept}")
    
    def get_feature_names(self, include_target: bool = False) -> List[str]:
        """
        Get list of feature column names.
        
        Args:
            include_target: Whether to include target column.
            
        Returns:
            List of feature column names.
        """
        features = [col for col in self.df.columns if col != TARGET_COLUMN]
        
        if include_target and TARGET_COLUMN in self.df.columns:
            features.append(TARGET_COLUMN)
        
        return features
    
    def get_feature_importance_ready_df(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get X (features) and y (target) ready for model training.
        
        Returns:
            Tuple of (features DataFrame, target Series).
            
        Raises:
            ValueError: If target column not found.
        """
        if TARGET_COLUMN not in self.df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
        
        X = self.df.drop(columns=[TARGET_COLUMN])
        y = self.df[TARGET_COLUMN]
        
        return X, y


def engineer_features(
    df: pd.DataFrame,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
    """
    Convenience function for feature engineering.
    
    Args:
        df: Preprocessed DataFrame.
        **kwargs: Additional arguments passed to engineer_all().
        
    Returns:
        Tuple of (featured DataFrame, engineering report).
        
    Example:
        >>> featured_df, report = engineer_features(clean_df)
        >>> print(report.summary())
    """
    engineer = FeatureEngineer(df)
    featured_df = engineer.engineer_all(**kwargs)
    return featured_df, engineer.report