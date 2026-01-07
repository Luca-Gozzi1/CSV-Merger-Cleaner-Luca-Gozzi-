"""Quick script to test feature engineering with the actual DataCo dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector


def main():
    print("=" * 60)
    print("Testing Feature Engineering Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    df = loader.load(parse_dates=True, optimize_memory=True)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Step 2: Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor(df)
    clean_df = preprocessor.preprocess()
    print(f"   After preprocessing: {len(clean_df):,} rows, {len(clean_df.columns)} columns")
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    engineer = FeatureEngineer(clean_df)
    featured_df = engineer.engineer_all()
    print(f"   After feature engineering: {len(featured_df):,} rows, {len(featured_df.columns)} columns")
    
    # Step 4: Print feature engineering report
    print("\n4. Feature Engineering Report:")
    print(engineer.report.summary())
    
    # Step 5: Show sample of new features
    print("\n5. Sample of engineered features:")
    new_features = [
        "order_day_of_week", "order_month", "is_weekend", "is_holiday_season",
        "shipping_lead_time_variance", "shipping_mode_risk", "is_expedited",
        "market_risk_score", "is_international", "order_item_value",
    ]
    available_features = [f for f in new_features if f in featured_df.columns]
    print(featured_df[available_features].head(10).to_string())
    
    # Step 6: Show all numeric columns
    print("\n6. All numeric feature columns:")
    numeric_cols = featured_df.select_dtypes(include=['number']).columns.tolist()
    print(f"   Total numeric features: {len(numeric_cols)}")
    for i, col in enumerate(numeric_cols, 1):
        print(f"   {i:2}. {col}")
    
    # Step 7: Feature selection analysis
    print("\n7. Feature importance analysis...")
    X = featured_df.drop(columns=["Late_delivery_risk"])
    y = featured_df["Late_delivery_risk"]
    
    selector = FeatureSelector(X, y)
    
    # Get Random Forest importance
    print("\n   Top 15 features by Random Forest importance:")
    rf_importance = selector.get_feature_importance_rf(n_estimators=50)
    print(rf_importance.head(15).to_string())
    
    # Get correlation with target
    print("\n   Top 10 features by correlation with target:")
    corr_df = selector.get_correlation_with_target()
    print(corr_df.head(10).to_string())
    
    # Step 8: Check for any remaining issues
    print("\n8. Data quality check:")
    print(f"   Missing values: {featured_df.isna().sum().sum()}")
    print(f"   Target distribution:")
    print(featured_df["Late_delivery_risk"].value_counts())
    
    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()