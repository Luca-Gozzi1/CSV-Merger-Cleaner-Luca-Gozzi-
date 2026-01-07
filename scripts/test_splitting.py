"""Quick script to test data splitting with the actual DataCo dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.ml.splitter import DataSplitter, get_X_y


def main():
    print("=" * 60)
    print("Testing Train/Validation/Test Splitting")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    df = loader.load(parse_dates=True, optimize_memory=True)
    print(f"   Loaded {len(df):,} rows")
    
    # Step 2: Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor(df)
    clean_df = preprocessor.preprocess()
    print(f"   After preprocessing: {len(clean_df):,} rows")
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    engineer = FeatureEngineer(clean_df)
    featured_df = engineer.engineer_all()
    print(f"   After feature engineering: {len(featured_df):,} rows, {len(featured_df.columns)} columns")
    
    # Step 4: Time-based splitting
    print("\n4. Performing time-based split...")
    splitter = DataSplitter(featured_df)
    result = splitter.split(method="time")
    
    # Step 5: Print split summary
    print("\n5. Split Summary:")
    print(result.summary())
    
    # Step 6: Verify temporal ordering
    print("\n6. Temporal Order Verification:")
    date_col = "order date (DateOrders)"
    
    if date_col in result.train.columns:
        print(f"   Train:      {result.train[date_col].min()} to {result.train[date_col].max()}")
        print(f"   Validation: {result.validation[date_col].min()} to {result.validation[date_col].max()}")
        print(f"   Test:       {result.test[date_col].min()} to {result.test[date_col].max()}")
    else:
        print(f"   Date column '{date_col}' not in final features (already processed)")
    
    # Step 7: Get X, y for each split
    print("\n7. Extracting X and y for each split:")
    
    X_train, y_train = get_X_y(result.train)
    X_val, y_val = get_X_y(result.validation)
    X_test, y_test = get_X_y(result.test)
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_val shape:   {X_val.shape}")
    print(f"   y_val shape:   {y_val.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_test shape:  {y_test.shape}")
    
    # Step 8: Target distribution per split
    print("\n8. Target Distribution per Split:")
    print(f"   Train - Late: {y_train.mean()*100:.1f}%, On-time: {(1-y_train.mean())*100:.1f}%")
    print(f"   Val   - Late: {y_val.mean()*100:.1f}%, On-time: {(1-y_val.mean())*100:.1f}%")
    print(f"   Test  - Late: {y_test.mean()*100:.1f}%, On-time: {(1-y_test.mean())*100:.1f}%")
    
    # Step 9: Feature columns check
    print("\n9. Feature Columns (first 20):")
    feature_cols = X_train.columns.tolist()
    for i, col in enumerate(feature_cols[:20], 1):
        print(f"   {i:2}. {col}")
    if len(feature_cols) > 20:
        print(f"   ... and {len(feature_cols) - 20} more features")
    
    # Step 10: Save splits (optional)
    print("\n10. Saving splits to data/processed/...")
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    paths = splitter.save_splits(result, output_dir=output_dir)
    print(f"    Saved to: {output_dir}")
    for name, path in paths.items():
        print(f"    - {name}: {path.name}")
    
    print("\n" + "=" * 60)
    print("Data splitting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()