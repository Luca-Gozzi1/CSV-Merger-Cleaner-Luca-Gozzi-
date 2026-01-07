"""Quick script to test data preprocessing with the actual DataCo dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor


def main():
    print("=" * 60)
    print("Testing Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    df = loader.load(parse_dates=True, optimize_memory=True)
    print(f"   Loaded {len(df):,} rows")
    
    # Step 2: Check missing values before preprocessing
    print("\n2. Missing values BEFORE preprocessing:")
    preprocessor = DataPreprocessor(df)
    missing_summary = preprocessor.get_missing_value_summary()
    if len(missing_summary) > 0:
        print(missing_summary.to_string())
    else:
        print("   No missing values found!")
    
    # Step 3: Check outliers before preprocessing
    print("\n3. Outliers BEFORE preprocessing:")
    outlier_summary = preprocessor.get_outlier_summary()
    if len(outlier_summary) > 0:
        print(outlier_summary.to_string())
    else:
        print("   No outliers detected!")
    
    # Step 4: Run preprocessing
    print("\n4. Running preprocessing pipeline...")
    clean_df = preprocessor.preprocess()
    
    # Step 5: Print report
    print("\n5. Preprocessing Report:")
    print(preprocessor.report.summary())
    
    # Step 6: Verify no missing values remain
    print("\n6. Missing values AFTER preprocessing:")
    remaining_missing = clean_df.isna().sum().sum()
    print(f"   Total missing values: {remaining_missing}")
    
    # Step 7: Show target distribution
    print("\n7. Target distribution after preprocessing:")
    print(clean_df["Late_delivery_risk"].value_counts())
    print(clean_df["Late_delivery_risk"].value_counts(normalize=True))


if __name__ == "__main__":
    main()