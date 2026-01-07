"""
Script to test the DelayPredictor with real data.

Author: Luca Gozzi
Date: November 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.splitter import DataSplitter, get_X_y
from src.ml.predictor import DelayPredictor, load_predictor


def main():
    print("=" * 60)
    print("Testing DelayPredictor")
    print("=" * 60)
    
    # Step 1: Load test data
    print("\n1. Loading test data...")
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    try:
        splits = DataSplitter.load_splits(data_dir)
        X_test, y_test = get_X_y(splits.test)
        print(f"   Loaded {len(X_test):,} test samples")
    except FileNotFoundError:
        print("   ERROR: Data not found. Run test_splitting.py first.")
        return
    
    # Step 2: Load predictor
    print("\n2. Loading Random Forest predictor...")
    try:
        predictor = load_predictor("random_forest")
        print(f"   ✓ Loaded {predictor.model_name}")
    except FileNotFoundError:
        print("   ERROR: Model not found. Run test_training.py first.")
        return
    
    # Step 3: Make predictions
    print("\n3. Making predictions...")
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test)
    
    print(f"   Predictions: {len(predictions):,}")
    print(f"   Predicted Late: {predictions.sum():,} ({predictions.mean()*100:.1f}%)")
    print(f"   Predicted On-Time: {(1-predictions).sum():,.0f} ({(1-predictions.mean())*100:.1f}%)")
    
    # Step 4: Get risk scores
    print("\n4. Calculating risk scores...")
    risk_df = predictor.get_risk_scores(X_test)
    
    print(f"   Risk Score Distribution:")
    print(f"     Mean: {risk_df['risk_score'].mean():.1f}")
    print(f"     Median: {risk_df['risk_score'].median():.1f}")
    print(f"     Min: {risk_df['risk_score'].min():.1f}")
    print(f"     Max: {risk_df['risk_score'].max():.1f}")
    
    print(f"\n   Risk Category Distribution:")
    for category, count in risk_df["risk_category"].value_counts().items():
        pct = count / len(risk_df) * 100
        print(f"     {category}: {count:,} ({pct:.1f}%)")
    
    # Step 5: Get high-risk shipments
    print("\n5. High-risk shipments (probability >= 60%)...")
    high_risk = predictor.get_high_risk_shipments(X_test, threshold=0.6)
    print(f"   Found {len(high_risk):,} high-risk shipments")
    
    # Step 6: Risk summary
    print("\n6. Risk Summary Statistics:")
    summary = predictor.summarize_risk_distribution(X_test)
    
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Step 7: Compare with actual outcomes
    print("\n7. Comparison with Actual Outcomes:")
    print(f"   Actual Late: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    print(f"   Predicted Late: {predictions.sum():,} ({predictions.mean()*100:.1f}%)")
    
    # Accuracy check
    accuracy = (predictions == y_test).mean()
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("DelayPredictor test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()