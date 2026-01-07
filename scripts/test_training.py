"""Quick script to test model training with the actual DataCo dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.splitter import DataSplitter, get_X_y
from src.ml.trainer import ModelTrainer, train_models


def main():
    print("=" * 60)
    print("Testing Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load pre-split data
    print("\n1. Loading pre-split data...")
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    try:
        result = DataSplitter.load_splits(data_dir)
        print(f"   Train: {len(result.train):,} rows")
        print(f"   Validation: {len(result.validation):,} rows")
        print(f"   Test: {len(result.test):,} rows")
    except FileNotFoundError:
        print("   ERROR: Split data not found. Run test_splitting.py first.")
        return
    
    # Step 2: Extract X and y
    print("\n2. Extracting features and target...")
    X_train, y_train = get_X_y(result.train)
    X_val, y_val = get_X_y(result.validation)
    X_test, y_test = get_X_y(result.test)
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train distribution: {y_train.value_counts().to_dict()}")
    
    # Step 3: Initialize trainer
    print("\n3. Initializing ModelTrainer...")
    trainer = ModelTrainer(X_train, y_train, X_val, y_val)
    
    # Step 4: Train Logistic Regression
    print("\n4. Training Logistic Regression...")
    lr_result = trainer.train_logistic_regression()
    print(lr_result.summary())
    
    # Step 5: Train Random Forest
    print("\n5. Training Random Forest...")
    rf_result = trainer.train_random_forest()
    print(rf_result.summary())
    
    # Step 6: Train XGBoost
    print("\n6. Training XGBoost...")
    try:
        xgb_result = trainer.train_xgboost()
        print(xgb_result.summary())
    except ImportError as e:
        print(f"   Skipping XGBoost: {e}")
        xgb_result = None
    
    # Step 7: Save models
    print("\n7. Saving models...")
    models_dir = Path(__file__).parent.parent / "models"
    
    lr_path = ModelTrainer.save_model(lr_result, models_dir / "logistic_regression.pkl")
    print(f"   Saved: {lr_path}")
    
    rf_path = ModelTrainer.save_model(rf_result, models_dir / "random_forest.pkl")
    print(f"   Saved: {rf_path}")
    
    if xgb_result:
        xgb_path = ModelTrainer.save_model(xgb_result, models_dir / "xgboost.pkl")
        print(f"   Saved: {xgb_path}")
    
    # Step 8: Quick validation predictions
    print("\n8. Quick validation check...")
    
    # Logistic Regression predictions
    X_val_scaled = trainer._get_scaled_data(X_val)
    lr_preds = lr_result.model.predict(X_val_scaled)
    lr_accuracy = (lr_preds == y_val).mean()
    print(f"   Logistic Regression validation accuracy: {lr_accuracy:.4f}")
    
    # Random Forest predictions
    rf_preds = rf_result.model.predict(X_val.values)
    rf_accuracy = (rf_preds == y_val).mean()
    print(f"   Random Forest validation accuracy: {rf_accuracy:.4f}")
    
    # XGBoost predictions
    if xgb_result:
        xgb_preds = xgb_result.model.predict(X_val.values)
        xgb_accuracy = (xgb_preds == y_val).mean()
        print(f"   XGBoost validation accuracy: {xgb_accuracy:.4f}")
    
    # Step 9: Feature importance comparison
    print("\n9. Top 10 Features by Random Forest Importance:")
    importances = rf_result.model.feature_importances_
    feature_importance = list(zip(trainer.feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"   {i:2}. {feat}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("Model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()