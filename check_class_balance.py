"""
Script to analyze class imbalance in the supply chain dataset.
Run from project root: python check_class_balance.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the raw data
print("=" * 60)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 60)

# Load data
raw_path = "data/raw/DataCoSupplyChainDataset.csv"
df = pd.read_csv(raw_path, encoding='latin-1')

target_col = "Late_delivery_risk"

print(f"\nDataset shape: {df.shape}")
print(f"Total samples: {len(df):,}")

# Class distribution
print("\n" + "-" * 40)
print("CLASS DISTRIBUTION")
print("-" * 40)

class_counts = df[target_col].value_counts()
class_percentages = df[target_col].value_counts(normalize=True) * 100

print(f"\nTarget column: '{target_col}'")
print(f"\nValue counts:")
for value, count in class_counts.items():
    pct = class_percentages[value]
    label = "Late" if value == 1 else "On-Time"
    print(f"  {value} ({label}): {count:,} samples ({pct:.2f}%)")

# Imbalance ratio
majority_class = class_counts.max()
minority_class = class_counts.min()
imbalance_ratio = majority_class / minority_class

print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

# Interpretation
print("\n" + "-" * 40)
print("INTERPRETATION")
print("-" * 40)

late_pct = class_percentages.get(1, 0)
ontime_pct = class_percentages.get(0, 0)

if imbalance_ratio < 1.5:
    print("✓ Dataset is relatively BALANCED (ratio < 1.5)")
    print("  Standard classification should work well.")
elif imbalance_ratio < 3:
    print("⚠ Dataset has MODERATE imbalance (ratio 1.5-3)")
    print("  Consider using class_weight='balanced' in models.")
else:
    print("⚠ Dataset has SIGNIFICANT imbalance (ratio > 3)")
    print("  Consider SMOTE, undersampling, or adjusted thresholds.")

print(f"\nLate deliveries: {late_pct:.1f}%")
print(f"On-time deliveries: {ontime_pct:.1f}%")

# Check train/val/test splits if they exist
print("\n" + "-" * 40)
print("SPLIT DISTRIBUTION (if available)")
print("-" * 40)

splits = ["train", "validation", "test"]
for split in splits:
    split_path = f"data/processed/{split}.csv"
    try:
        split_df = pd.read_csv(split_path)
        if target_col in split_df.columns:
            split_pct = split_df[target_col].value_counts(normalize=True) * 100
            late_pct = split_pct.get(1, 0)
            print(f"\n{split.upper()}:")
            print(f"  Total samples: {len(split_df):,}")
            print(f"  Late deliveries: {late_pct:.1f}%")
            print(f"  On-time deliveries: {100-late_pct:.1f}%")
    except FileNotFoundError:
        print(f"\n{split.upper()}: File not found")

# Visual representation
print("\n" + "-" * 40)
print("VISUAL REPRESENTATION")
print("-" * 40)

bar_length = 50
late_bars = int(late_pct / 100 * bar_length)
ontime_bars = bar_length - late_bars

print(f"\nLate (1):    [{'█' * late_bars}{'░' * ontime_bars}] {late_pct:.1f}%")
print(f"On-Time (0): [{'█' * ontime_bars}{'░' * late_bars}] {ontime_pct:.1f}%")

# Recommendations based on results
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if late_pct > 50:
    print(f"""
The MAJORITY class is 'Late' ({late_pct:.1f}%).

This explains why threshold optimization pushed to very low values:
- At threshold=0.5, the model predicts based on probability
- Since most samples are 'Late', predicting everything as 'Late' 
  gives ~{late_pct:.0f}% accuracy

For your threshold tuning results:
- Threshold=0.10 means "predict Late if probability > 10%"
- This catches ALL late deliveries (recall=100%)
- But also mislabels ALL on-time as late (specificity=0%)

SOLUTIONS:
1. Optimize for 'balanced_accuracy' instead of 'f1'
2. Use a higher minimum threshold (e.g., 0.3-0.7 range)
3. Focus on precision-recall trade-off for business needs
""")
else:
    print(f"""
The MAJORITY class is 'On-Time' ({ontime_pct:.1f}%).

Standard approaches should work. Consider:
1. Using class_weight='balanced' in models
2. Optimizing threshold for your business metric
""")

print("=" * 60)