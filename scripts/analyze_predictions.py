"""Quick script to analyze prediction results."""
import pandas as pd
from pathlib import Path

csv_path = Path("outputs/predictions/predictions_top7_windowed_fixed.csv")
df = pd.read_csv(csv_path)

print("=" * 80)
print("PREDICTION ANALYSIS")
print("=" * 80)
print(f"\nTotal files: {len(df)}")
print(f"\nPrediction distribution:")
print(df['predicted_species'].value_counts())

print(f"\n\nTop 15 predictions (by confidence):")
print("-" * 80)
top15 = df.nlargest(15, 'predicted_prob')
for idx, row in top15.iterrows():
    filename = Path(row['filepath']).name
    print(f"\n{filename}")
    print(f"  Source: {row['source_folder']}")
    print(f"  Top1: {row['predicted_species']} ({row['predicted_prob']})")
    print(f"  Top2: {row['top2_species']} ({row['top2_prob']})")
    print(f"  Top3: {row['top3_species']} ({row['top3_prob']})")

print(f"\n\nConfidence statistics:")
print(f"  Mean: {df['predicted_prob'].astype(float).mean():.3f}")
print(f"  Max:  {df['predicted_prob'].astype(float).max():.3f}")
print(f"  Min:  {df['predicted_prob'].astype(float).min():.3f}")

print(f"\n\nPredictions by source folder:")
print("-" * 80)
for folder in df['source_folder'].unique():
    folder_df = df[df['source_folder'] == folder]
    print(f"\n{folder} ({len(folder_df)} files):")
    # Show top predictions for this folder
    top_preds = folder_df['predicted_species'].value_counts().head(3)
    for pred, count in top_preds.items():
        print(f"  {pred}: {count}")

