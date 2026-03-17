import pandas as pd
from pathlib import Path

CSV_PATH = Path(r"C:\Users\galag\GitHub\galago_audio_project\predictions_top7.csv")

df = pd.read_csv(CSV_PATH)

print("\nTotal files:", len(df))

print("\nPrediction confidence summary:")
print(df["predicted_prob"].describe())

print("\nCounts by predicted species:")
print(df["predicted_species"].value_counts())

# How many would be rejected at different thresholds?
for thr in [0.5, 0.6, 0.7, 0.8]:
    n_uncertain = (df["predicted_prob"] < thr).sum()
    print(f"Below {thr:.1f}: {n_uncertain} files ({n_uncertain/len(df)*100:.1f}%)")
