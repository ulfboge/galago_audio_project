import csv
from pathlib import Path

csv_path = Path("outputs/predictions/predictions_all_species.csv")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]  # Skip header

probs = []
for row in rows:
    if row[5] != 'uncertain':
        probs.append(float(row[5]))
    else:
        probs.append(0.0)

print(f"Improved 16-class model:")
print(f"  Mean confidence: {sum(probs)/len(probs):.3f}")
print(f"  Max confidence: {max(probs):.3f}")
print(f"  Min confidence: {min(probs):.3f}")
print(f"  Predictions > 0.6: {sum(1 for p in probs if p>=0.6)}/{len(probs)}")
print(f"  Total predictions: {len(rows)}")
print(f"  Uncertain: {sum(1 for row in rows if row[5] == 'uncertain')}")

