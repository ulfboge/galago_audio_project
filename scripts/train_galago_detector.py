"""
Train a binary detector: galago vs not-galago.
This is Stage 1 of the Merlin-like 2-stage approach.

The detector filters out non-galago audio before species classification.
"""
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = PROJECT_ROOT / "models" / "detector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 30

# Minimum samples per class
MIN_SAMPLES_PER_CLASS = 50

# -------------------------------------------------------------------
# 1. Prepare dataset: galago (positive) vs not_galago (negative)
# -------------------------------------------------------------------
print("Preparing dataset for binary detector...")
print("=" * 60)

# Positive class: all galago species
# NOTE: Previously we required MIN_SAMPLES_PER_CLASS (50) to include a species.
# That excluded underrepresented taxa like Otolemur and Galagoides_sp_nov from
# the detector training entirely, causing the detector to reject them.
# We now include ALL species folders (except 'not_galago') regardless of count.
galago_species_dirs = []
for species_dir in sorted(MELS_DIR.iterdir()):
    if not species_dir.is_dir():
        continue
    if species_dir.name == "not_galago":
        continue  # Skip negative class folder
    
    png_files = list(species_dir.glob("*.png"))
    if len(png_files) == 0:
        continue  # Skip empty species folders
    
    galago_species_dirs.append(species_dir)
    print(f"  [GALAGO] {species_dir.name}: {len(png_files)} samples")

# Negative class: not_galago folder
not_galago_dir = MELS_DIR / "not_galago"
if not_galago_dir.exists():
    not_galago_files = list(not_galago_dir.glob("*.png"))
    print(f"  [NOT GALAGO] {not_galago_dir.name}: {len(not_galago_files)} samples")
else:
    print(f"  [WARNING] {not_galago_dir} not found!")
    print(f"  Please create negative class data first.")
    print(f"  See docs/merlin_like_roadmap.md for instructions.")
    raise FileNotFoundError(f"Negative class folder not found: {not_galago_dir}")

# Check if we have enough data
if len(galago_species_dirs) == 0:
    raise ValueError("No galago species found with sufficient samples!")

if len(not_galago_files) < MIN_SAMPLES_PER_CLASS:
    print(f"  [WARNING] Only {len(not_galago_files)} negative samples (minimum: {MIN_SAMPLES_PER_CLASS})")
    print(f"  Consider collecting more negative class data.")

# -------------------------------------------------------------------
# 2. Create temporary structure for binary classification
# -------------------------------------------------------------------
# We need to create a temporary directory structure:
#   detector_data/
#     galago/        (all galago species combined)
#     not_galago/   (negative class)

TEMP_DIR = PROJECT_ROOT / "data" / "detector_temp"
TEMP_GALAGO_DIR = TEMP_DIR / "galago"
TEMP_NOT_GALAGO_DIR = TEMP_DIR / "not_galago"

# Clean and create temp directories
if TEMP_DIR.exists():
    import shutil
    shutil.rmtree(TEMP_DIR)
TEMP_GALAGO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_NOT_GALAGO_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nCreating temporary dataset structure...")
print(f"  Copying galago samples...")

# Copy all galago samples to galago/ folder
galago_count = 0
for species_dir in galago_species_dirs:
    for png_file in species_dir.glob("*.png"):
        # Create unique name to avoid collisions
        new_name = f"{species_dir.name}_{png_file.name}"
        dest = TEMP_GALAGO_DIR / new_name
        dest.write_bytes(png_file.read_bytes())
        galago_count += 1

print(f"    Copied {galago_count} galago samples")

# Copy negative samples
print(f"  Copying not_galago samples...")
not_galago_count = 0
for png_file in not_galago_dir.glob("*.png"):
    dest = TEMP_NOT_GALAGO_DIR / png_file.name
    dest.write_bytes(png_file.read_bytes())
    not_galago_count += 1

print(f"    Copied {not_galago_count} not_galago samples")

print(f"\nDataset summary:")
print(f"  Galago (positive): {galago_count} samples")
print(f"  Not galago (negative): {not_galago_count} samples")
print(f"  Total: {galago_count + not_galago_count} samples")

# -------------------------------------------------------------------
# 3. Load datasets
# -------------------------------------------------------------------
print(f"\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TEMP_DIR,
    labels="inferred",
    label_mode="binary",  # Binary classification
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEMP_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
)

class_names = train_ds.class_names
print(f"  Classes: {class_names}")
print(f"  Training samples: {len(train_ds) * BATCH_SIZE}")
print(f"  Validation samples: {len(val_ds) * BATCH_SIZE}")

# -------------------------------------------------------------------
# 4. Calculate class weights (handle imbalance)
# -------------------------------------------------------------------
# Count samples per class
galago_train = sum(1 for _ in train_ds.unbatch() if _[1].numpy() == 1.0)
not_galago_train = sum(1 for _ in train_ds.unbatch() if _[1].numpy() == 0.0)

total = galago_train + not_galago_train
class_weights = {
    0: total / (2 * not_galago_train) if not_galago_train > 0 else 1.0,  # not_galago
    1: total / (2 * galago_train) if galago_train > 0 else 1.0,        # galago
}

print(f"\nClass weights: {class_weights}")

# -------------------------------------------------------------------
# 5. Build model (smaller than species classifier)
# -------------------------------------------------------------------
print(f"\nBuilding detector model...")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Rescaling(1.0 / 255)(inputs)

# Smaller CNN than species classifier
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # Binary output

model = keras.Model(inputs, outputs, name="galago_detector")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall"],
)

print(f"  Model parameters: {model.count_params():,}")

# -------------------------------------------------------------------
# 6. Training callbacks
# -------------------------------------------------------------------
callbacks = [
    keras.callbacks.ModelCheckpoint(
        OUT_DIR / "galago_detector_best.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    ),
]

# -------------------------------------------------------------------
# 7. Train
# -------------------------------------------------------------------
print(f"\nTraining detector model...")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Validation split: {VAL_SPLIT}")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

# -------------------------------------------------------------------
# 8. Evaluate
# -------------------------------------------------------------------
print(f"\nEvaluating model...")
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_ds, verbose=0)

print(f"\nValidation metrics:")
print(f"  Accuracy: {val_acc:.3f}")
print(f"  Precision: {val_precision:.3f}")
print(f"  Recall: {val_recall:.3f}")
print(f"  F1-score: {2 * (val_precision * val_recall) / (val_precision + val_recall):.3f}")

# -------------------------------------------------------------------
# 9. Save model and metadata
# -------------------------------------------------------------------
print(f"\nSaving model and metadata...")

# Save final model
final_model_path = OUT_DIR / "galago_detector_final.keras"
model.save(final_model_path)
print(f"  Saved: {final_model_path}")

# Save metadata
metadata = {
    "model_type": "binary_detector",
    "classes": class_names,
    "training_samples": {
        "galago": galago_count,
        "not_galago": not_galago_count,
    },
    "validation_metrics": {
        "accuracy": float(val_acc),
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1_score": float(2 * (val_precision * val_recall) / (val_precision + val_recall)),
    },
    "architecture": {
        "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
        "parameters": int(model.count_params()),
    },
}

metadata_path = OUT_DIR / "detector_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved: {metadata_path}")

# -------------------------------------------------------------------
# 10. Plot training curves
# -------------------------------------------------------------------
print(f"\nPlotting training curves...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy
axes[0, 0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0, 0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(history.history["loss"], label="Train Loss")
axes[0, 1].plot(history.history["val_loss"], label="Val Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision
axes[1, 0].plot(history.history["precision"], label="Train Precision")
axes[1, 0].plot(history.history["val_precision"], label="Val Precision")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].set_title("Precision")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Recall
axes[1, 1].plot(history.history["recall"], label="Train Recall")
axes[1, 1].plot(history.history["val_recall"], label="Val Recall")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Recall")
axes[1, 1].set_title("Recall")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
curve_path = OUT_DIR / "training_curves_detector.png"
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"  Saved: {curve_path}")

# -------------------------------------------------------------------
# 11. Cleanup
# -------------------------------------------------------------------
print(f"\nCleaning up temporary files...")
import shutil
shutil.rmtree(TEMP_DIR)
print(f"  Removed: {TEMP_DIR}")

print(f"\n{'='*60}")
print(f"Detector training complete!")
print(f"{'='*60}")
print(f"\nModel saved to: {OUT_DIR}")
print(f"\nNext steps:")
print(f"  1. Test detector on mixed audio (galago + non-galago)")
print(f"  2. Integrate detector into prediction pipeline")
print(f"  3. Adjust detection threshold based on false positive rate")

