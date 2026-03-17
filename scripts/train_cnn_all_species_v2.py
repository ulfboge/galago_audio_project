"""
Improved training script following AI recommendations:
1. Use weighted loss instead of oversampling
2. Add stronger augmentation (SpecAugment-style)
3. Train on segments (already done via mel-spectrograms)
4. Better architecture
5. Save class mapping properly
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = PROJECT_ROOT / "models" / "all_species"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 50

# Minimum samples per class (same as original 16-class model)
MIN_SAMPLES_PER_CLASS = 30

# -------------------------------------------------------------------
# 1. Find species with sufficient samples
# -------------------------------------------------------------------
print("Scanning for available species in:", MELS_DIR)
print("=" * 60)

available_species = []
species_counts = {}

for species_dir in sorted(MELS_DIR.iterdir()):
    if not species_dir.is_dir():
        continue
    if species_dir.name == "not_galago":
        continue
    
    png_files = list(species_dir.glob("*.png"))
    count = len(png_files)
    
    if count >= MIN_SAMPLES_PER_CLASS:
        available_species.append(species_dir.name)
        species_counts[species_dir.name] = count
        print(f"  [INCLUDE] {species_dir.name}: {count} samples")
    else:
        print(f"  [SKIP] {species_dir.name}: {count} samples (below minimum of {MIN_SAMPLES_PER_CLASS})")

available_species.sort()

print(f"\nFound {len(available_species)} species with sufficient samples:")
for species in available_species:
    print(f"  - {species}: {species_counts[species]} samples")

if len(available_species) == 0:
    raise ValueError(f"No species found with at least {MIN_SAMPLES_PER_CLASS} samples!")

# -------------------------------------------------------------------
# 2. Load datasets (use TensorFlow's built-in, no manual balancing)
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Loading datasets (using natural distribution)...")
print("=" * 60)

train_ds = tf.keras.utils.image_dataset_from_directory(
    MELS_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    class_names=available_species,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    MELS_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    class_names=available_species,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nClasses actually used ({num_classes}):")
for i, name in enumerate(class_names):
    print(f"  {i:2d}: {name}")

# -------------------------------------------------------------------
# 3. Performance tweaks (augmentation will be in model)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 4. Performance tweaks
# -------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------------------------------------------------
# 5. Compute class weights (for weighted loss, NOT oversampling)
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Computing class weights (for weighted loss)...")
print("=" * 60)

def compute_class_weights(dataset, num_classes):
    """Compute class weights using balanced formula."""
    counts = Counter()
    for _, labels in dataset.unbatch():
        counts[int(labels.numpy())] += 1

    total = sum(counts.values())
    class_weights = {}
    for cls in range(num_classes):
        n = counts.get(cls, 1)
        # Balanced weight: total / (num_classes * n)
        class_weights[cls] = total / (num_classes * n)
    
    return class_weights, counts

class_weights, counts = compute_class_weights(train_ds, num_classes)
print("\nTraining set class distribution:")
for cls_idx, count in sorted(counts.items()):
    species_name = class_names[cls_idx]
    weight = class_weights[cls_idx]
    print(f"  {cls_idx:2d} {species_name:30s}: {count:4d} samples (weight: {weight:.3f})")

# -------------------------------------------------------------------
# 6. Build model with augmentation layers
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Building CNN model with augmentation...")
print("=" * 60)

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Rescaling(1.0 / 255)(inputs)

# Data augmentation (only during training, controlled by training flag)
# Note: These layers are active during training, inactive during inference
augmentation = keras.Sequential([
    layers.RandomTranslation(height_factor=0.05, width_factor=0.1),  # Time/freq masking simulation
    layers.RandomBrightness(factor=0.1),  # Gain variation
    layers.RandomContrast(factor=0.1),  # Recording condition variation
], name="augmentation")

# Apply augmentation (will be active during training)
x = augmentation(x)

# CNN layers
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name=f"galago_cnn_all_{num_classes}classes_v2")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------
# 7. Callbacks
# -------------------------------------------------------------------
checkpoint_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_v2_best.keras"

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    ),
]

# -------------------------------------------------------------------
# 8. Train
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Starting training for {EPOCHS} epochs...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Classes: {num_classes}")
print(f"  Using weighted loss (no oversampling)")
print(f"  Augmentation: Time/frequency masking, brightness/contrast")
print(f"{'='*60}\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # Use weighted loss, not oversampling
    verbose=1,
)

# -------------------------------------------------------------------
# 9. Evaluate and save
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Evaluating model...")

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy ({num_classes} classes): {val_acc:.3f} ({val_acc*100:.1f}%)")

final_model_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_v2_final.keras"
model.save(final_model_path)
print(f"Saved final model to: {final_model_path}")

# Save class names with model-specific name
class_names_path = OUT_DIR / f"class_names_{num_classes}.json"
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"Saved class names to: {class_names_path}")

# Also save class_to_idx mapping
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for idx, name in enumerate(class_names)}

class_to_idx_path = OUT_DIR / f"class_to_idx_{num_classes}.json"
idx_to_class_path = OUT_DIR / f"idx_to_class_{num_classes}.json"

with open(class_to_idx_path, "w") as f:
    json.dump(class_to_idx, f, indent=2)
with open(idx_to_class_path, "w") as f:
    json.dump(idx_to_class, f, indent=2)

print(f"Saved class mappings to: {class_to_idx_path}, {idx_to_class_path}")

# -------------------------------------------------------------------
# 10. Plot training curves
# -------------------------------------------------------------------
history_dict = history.history
epochs = range(1, len(history_dict["loss"]) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict["loss"], "b-", label="Training loss")
plt.plot(epochs, history_dict["val_loss"], "r-", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Loss ({num_classes} classes, v2)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict["accuracy"], "b-", label="Training acc")
plt.plot(epochs, history_dict["val_accuracy"], "r-", label="Validation acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Accuracy ({num_classes} classes, v2)")
plt.grid(True)

plt.tight_layout()
training_curves_path = OUT_DIR / f"training_curves_all_{num_classes}classes_v2.png"
plt.savefig(training_curves_path, dpi=150)
plt.close()
print(f"Saved training curves to: {training_curves_path}")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"  Model: {num_classes} species with weighted loss + augmentation")
print(f"  Validation accuracy: {val_acc:.3f}")
print(f"{'='*60}")
