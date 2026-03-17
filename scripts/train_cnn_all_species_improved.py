"""
Train 19-class species classifier with improved balanced sampling.
This version:
1. Uses weighted sampling instead of repetition
2. Caps oversampling to avoid extreme repetition
3. Uses class weights in loss function
4. Better handles underrepresented species
"""
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

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
OUT_DIR = PROJECT_ROOT / "models" / "all_species"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 50

# Maximum oversampling factor (don't repeat samples more than this)
MAX_OVERSAMPLE_FACTOR = 3  # Cap at 3x repetition

# Optional: boost target samples for species that need more exposure (e.g. weak in evaluation).
# Keys must match folder names under MELS_DIR. Value = multiplier (e.g. 1.5 = 50% more samples).
SPECIES_OVERSAMPLE_BOOST = {
    "Galagoides_sp_nov": 1.5,
    "Paragalago_orinus": 1.5,
}

# -------------------------------------------------------------------
# 1. Find all species (include ALL, no minimum)
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
    
    if count > 0:
        available_species.append(species_dir.name)
        species_counts[species_dir.name] = count
        print(f"  [INCLUDE] {species_dir.name}: {count} samples")

available_species.sort()

print(f"\nFound {len(available_species)} species (all included):")
for species in available_species:
    print(f"  - {species}: {species_counts[species]} samples")

if len(available_species) == 0:
    raise ValueError("No species found!")

# -------------------------------------------------------------------
# 2. Create balanced dataset with capped oversampling
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Creating balanced dataset with capped oversampling...")

# Collect all files per species
species_files = defaultdict(list)
for species_dir in sorted(MELS_DIR.iterdir()):
    if not species_dir.is_dir() or species_dir.name == "not_galago":
        continue
    if species_dir.name not in available_species:
        continue
    
    png_files = list(species_dir.glob("*.png"))
    species_files[species_dir.name] = png_files

# Calculate target samples per species
# Use 75th percentile to avoid extreme oversampling
sample_counts = [len(files) for files in species_files.values()]
percentile_75 = int(np.percentile(sample_counts, 75))
target_samples = min(percentile_75, max(sample_counts) // 3)  # Cap at 1/3 of max
print(f"  Target samples per species: {target_samples} (75th percentile: {percentile_75})")

# Create balanced file lists with capped oversampling
balanced_files = []
balanced_labels = []
label_to_idx = {species: idx for idx, species in enumerate(available_species)}

for species in available_species:
    files = species_files[species]
    label_idx = label_to_idx[species]
    n_available = len(files)
    
    if n_available == 0:
        continue
    
    # Per-species target (boost for problem species if set)
    boost = SPECIES_OVERSAMPLE_BOOST.get(species, 1.0)
    species_target = max(1, int(target_samples * boost))
    
    # Calculate how many samples we need
    if n_available >= species_target:
        # Enough samples - randomly select
        np.random.seed(SEED)
        selected = np.random.choice(files, size=species_target, replace=False).tolist()
    else:
        # Need to oversample, but cap the repetition
        oversample_factor = min(species_target // n_available, MAX_OVERSAMPLE_FACTOR)
        n_needed = min(species_target, n_available * oversample_factor)
        
        # Repeat files up to MAX_OVERSAMPLE_FACTOR times
        selected = []
        for rep in range(oversample_factor):
            selected.extend(files)
        # If still need more, randomly sample with replacement
        if len(selected) < n_needed:
            np.random.seed(SEED)
            additional = np.random.choice(files, size=n_needed - len(selected), replace=True).tolist()
            selected.extend(additional)
        selected = selected[:n_needed]
    
    balanced_files.extend(selected)
    balanced_labels.extend([label_idx] * len(selected))
    
    oversample_ratio = len(selected) / n_available if n_available > 0 else 0
    boost_note = f" (boost {boost}x)" if boost != 1.0 else ""
    print(f"  {species}: {len(selected)} samples (from {n_available}, {oversample_ratio:.1f}x){boost_note}")

# Shuffle
indices = np.arange(len(balanced_files))
np.random.seed(SEED)
np.random.shuffle(indices)
balanced_files = [balanced_files[i] for i in indices]
balanced_labels = [balanced_labels[i] for i in indices]

# Convert Path objects to strings for TensorFlow
balanced_files = [str(f) for f in balanced_files]

# Split train/val
split_idx = int(len(balanced_files) * (1 - VAL_SPLIT))
train_files = balanced_files[:split_idx]
train_labels = balanced_labels[:split_idx]
val_files = balanced_files[split_idx:]
val_labels = balanced_labels[split_idx:]

print(f"\nDataset split:")
print(f"  Training: {len(train_files)} samples")
print(f"  Validation: {len(val_files)} samples")

# -------------------------------------------------------------------
# 3. Create TensorFlow datasets from file paths
# -------------------------------------------------------------------
def load_and_preprocess_image(file_path, label):
    """Load and preprocess image from file path."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(
    lambda x, y: load_and_preprocess_image(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_dataset = val_dataset.map(
    lambda x, y: load_and_preprocess_image(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_classes = len(available_species)
class_names = available_species

print(f"\nClasses ({num_classes}):")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# -------------------------------------------------------------------
# 4. Compute class weights (for loss function)
# -------------------------------------------------------------------
train_label_counts = Counter(train_labels)
total_train = len(train_labels)
class_weights = {}
for cls_idx in range(num_classes):
    count = train_label_counts.get(cls_idx, 1)
    # Balanced weight - stronger weighting for underrepresented classes
    class_weights[cls_idx] = total_train / (num_classes * count)

print(f"\nClass weights:")
for cls_idx in range(num_classes):
    species_name = class_names[cls_idx]
    count = train_label_counts.get(cls_idx, 0)
    weight = class_weights[cls_idx]
    print(f"  {cls_idx:2d} {species_name:30s}: {count:4d} samples (weight: {weight:.3f})")

# -------------------------------------------------------------------
# 5. Build model
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Building CNN model...")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# IMPORTANT: images are already normalized to [0,1] in load_and_preprocess_image()
# (tf.cast(...)/255.0). Do NOT rescale again here (double-normalization kills learning).
x = inputs

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

model = keras.Model(inputs, outputs, name=f"galago_cnn_all_{num_classes}classes_improved")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------
# 6. Callbacks
# -------------------------------------------------------------------
checkpoint_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_improved_best.keras"

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
# 7. Train
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Starting training for {EPOCHS} epochs...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Classes: {num_classes}")
print(f"  Training samples: {len(train_files)}")
print(f"  Validation samples: {len(val_files)}")
print(f"  Max oversample factor: {MAX_OVERSAMPLE_FACTOR}x")
print(f"{'='*60}\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

# -------------------------------------------------------------------
# 8. Evaluate and save
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Evaluating model...")

val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
print(f"Validation accuracy ({num_classes} classes): {val_acc:.3f} ({val_acc*100:.1f}%)")

final_model_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_improved_final.keras"
model.save(final_model_path)
print(f"Saved final model to: {final_model_path}")

# Save class names (versioned; do not overwrite the repo-wide class_names.json)
class_names_path = OUT_DIR / f"class_names_{num_classes}.json"
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"Saved class names to: {class_names_path}")

# -------------------------------------------------------------------
# 9. Plot training curves
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
plt.title(f"Loss ({num_classes} classes, improved)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict["accuracy"], "b-", label="Training acc")
plt.plot(epochs, history_dict["val_accuracy"], "r-", label="Validation acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Accuracy ({num_classes} classes, improved)")
plt.grid(True)

plt.tight_layout()
training_curves_path = OUT_DIR / f"training_curves_all_{num_classes}classes_improved.png"
plt.savefig(training_curves_path, dpi=150)
plt.close()
print(f"Saved training curves to: {training_curves_path}")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"  Model: {num_classes} species with improved balanced sampling")
print(f"  Validation accuracy: {val_acc:.3f}")
print(f"{'='*60}")
