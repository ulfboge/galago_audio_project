"""
Train 16-class species classifier with improved balanced sampling.
This version:
1. Includes ALL species (no minimum sample requirement)
2. Uses balanced sampling for better representation of rare species
3. Improved class weights
4. Better architecture for 16 classes
"""
import sys
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

# REMOVED: MIN_SAMPLES_PER_CLASS - now include ALL species regardless of count

def _parse_args() -> tuple[int, str]:
    """
    Minimal CLI:
      --epochs N   (override default EPOCHS)
      --tag TAG    (suffix for saved model/curve filenames)
    """
    args = sys.argv[1:]
    epochs = EPOCHS
    tag = ""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--epochs":
            epochs = int(args[i + 1])
            i += 2
            continue
        if a == "--tag":
            tag = str(args[i + 1])
            i += 2
            continue
        raise SystemExit(f"Unknown arg: {a}")
    return epochs, tag


EPOCHS, RUN_TAG = _parse_args()
RUN_TAG_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""

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
        continue  # Skip negative class
    
    png_files = list(species_dir.glob("*.png"))
    count = len(png_files)
    
    if count > 0:  # Include any species with at least 1 sample
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
# 2. Create balanced dataset using manual sampling
# -------------------------------------------------------------------
print(f"\n{'='*60}")
print("Creating balanced dataset...")

# Collect all files per species
species_files = defaultdict(list)
for species_dir in sorted(MELS_DIR.iterdir()):
    if not species_dir.is_dir() or species_dir.name == "not_galago":
        continue
    if species_dir.name not in available_species:
        continue
    
    png_files = list(species_dir.glob("*.png"))
    species_files[species_dir.name] = png_files

# Calculate target samples per species (use median to avoid extreme oversampling)
sample_counts = [len(files) for files in species_files.values()]
median_samples = int(np.median(sample_counts))
target_samples = min(median_samples, max(sample_counts) // 2)  # Cap at half of max
print(f"  Target samples per species: {target_samples} (median: {median_samples})")

# Create balanced file lists
balanced_files = []
balanced_labels = []
label_to_idx = {species: idx for idx, species in enumerate(available_species)}

for species in available_species:
    files = species_files[species]
    label_idx = label_to_idx[species]
    
    # If species has fewer samples, repeat them (with augmentation in mind)
    # If species has more, randomly sample
    if len(files) < target_samples:
        # Repeat files to reach target
        repeats = (target_samples // len(files)) + 1
        selected = (files * repeats)[:target_samples]
    else:
        # Randomly sample
        np.random.seed(SEED)
        selected = np.random.choice(files, size=target_samples, replace=False).tolist()
    
    balanced_files.extend(selected)
    balanced_labels.extend([label_idx] * len(selected))
    print(f"  {species}: {len(selected)} samples selected (from {len(files)} available)")

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
    # Balanced weight
    class_weights[cls_idx] = total_train / (num_classes * count)

print(f"\nClass weights:")
for cls_idx in range(num_classes):
    species_name = class_names[cls_idx]
    count = train_label_counts.get(cls_idx, 0)
    weight = class_weights[cls_idx]
    print(f"  {cls_idx:2d} {species_name:30s}: {count:4d} samples (weight: {weight:.3f})")

# -------------------------------------------------------------------
# 5. Build model (same architecture as before, but optimized)
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

model = keras.Model(inputs, outputs, name=f"galago_cnn_all_{num_classes}classes_balanced")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------
# 6. Callbacks
# -------------------------------------------------------------------
checkpoint_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_balanced{RUN_TAG_SUFFIX}_best.keras"

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

final_model_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_balanced{RUN_TAG_SUFFIX}_final.keras"
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
plt.title(f"Loss ({num_classes} classes, balanced)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict["accuracy"], "b-", label="Training acc")
plt.plot(epochs, history_dict["val_accuracy"], "r-", label="Validation acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Accuracy ({num_classes} classes, balanced)")
plt.grid(True)

plt.tight_layout()
training_curves_path = OUT_DIR / f"training_curves_all_{num_classes}classes_balanced{RUN_TAG_SUFFIX}.png"
plt.savefig(training_curves_path, dpi=150)
plt.close()
print(f"Saved training curves to: {training_curves_path}")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"  Model: {num_classes} species with balanced sampling")
print(f"  Validation accuracy: {val_acc:.3f}")
print(f"{'='*60}")
