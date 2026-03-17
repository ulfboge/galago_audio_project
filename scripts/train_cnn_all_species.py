from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# PATHS – adjust if needed
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELS_DIR = PROJECT_ROOT / "data" / "melspectrograms"
OUT_DIR = PROJECT_ROOT / "models" / "all_species"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 50  # can lower/raise later

# Minimum number of samples per species to include in training
# Species with fewer samples will be excluded
# Increased from 10 to 30 to improve data balance
MIN_SAMPLES_PER_CLASS = 30

# -------------------------------------------------------------------
# 0. Auto-detect available species from directory structure
# -------------------------------------------------------------------
print("Scanning for available species in:", MELS_DIR)

available_species = []
species_counts = {}

for species_dir in sorted(MELS_DIR.iterdir()):
    if not species_dir.is_dir():
        continue
    
    # Count PNG files in this species directory
    png_files = list(species_dir.glob("*.png"))
    count = len(png_files)
    
    if count >= MIN_SAMPLES_PER_CLASS:
        available_species.append(species_dir.name)
        species_counts[species_dir.name] = count
    else:
        print(f"  [SKIP] {species_dir.name}: {count} samples (below minimum of {MIN_SAMPLES_PER_CLASS})")

# Sort species alphabetically for consistent ordering
available_species.sort()

print(f"\nFound {len(available_species)} species with sufficient samples:")
for species in available_species:
    print(f"  - {species}: {species_counts[species]} samples")

if len(available_species) == 0:
    raise ValueError(f"No species found with at least {MIN_SAMPLES_PER_CLASS} samples!")

# -------------------------------------------------------------------
# 1. Load datasets from folder structure (using all detected species)
# -------------------------------------------------------------------
print(f"\nLoading datasets with {len(available_species)} classes...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    MELS_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",   # PNGs saved via matplotlib -> RGB
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    class_names=available_species,  # Use all detected species
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
    print(f"  {i}: {name}")

# -------------------------------------------------------------------
# 2. Performance tweaks
# -------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------------------------------------------------
# 3. Compute class weights to handle imbalance (improved formula)
# -------------------------------------------------------------------
def compute_class_weights(dataset, num_classes):
    """
    Compute class weights using balanced formula.
    Weights are inversely proportional to class frequency.
    """
    counts = Counter()
    for _, labels in dataset.unbatch():
        counts[int(labels.numpy())] += 1

    total = sum(counts.values())
    class_weights = {}
    for cls in range(num_classes):
        n = counts.get(cls, 1)
        # Balanced weight: total / (num_classes * n)
        # This gives more weight to underrepresented classes
        class_weights[cls] = total / (num_classes * n)
    
    # Don't normalize - keep original weights
    return class_weights, counts

class_weights, counts = compute_class_weights(train_ds, num_classes)
print("\nTraining set class distribution:")
for cls_idx, count in sorted(counts.items()):
    species_name = class_names[cls_idx]
    weight = class_weights[cls_idx]
    print(f"  {cls_idx:2d} {species_name:30s}: {count:4d} samples (weight: {weight:.3f})")

# -------------------------------------------------------------------
# 4. Define CNN model (increased capacity for 17 classes)
# -------------------------------------------------------------------
print("\nBuilding CNN model with increased capacity...")
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Rescaling(1.0 / 255)(inputs)

# Increased filters: 64 -> 128 -> 256 -> 512 (was 32 -> 64 -> 128 -> 256)
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
# Increased dense layer: 512 units (was 256) for better capacity
# But removed the extra layer to avoid overfitting
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name=f"galago_call_cnn_all_{num_classes}classes")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),  # Reduced from 1e-3 to 5e-4
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------------------------
# 5. Callbacks
# -------------------------------------------------------------------
checkpoint_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_best.keras"

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
    ),
]

# -------------------------------------------------------------------
# 6. Train
# -------------------------------------------------------------------
print(f"\nStarting training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# -------------------------------------------------------------------
# 7. Evaluate and save
# -------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds)
print(f"\nValidation accuracy ({num_classes} classes): {val_acc:.3f}")

final_model_path = OUT_DIR / f"galago_cnn_all_{num_classes}classes_final.keras"
model.save(final_model_path)
print("Saved final model to:", final_model_path)

# Save class names for later use
import json
class_names_path = OUT_DIR / "class_names.json"
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=2)
print("Saved class names to:", class_names_path)

# -------------------------------------------------------------------
# 8. Plot training curves
# -------------------------------------------------------------------
history_dict = history.history
epochs = range(1, len(history_dict["loss"]) + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict["loss"], "b-", label="Training loss")
plt.plot(epochs, history_dict["val_loss"], "r-", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Loss ({num_classes} classes)")

plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict["accuracy"], "b-", label="Training acc")
plt.plot(epochs, history_dict["val_accuracy"], "r-", label="Validation acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Accuracy ({num_classes} classes)")

plt.tight_layout()
training_curves_path = OUT_DIR / f"training_curves_all_{num_classes}classes.png"
plt.savefig(training_curves_path, dpi=150)
plt.close()
print("Saved training curves to:", training_curves_path)

print(f"\nTraining complete! Model trained on {num_classes} species.")

